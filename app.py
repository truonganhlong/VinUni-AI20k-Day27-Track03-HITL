"""Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


async def list_recent_sessions() -> list[dict[str, Any]]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   CASE MIN(
                       CASE risk_level
                           WHEN 'high' THEN 1
                           WHEN 'med' THEN 2
                           WHEN 'low' THEN 3
                           ELSE 4
                       END
                   )
                       WHEN 1 THEN 'high'
                       WHEN 2 THEN 'med'
                       WHEN 3 THEN 'low'
                       ELSE 'unknown'
                   END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke or resume the graph once."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


def ensure_state() -> None:
    defaults = {
        "thread_id": None,
        "pr_url": "",
        "interrupt_payload": None,
        "final": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Recent sessions")
        try:
            rows = asyncio.run(list_recent_sessions())
        except Exception as exc:
            st.caption(f"No sessions yet ({exc})")
            return

        if not rows:
            st.caption("No audited sessions yet.")
            return

        for idx, row in enumerate(rows):
            label = f"{row['worst_risk']} risk | {row['events']} events"
            with st.container(border=True):
                st.caption(label)
                st.write(row["pr_url"])
                st.code(row["thread_id"], language="text")
                if st.button("Load", key=f"load_session_{idx}"):
                    st.session_state.thread_id = row["thread_id"]
                    st.session_state.pr_url = row["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()


def render_approval_card(payload: dict) -> dict | None:
    """58-72% bucket: show the LLM review + 3 buttons."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` - {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """<58% bucket: show risk factors + question form."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers = {}
        for idx, question in enumerate(payload.get("questions", [])):
            answers[question] = st.text_area(question, key=f"escalation_answer_{idx}")
        submitted = st.form_submit_button("Submit answers", type="primary")
    if submitted:
        return answers
    return None


def render_start_form() -> tuple[bool, str]:
    with st.form("start"):
        pr_url = st.text_input(
            "PR URL",
            value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
        submitted = st.form_submit_button("Run review", type="primary")
    return submitted, pr_url


def start_review(pr_url: str) -> None:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR and asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result


def resume_review(answer: dict) -> None:
    with st.spinner("Resuming graph..."):
        result = asyncio.run(run_graph(
            st.session_state.pr_url,
            st.session_state.thread_id,
            resume_value=answer,
        ))
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result
    st.rerun()


def render_final() -> None:
    final = st.session_state.final
    if final is None:
        return

    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"{action} - comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected - no comment posted")
    elif action == "commit_failed":
        st.error("Commit failed - check GitHub token permissions or network access.")
    else:
        st.info(f"final_action = {action}")

    if final.get("posted_comment_body"):
        with st.expander("Posted comment"):
            st.markdown(final["posted_comment_body"])

    st.caption(
        f"thread_id = {st.session_state.thread_id} | "
        f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )


def main() -> None:
    st.set_page_config(page_title="HITL PR Review", layout="wide")
    ensure_state()
    st.title("HITL PR Review Agent")
    render_sidebar()

    submitted, pr_url = render_start_form()
    if submitted and pr_url:
        try:
            start_review(pr_url)
        except Exception as exc:
            st.error(str(exc))

    payload = st.session_state.interrupt_payload
    if payload is not None:
        kind = payload["kind"]
        if kind == "approval_request":
            answer = render_approval_card(payload)
        elif kind == "escalation":
            answer = render_escalation_card(payload)
        else:
            st.error(f"Unknown interrupt kind: {kind}")
            answer = None
        if answer is not None:
            try:
                resume_review(answer)
            except Exception as exc:
                st.error(str(exc))

    render_final()


if __name__ == "__main__":
    main()
