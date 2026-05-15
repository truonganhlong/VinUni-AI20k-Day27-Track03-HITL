"""LLM factory. Returns an OpenAI chat model."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set - copy .env.example to .env")

    kwargs = {}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
