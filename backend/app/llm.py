from __future__ import annotations
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLM(Protocol):
    async def structured(self, schema: type[T], messages: list, *, name: str) -> T: ...
    async def text(self, messages: list, *, name: str) -> str: ...


class LangChainLLM:
    """Adapter over any LangChain chat model."""

    def __init__(self, chat_model):
        self.chat = chat_model

    async def structured(self, schema: type[T], messages: list, *, name: str) -> T:
        return await self.chat.with_structured_output(schema).ainvoke(messages)

    async def text(self, messages: list, *, name: str) -> str:
        msg = await self.chat.ainvoke(messages)
        c = msg.content
        if isinstance(c, list):
            return "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in c
            )
        return str(c)


def make_llm(settings) -> LLM:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return LangChainLLM(ChatOpenAI(model=settings.openai_model, temperature=0))
    from langchain_google_genai import ChatGoogleGenerativeAI
    return LangChainLLM(
        ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
    )
