"""summarize node — trim long history and replace with a compact summary."""
from __future__ import annotations
import time

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from app.graph.prompts import SUMMARIZE_PROMPT
from app.graph.state import GraphState, TraceEvent


def make_summarize(llm):
    async def summarize(state: GraphState):
        t0 = time.perf_counter()
        messages = state.get("messages", [])

        # Keep the 6 most recent messages; summarize the rest
        keep_n = 6
        old = messages[:-keep_n] if len(messages) > keep_n else []

        summary_text = ""
        if old:
            summary_text = await llm.text(
                old + [HumanMessage(content="Summarize the conversation so far in 3-5 sentences.")],
                name="summarize",
            )

        removes = [RemoveMessage(id=m.id) for m in old]
        inserts = []
        if summary_text:
            inserts = [SystemMessage(content=f"[Earlier conversation summary]: {summary_text}")]

        ms = int((time.perf_counter() - t0) * 1000)
        event = TraceEvent(node="summarize", ms=ms, llm_calls=1 if old else 0, info={"trimmed": len(old)})
        return {"messages": removes + inserts, "trace": [event]}

    return summarize
