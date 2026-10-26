"""respond_direct node — handles chit-chat and meta questions."""
from __future__ import annotations
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.prompts import RESPOND_DIRECT_PROMPT
from app.graph.state import GraphState, TraceEvent

_STATIC_GREETINGS = re.compile(
    r"^(hi|hello|hey|thanks|thank\s+you|bye|good\s+(?:morning|afternoon|evening))\b",
    re.I,
)
_STATIC_REPLY = (
    "Hello! I can search your company email, the employee and project database, "
    "the company website, and the public web. What would you like to find?"
)


def make_respond_direct(llm):
    async def respond_direct(state: GraphState) -> dict:
        t0 = time.perf_counter()
        query: str = state.get("query", "")
        llm_calls = 0

        if _STATIC_GREETINGS.match(query.strip()) and len(query.split()) <= 6:
            answer = _STATIC_REPLY
        else:
            msgs = [
                SystemMessage(content=RESPOND_DIRECT_PROMPT),
                HumanMessage(content=query),
            ]
            answer = await llm.text(msgs, name="respond_direct")
            llm_calls = 1

        ms = int((time.perf_counter() - t0) * 1000)
        event = TraceEvent(node="respond_direct", ms=ms, llm_calls=llm_calls)
        return {
            "answer": answer,
            "status": "answered",
            "messages": [AIMessage(answer)],
            "trace": [event],
        }

    return respond_direct
