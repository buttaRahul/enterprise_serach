"""preprocess node — deterministic, no LLM calls."""
from __future__ import annotations
import re
import time
from datetime import date
from typing import TYPE_CHECKING

from langchain_core.messages import AnyMessage

from app.graph.dates import extract_date_hints
from app.graph.state import GraphState, TraceEvent

if TYPE_CHECKING:
    pass

# Greetings regex: whole message ≤ 6 words, starts with a greeting word
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank\s+you|bye|good\s+(?:morning|afternoon|evening))\b",
    re.I,
)
# "what can you do" style
_META_RE = re.compile(
    r"\b(what\s+can\s+you|what\s+do\s+you|help\s+me|how\s+do\s+you\s+work)\b",
    re.I,
)

HISTORY_TOKEN_LIMIT = 3000  # overridden by settings in build.py


def _is_chitchat(query: str) -> bool:
    words = query.split()
    if len(words) <= 6 and _GREETING_RE.search(query):
        return True
    return False


def _token_estimate(messages: list[AnyMessage]) -> int:
    total = 0
    for m in messages:
        c = m.content
        if isinstance(c, str):
            total += len(c) // 4
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total += len(str(part.get("text", ""))) // 4
    return total


async def preprocess(state: GraphState) -> dict:
    t0 = time.perf_counter()
    query: str = state.get("query", "")
    messages: list = state.get("messages", [])

    chitchat = _is_chitchat(query)
    date_hints = extract_date_hints(query, today=date.today())

    needs_summary = _token_estimate(messages) > HISTORY_TOKEN_LIMIT

    ms = int((time.perf_counter() - t0) * 1000)
    event = TraceEvent(
        node="preprocess",
        ms=ms,
        info={"chitchat": chitchat, "date_hints": date_hints, "needs_summary": needs_summary},
    )
    return {"date_hints": date_hints, "needs_summary": needs_summary, "trace": [event]}


def route_after_preprocess(state: GraphState) -> str:
    query: str = state.get("query", "")
    if _is_chitchat(query):
        return "respond_direct"
    if state.get("needs_summary"):
        return "summarize"
    return "understand"
