from __future__ import annotations
import operator
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Source = Literal["email", "site", "web", "sql"]


def append_or_reset(left: list | None, right: list | None) -> list:
    """
    Append reducer that can be cleared: an update of None resets the field.
    Required because operator.add would keep accumulating across turns of the same thread.
    """
    if right is None:
        return []
    return (left or []) + right


class Step(BaseModel):
    """One retrieval action planned by the understand node."""
    id: str = Field(description="short id, e.g. 's1'")
    source: Source
    query: str = Field(description="self-contained sub-question for this source")
    sender: Optional[str] = None           # email only
    sent_on: Optional[str] = None          # email only, YYYY-MM-DD
    sent_after: Optional[str] = None       # email only
    sent_before: Optional[str] = None      # email only
    depends_on: Optional[str] = Field(
        None, description="id of a step whose results this step needs"
    )


class Plan(BaseModel):
    standalone_query: str
    intent: Literal["search", "chitchat", "clarify"]
    clarification_question: Optional[str] = None
    steps: list[Step] = []


class Evidence(BaseModel):
    id: str                                # "email:e06", "site:3", "web:s1:2", "sql:s2"
    source: Source
    step_id: str
    title: str = ""
    text: str
    url: Optional[str] = None
    score: Optional[float] = None


class ToolResult(BaseModel):
    step_id: str
    source: Source
    status: Literal["ok", "empty", "error", "needs_user"]
    error_kind: Optional[str] = None
    error: Optional[str] = None
    latency_ms: int = 0
    detail: dict = {}


class TraceEvent(BaseModel):
    node: str
    ms: int
    llm_calls: int = 0
    info: dict = {}


class GraphState(TypedDict, total=False):
    # thread-scoped (persisted, accumulate across turns)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    schema_cache: dict[str, str]
    # turn-scoped (reset at the start of every turn)
    query: str
    date_hints: dict[str, str]
    needs_summary: bool
    plan: Optional[Plan]
    wave: int
    pending_steps: list[Step]
    evidence: Annotated[list[Evidence], append_or_reset]
    results: Annotated[list[ToolResult], append_or_reset]
    trace: Annotated[list[TraceEvent], append_or_reset]
    answer: str
    citations: list[str]
    status: Literal["answered", "needs_clarification", "error"]


def turn_input(query: str) -> dict:
    """Build the initial state dict for a new turn, resetting all turn-scoped fields."""
    return {
        "query": query,
        "messages": [HumanMessage(query)],
        "plan": None,
        "wave": 0,
        "pending_steps": [],
        "evidence": None,
        "results": None,
        "trace": None,
        "answer": "",
        "citations": [],
        "date_hints": {},
        "status": "answered",
    }
