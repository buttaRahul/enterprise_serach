"""clarify node — interrupt/resume for one clarification per turn."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from app.graph.state import GraphState


async def clarify(state: GraphState) -> dict:
    plan = state.get("plan")
    question = (
        plan.clarification_question
        if plan and plan.clarification_question
        else "Could you clarify what you are looking for?"
    )
    # Graph pauses here; API caller sees needs_clarification
    reply: str = interrupt({"question": question})

    combined = f"{state.get('query', '')}\n(Clarification) {question} -> {reply}"
    return {
        "query": combined,
        "messages": [AIMessage(question), HumanMessage(reply)],
    }
