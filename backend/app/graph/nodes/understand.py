"""understand node — LLM structured output → Plan."""
from __future__ import annotations
import re
import time
from contextlib import contextmanager
from datetime import date
from typing import Generator

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.prompts import UNDERSTAND_PROMPT
from app.graph.state import (
    GraphState, Plan, Step, ToolResult, TraceEvent,
)

MAX_STEPS = 4  # overridden at build time via closure


@contextmanager
def _traced(node: str, info: dict | None = None) -> Generator[dict, None, None]:
    t = time.perf_counter()
    box: dict = {"llm_calls": 0, "info": info or {}}
    yield box
    box["event"] = TraceEvent(
        node=node,
        ms=int((time.perf_counter() - t) * 1000),
        llm_calls=box["llm_calls"],
        info=box["info"],
    )


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_date(s: str | None) -> bool:
    return bool(s and _DATE_RE.match(s))


def _render_date_hints(hints: dict) -> str:
    if not hints or not hints.get("dates"):
        return ""
    parts = []
    for d, r in zip(hints["dates"], hints["relations"]):
        parts.append(f"{r} {d}")
    return "Detected dates: " + ", ".join(parts)


SOURCE_NODE = {
    "email": "email_retrieve",
    "site": "site_retrieve",
    "web": "web_retrieve",
    "sql": "sql_answer",
}


def make_understand(llm, services=None, max_steps: int = MAX_STEPS):
    available_sources = _available_sources(services)

    async def understand(state: GraphState) -> dict:
        with _traced("understand") as tb:
            query: str = state.get("query", "")
            summary: str = state.get("summary", "")
            messages: list = state.get("messages", [])
            date_hints: dict = state.get("date_hints", {})

            today = date.today().isoformat()
            sources_str = ", ".join(available_sources) if available_sources else "email, sql, site, web"

            system_content = UNDERSTAND_PROMPT.format(today=today, sources=sources_str)
            context_msgs = []
            if summary:
                context_msgs.append(SystemMessage(content=f"Conversation summary: {summary}"))
            context_msgs.extend(messages[-6:])

            hint_str = _render_date_hints(date_hints)
            user_content = query
            if hint_str:
                user_content += f"\n{hint_str}"

            llm_messages = [SystemMessage(content=system_content)] + context_msgs + [HumanMessage(content=user_content)]

            plan: Plan = await llm.structured(Plan, llm_messages, name="understand")
            tb["llm_calls"] += 1

            # Post-process: drop unavailable sources
            plan = _filter_steps(plan, available_sources, max_steps)

            # Validate dates; bad dates become error ToolResults dropped from plan
            plan, bad_results = _validate_dates(plan)

            # Split into wave-0 (no depends_on) and pending
            wave0 = [s for s in plan.steps if not s.depends_on]
            pending = [s for s in plan.steps if s.depends_on]

            tb["info"]["intent"] = plan.intent
            tb["info"]["steps"] = len(plan.steps)

        updates: dict = {
            "plan": plan,
            "pending_steps": pending,
            "trace": [tb["event"]],
        }
        if bad_results:
            updates["results"] = bad_results
        return updates

    return understand


def route_after_understand(state: GraphState):
    from langgraph.types import Send

    plan: Plan | None = state.get("plan")
    if plan is None:
        return "synthesize"

    if plan.intent == "clarify":
        return "clarify"
    if plan.intent == "chitchat":
        return "respond_direct"

    # intent == "search"
    wave0 = [s for s in plan.steps if not s.depends_on]
    if not wave0:
        return "synthesize"

    return [Send(SOURCE_NODE[s.source], {"step": s.model_dump()}) for s in wave0]


# ── helpers ──────────────────────────────────────────────────────────────────

def _available_sources(services) -> list[str]:
    if services is None:
        return ["email", "sql", "site", "web"]
    sources = ["email"]
    if services.database:
        sources.append("sql")
    if services.company_site:
        sources.append("site")
    sources.append("web")
    return sources


def _filter_steps(plan: Plan, available: list[str], max_steps: int) -> Plan:
    filtered = [s for s in plan.steps if s.source in available][:max_steps]
    return plan.model_copy(update={"steps": filtered})


def _validate_dates(plan: Plan) -> tuple[Plan, list[ToolResult]]:
    bad: list[ToolResult] = []
    good: list[Step] = []
    for step in plan.steps:
        invalid = False
        for field in ("sent_on", "sent_after", "sent_before"):
            val = getattr(step, field, None)
            if val and not _valid_date(val):
                bad.append(ToolResult(
                    step_id=step.id, source=step.source,
                    status="error", error_kind="bad_args",
                    error=f"Invalid date '{val}' in field '{field}'",
                ))
                invalid = True
                break
        if not invalid:
            good.append(step)
    return plan.model_copy(update={"steps": good}), bad
