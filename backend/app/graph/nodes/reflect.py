"""reflect node — dispatch wave-1 dependent steps or ask LLM to retry."""
from __future__ import annotations
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.graph.prompts import REFLECT_PROMPT
from app.graph.state import Evidence, GraphState, Step, ToolResult, TraceEvent

MAX_CONTEXT_CHARS = 1500

SOURCE_NODE = {
    "email": "email_retrieve",
    "site": "site_retrieve",
    "web": "web_retrieve",
    "sql": "sql_answer",
}


class ReflectDecision(BaseModel):
    retry: bool
    steps: list[Step] = []


def _evidence_summary(evidence: list[Evidence], step_id: str) -> str:
    snippets = [
        f"[{e.id}] {e.title}: {e.text[:200]}"
        for e in evidence
        if e.step_id == step_id
    ]
    return "\n".join(snippets)[:MAX_CONTEXT_CHARS]


def make_reflect(llm, available_sources: list[str] | None = None):
    async def reflect(state: GraphState):
        t0 = time.perf_counter()
        llm_calls = 0
        info: dict = {}

        wave: int = state.get("wave", 0)
        pending: list[Step] = state.get("pending_steps", [])
        results: list[ToolResult] = state.get("results") or []
        plan = state.get("plan")
        standalone = plan.standalone_query if plan else state.get("query", "")
        srcs = available_sources or ["email", "sql", "site", "web"]

        updates: dict = {}

        # ── Wave-0 + pending dependent steps → signal route to dispatch them ──
        if pending and wave == 0:
            info["dispatched_wave1"] = len(pending)
            # Set wave=1 so route_after_reflect can dispatch; DON'T clear pending
            # (route_after_reflect needs to read it, then wave-1 reflect clears it)
            updates = {"wave": 1}

        # ── Wave ≥ 1 → we've executed all retry/dependent steps → synthesize ─
        elif wave >= 1:
            info["route"] = "synthesize"
            # Clear pending_steps so route_after_reflect doesn't re-dispatch
            if pending:
                updates = {"pending_steps": []}

        else:
            # Wave-0, no pending — check for retryable failures
            failed = [
                r for r in results
                if r.status in ("empty", "error") and r.error_kind != "bad_args"
            ]
            if not failed:
                info["route"] = "synthesize"
            else:
                failed_summary = "; ".join(
                    f"{r.source}({r.step_id}): {r.status}" for r in failed
                )
                msgs = [
                    SystemMessage(content=REFLECT_PROMPT.format(
                        standalone_query=standalone,
                        failed_summary=failed_summary,
                        sources=", ".join(srcs),
                    )),
                    HumanMessage(content="Should we retry?"),
                ]
                decision: ReflectDecision = await llm.structured(
                    ReflectDecision, msgs, name="reflect"
                )
                llm_calls += 1

                if decision.retry and decision.steps:
                    valid = [s for s in decision.steps if s.source in srcs]
                    if valid:
                        # wave=1 + pending_steps → route_after_reflect dispatches them
                        updates = {"wave": 1, "pending_steps": valid}
                        info["route"] = "retry"
                        info["dispatched_retry"] = len(valid)
                    else:
                        info["route"] = "synthesize"
                else:
                    info["route"] = "synthesize"

        ms = int((time.perf_counter() - t0) * 1000)
        event = TraceEvent(node="reflect", ms=ms, llm_calls=llm_calls, info=info)
        return {**updates, "trace": [event]}

    return reflect


def route_after_reflect(state: GraphState):
    from langgraph.types import Send

    wave: int = state.get("wave", 0)
    pending: list[Step] = state.get("pending_steps", [])
    evidence: list[Evidence] = state.get("evidence") or []

    # Dispatch wave-1 dependent steps OR retry steps, with context from prior steps
    if wave == 1 and pending:
        return [
            Send(
                SOURCE_NODE[s.source],
                {
                    "step": s.model_dump(),
                    "context": _evidence_summary(evidence, s.depends_on) if s.depends_on else "",
                },
            )
            for s in pending
        ]
    return "synthesize"
