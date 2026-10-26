"""synthesize node — LLM text answer from evidence."""
from __future__ import annotations
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.prompts import SYNTHESIZE_PROMPT
from app.graph.state import Evidence, GraphState, ToolResult, TraceEvent

MAX_EVIDENCE_CHARS = 6000  # overridden by settings


def make_synthesize(llm, max_evidence_chars: int = MAX_EVIDENCE_CHARS):
    async def synthesize(state: GraphState) -> dict:
        t0 = time.perf_counter()
        evidence: list[Evidence] = state.get("evidence") or []
        results: list[ToolResult] = state.get("results") or []
        plan = state.get("plan")
        standalone = plan.standalone_query if plan else state.get("query", "")
        llm_calls = 0

        if not evidence:
            # Deterministic fallback — skip LLM call
            sources = list({r.source for r in results}) if results else ["unknown"]
            answer = (
                f"I searched {', '.join(sources)} but found nothing relevant to \"{standalone}\"."
            )
        else:
            evidence_block = _build_evidence_block(evidence, max_evidence_chars)
            failed_summary = _failed_summary(results)

            msgs = [
                SystemMessage(content=SYNTHESIZE_PROMPT.format(
                    evidence_block=evidence_block,
                    failed_summary=failed_summary or "none",
                    standalone_query=standalone,
                )),
                HumanMessage(content=standalone),
            ]
            answer = await llm.text(msgs, name="synthesize")
            llm_calls = 1

        ms = int((time.perf_counter() - t0) * 1000)
        event = TraceEvent(node="synthesize", ms=ms, llm_calls=llm_calls)
        return {"answer": answer, "trace": [event]}

    return synthesize


def _build_evidence_block(evidence: list[Evidence], max_chars: int) -> str:
    """Build numbered evidence block, ordered by step then score, truncated."""
    sorted_ev = sorted(
        evidence,
        key=lambda e: (e.step_id, -(e.score or 0.0)),
    )
    lines: list[str] = []
    total = 0
    for e in sorted_ev:
        chunk = f"[{e.id}] ({e.source}) {e.title}\n{e.text}"
        if e.url:
            chunk += f"\n{e.url}"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                lines.append(chunk[:remaining])
            break
        lines.append(chunk)
        total += len(chunk)
    return "\n\n".join(lines)


def _failed_summary(results: list[ToolResult]) -> str:
    failed = [r for r in results if r.status in ("empty", "error")]
    if not failed:
        return ""
    return "; ".join(f"{r.source}({r.step_id}): {r.status}" for r in failed)
