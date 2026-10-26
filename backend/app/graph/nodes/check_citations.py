"""check_citations node — deterministic citation extraction and validation."""
from __future__ import annotations
import re
import time

from langchain_core.messages import AIMessage

from app.graph.state import Evidence, GraphState, TraceEvent

_CITATION_RE = re.compile(r"\[([a-z]+:[^\]]+)\]")


async def check_citations(state: GraphState) -> dict:
    t0 = time.perf_counter()
    answer: str = state.get("answer", "")
    evidence: list[Evidence] = state.get("evidence") or []

    valid_ids = {e.id for e in evidence}
    found_ids: list[str] = []
    invalid_ids: list[str] = []

    for m in _CITATION_RE.finditer(answer):
        cid = m.group(1)
        if cid in valid_ids:
            if cid not in found_ids:
                found_ids.append(cid)
        else:
            invalid_ids.append(cid)

    # Remove unknown citation brackets from the text
    clean_answer = answer
    for bad in invalid_ids:
        clean_answer = clean_answer.replace(f"[{bad}]", "")

    info: dict = {}
    if invalid_ids:
        info["invalid_citations"] = invalid_ids
    if evidence and not found_ids:
        info["uncited"] = True  # feeds eval metric; don't fail the request

    ms = int((time.perf_counter() - t0) * 1000)
    event = TraceEvent(node="check_citations", ms=ms, info=info)

    return {
        "answer": clean_answer,
        "citations": found_ids,
        "status": "answered",
        "messages": [AIMessage(clean_answer)],
        "trace": [event],
    }
