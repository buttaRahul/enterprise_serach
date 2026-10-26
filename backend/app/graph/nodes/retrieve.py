"""Retrieval nodes — email, site, web, sql_answer."""
from __future__ import annotations
import asyncio
import re
import time
from contextlib import contextmanager
from typing import Generator

from langchain_core.runnables import RunnableConfig

from app.graph.state import Evidence, Step, ToolResult, TraceEvent

_STOP_WORDS = frozenset({
    "emails", "email", "mail", "mails", "messages", "message",
    "sent", "from", "to", "about", "regarding", "related",
})


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


def _is_pure_filter(step: Step) -> bool:
    """True when the step has date/sender filters and the query adds no content signal."""
    has_filter = any([step.sender, step.sent_on, step.sent_after, step.sent_before])
    if not has_filter:
        return False
    remaining = step.query.lower()
    for w in _STOP_WORDS:
        remaining = remaining.replace(w, " ")
    # strip sender/date values
    for val in filter(None, [step.sender, step.sent_on, step.sent_after, step.sent_before]):
        remaining = remaining.replace(val.lower(), " ")
    content_words = [w for w in re.split(r"\W+", remaining) if len(w) > 2]
    return len(content_words) < 2


# ── email_retrieve ────────────────────────────────────────────────────────────

def make_email_retrieve(services):
    async def email_retrieve(payload: dict, config: RunnableConfig) -> dict:
        step = Step(**payload["step"])
        with _traced("email_retrieve", {"step": step.id}) as tb:
            try:
                from app.tools.emails import EmailSearchInput
                inp = EmailSearchInput(
                    content_query=step.query if not _is_pure_filter(step) else None,
                    sender=step.sender,
                    sent_on=step.sent_on,
                    sent_after=step.sent_after,
                    sent_before=step.sent_before,
                    top_k=5,
                )
                out = await asyncio.to_thread(services.emails.search, inp)
                ev = [
                    Evidence(
                        id=f"email:{e['id']}",
                        source="email",
                        step_id=step.id,
                        title=f"{e['sender']} · {e['timestamp']}",
                        text=e["content"],
                        score=e.get("score"),
                    )
                    for e in out["emails"]
                ]
                res = ToolResult(
                    step_id=step.id,
                    source="email",
                    status="ok" if ev else "empty",
                )
            except ValueError as exc:
                ev, res = [], ToolResult(
                    step_id=step.id, source="email",
                    status="error", error_kind="bad_args", error=str(exc),
                )
        res = res.model_copy(update={"latency_ms": tb["event"].ms})
        return {"evidence": ev, "results": [res], "trace": [tb["event"]]}

    return email_retrieve


# ── site_retrieve ─────────────────────────────────────────────────────────────

def make_site_retrieve(services):
    async def site_retrieve(payload: dict, config: RunnableConfig) -> dict:
        step = Step(**payload["step"])
        with _traced("site_retrieve", {"step": step.id}) as tb:
            if services.company_site is None:
                ev, res = [], ToolResult(
                    step_id=step.id, source="site",
                    status="error", error_kind="not_configured",
                    error="Company site index not configured.",
                )
            else:
                try:
                    from app.tools.company_site import CompanySearchInput
                    inp = CompanySearchInput(query=step.query)
                    out = await asyncio.to_thread(services.company_site.search, inp)
                    ev = [
                        Evidence(
                            id=f"site:{i}",
                            source="site",
                            step_id=step.id,
                            title=p.get("title", ""),
                            text=p["text"],
                            url=p.get("url"),
                            score=p.get("score"),
                        )
                        for i, p in enumerate(out["passages"])
                    ]
                    res = ToolResult(
                        step_id=step.id, source="site",
                        status="ok" if ev else "empty",
                    )
                except Exception as exc:
                    ev, res = [], ToolResult(
                        step_id=step.id, source="site",
                        status="error", error_kind="unknown", error=str(exc),
                    )
        res = res.model_copy(update={"latency_ms": tb["event"].ms})
        return {"evidence": ev, "results": [res], "trace": [tb["event"]]}

    return site_retrieve


# ── web_retrieve ──────────────────────────────────────────────────────────────

def make_web_retrieve(services):  # noqa: ARG001
    async def web_retrieve(payload: dict, config: RunnableConfig) -> dict:
        step = Step(**payload["step"])
        with _traced("web_retrieve", {"step": step.id}) as tb:
            try:
                from app.tools.web import WebSearchInput, web_search
                inp = WebSearchInput(query=step.query)
                out = await asyncio.to_thread(web_search, inp)
                if "error" in out:
                    raise RuntimeError(out["error"])
                ev = [
                    Evidence(
                        id=f"web:{step.id}:{i}",
                        source="web",
                        step_id=step.id,
                        title=r.get("title", ""),
                        text=r.get("snippet", ""),
                        url=r.get("url"),
                    )
                    for i, r in enumerate(out["results"])
                ]
                res = ToolResult(
                    step_id=step.id, source="web",
                    status="ok" if ev else "empty",
                )
            except Exception as exc:
                ev, res = [], ToolResult(
                    step_id=step.id, source="web",
                    status="error", error_kind="network", error=str(exc),
                )
        res = res.model_copy(update={"latency_ms": tb["event"].ms})
        return {"evidence": ev, "results": [res], "trace": [tb["event"]]}

    return web_retrieve


# ── sql_answer ────────────────────────────────────────────────────────────────

def make_sql_answer(services, llm, settings, feedback_store=None):
    async def sql_answer(payload: dict, config: RunnableConfig) -> dict:
        step = Step(**payload["step"])
        context = payload.get("context", "")
        workspace_id = (config or {}).get("configurable", {}).get("workspace_id", "")
        with _traced("sql_answer", {"step": step.id}) as tb:
            if services.database is None:
                ev, res = [], ToolResult(
                    step_id=step.id, source="sql",
                    status="error", error_kind="not_configured",
                    error="Database not configured.",
                )
            else:
                from app.graph.sql.subgraph import build_sql_subgraph
                sql_graph = build_sql_subgraph(services.database, llm, settings, feedback_store)
                sql_input = {
                    "question": step.query,
                    "step_id": step.id,
                    "context": context,
                    "workspace_id": workspace_id,
                    "attempts": 0,
                    "signatures": [],
                    "excluded_tables": [],
                }
                result = await sql_graph.ainvoke(sql_input)
                status = result.get("status", "error")
                sql_text = result.get("sql", "")
                rows = result.get("rows", [])

                if status in ("ok", "empty"):
                    # Format rows as a markdown table (up to 20 rows)
                    md = _rows_to_markdown(rows[:20])
                    if len(rows) > 20:
                        md += f"\n_({len(rows)} rows total)_"
                    ev = [Evidence(
                        id=f"sql:{step.id}",
                        source="sql",
                        step_id=step.id,
                        title=sql_text,
                        text=md or "_(no rows)_",
                    )]
                    res = ToolResult(
                        step_id=step.id, source="sql", status=status,
                        detail={
                            "sql": sql_text,
                            "attempts": result.get("attempts", 1),
                            "first_try_ok": result.get("first_try_ok", True),
                        },
                    )
                else:
                    ev = []
                    res = ToolResult(
                        step_id=step.id, source="sql", status="error",
                        error_kind=result.get("last_error_kind", "unknown"),
                        error=result.get("last_error", ""),
                        detail={"sql": sql_text, "attempts": result.get("attempts", 1)},
                    )
        res = res.model_copy(update={"latency_ms": tb["event"].ms})
        return {"evidence": ev, "results": [res], "trace": [tb["event"]]}

    return sql_answer


def _rows_to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)
