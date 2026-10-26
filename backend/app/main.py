"""FastAPI application factory and lifespan."""
from __future__ import annotations
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Settings, settings as _default_settings
from app.embeddings import default_embedder


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    thread_id: str
    rating: str  # "up" | "down"
    correction: Optional[str] = None  # required for "down", optional for "up"


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(services=None, llm=None, settings: Settings = _default_settings) -> FastAPI:
    """
    Build and return the FastAPI application.

    In production, services and llm are created inside the lifespan.
    For testing, pass them explicitly to avoid network calls.
    """
    _state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build services lazily (skipped when injected for tests)
        if services is None:
            _state["services"] = _build_services(settings)
        else:
            _state["services"] = services

        if llm is None:
            _state["llm"] = _build_llm(settings)
        else:
            _state["llm"] = llm

        # Feedback store (always available)
        from app.memory.feedback_store import FeedbackStore
        os.makedirs(os.path.dirname(os.path.abspath(settings.feedback_db_path)), exist_ok=True)
        _state["feedback"] = FeedbackStore(settings.feedback_db_path)

        if settings.agent_mode == "graph":
            from app.graph.build import build_graph

            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            os.makedirs(os.path.dirname(os.path.abspath(settings.checkpoint_path)), exist_ok=True)
            checkpointer = await AsyncSqliteSaver.from_conn_string(settings.checkpoint_path).__aenter__()
            _state["_checkpointer_ctx"] = checkpointer

            _state["graph"] = build_graph(
                _state["services"], _state["llm"],
                checkpointer=checkpointer, settings=settings,
                feedback_store=_state.get("feedback"),
            )

        yield

        # Cleanup
        ctx = _state.get("_checkpointer_ctx")
        if ctx and hasattr(ctx, "__aexit__"):
            await ctx.__aexit__(None, None, None)

    app = FastAPI(title="Enterprise Search API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "agent_mode": settings.agent_mode}

    # ------------------------------------------------------------------
    # v1 alias: POST /api/search  (ReAct)
    # ------------------------------------------------------------------

    @app.post("/api/search")
    async def search(req: SearchRequest):
        from app.agent import run_agent
        import asyncio

        result = await asyncio.to_thread(
            run_agent, req.query, _state["services"], _state["llm"]
        )
        return result

    # ------------------------------------------------------------------
    # v2: POST /api/chat  (LangGraph)
    # ------------------------------------------------------------------

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        from langgraph.types import Command
        from app.graph.state import turn_input

        graph = _state.get("graph")
        if graph is None:
            raise HTTPException(500, "Graph not initialised (AGENT_MODE != graph)")

        thread_id = req.thread_id or str(uuid.uuid4())
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "workspace_id": settings.workspace_id,
            },
            "recursion_limit": 40,
        }

        # Check for pending interrupt (clarification)
        state = await graph.aget_state(cfg)
        has_interrupt = any(i for t in state.tasks for i in t.interrupts) if state.tasks else False

        if has_interrupt:
            final = await graph.ainvoke(Command(resume=req.query), cfg)
        else:
            final = await graph.ainvoke(turn_input(req.query), cfg)

        # Check if the new state has an interrupt
        new_state = await graph.aget_state(cfg)
        new_interrupt = None
        for t in (new_state.tasks or []):
            for i in t.interrupts:
                new_interrupt = i
                break
            if new_interrupt:
                break

        if new_interrupt:
            return {
                "thread_id": thread_id,
                "status": "needs_clarification",
                "answer": new_interrupt.value.get("question", "Could you clarify?"),
                "citations": [],
                "evidence": [],
                "plan": None,
                "trace": [],
            }

        evidence = [e.model_dump() for e in (final.get("evidence") or [])]
        plan = final.get("plan")

        return {
            "thread_id": thread_id,
            "status": final.get("status", "answered"),
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "evidence": evidence,
            "plan": plan.model_dump() if plan else None,
            "trace": [t.model_dump() for t in (final.get("trace") or [])],
        }

    # ------------------------------------------------------------------
    # v2: POST /api/feedback
    # ------------------------------------------------------------------

    @app.post("/api/feedback")
    async def feedback(req: FeedbackRequest):
        from app.memory.feedback_store import FeedbackRecord
        store = _state.get("feedback")
        if store is None:
            raise HTTPException(500, "Feedback store not initialised")

        if req.rating not in ("up", "down"):
            raise HTTPException(400, "rating must be 'up' or 'down'")

        # Read question and SQL from the saved thread state — never trust the client
        graph = _state.get("graph")
        question = ""
        sql = ""
        if graph is not None:
            cfg = {"configurable": {"thread_id": req.thread_id, "workspace_id": settings.workspace_id}}
            try:
                thread_state = await graph.aget_state(cfg)
                vals = thread_state.values if thread_state else {}
                question = vals.get("query", "")
                for ev in (vals.get("evidence") or []):
                    src = ev.source if hasattr(ev, "source") else ev.get("source")
                    if src == "sql":
                        sql = (ev.title if hasattr(ev, "title") else ev.get("title")) or ""
                        break
            except Exception:
                pass

        rec = FeedbackRecord(
            workspace_id=settings.workspace_id,
            thread_id=req.thread_id,
            question=question,
            sql=sql,
            correction=req.correction or "",
            rating=req.rating,
        )
        fb_id = store.save(rec)
        return {"id": fb_id, "status": "saved"}

    # ------------------------------------------------------------------
    # v2: POST /api/chat/stream  (SSE)
    # ------------------------------------------------------------------

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest):
        from fastapi.responses import StreamingResponse
        from langgraph.types import Command
        from app.graph.state import turn_input

        graph = _state.get("graph")
        if graph is None:
            raise HTTPException(500, "Graph not initialised (AGENT_MODE != graph)")

        thread_id = req.thread_id or str(uuid.uuid4())
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "workspace_id": settings.workspace_id,
            },
            "recursion_limit": 40,
        }

        state = await graph.aget_state(cfg)
        has_interrupt = any(i for t in state.tasks for i in t.interrupts) if state.tasks else False
        inp = Command(resume=req.query) if has_interrupt else turn_input(req.query)

        async def event_stream():
            import json as _json
            async for chunk in graph.astream(inp, cfg, stream_mode=["updates", "custom"]):
                kind, data = chunk if isinstance(chunk, tuple) else ("updates", chunk)
                if kind == "custom":
                    yield f"data: {_json.dumps(data)}\n\n"
                elif kind == "updates":
                    for node, updates in data.items():
                        trace = updates.get("trace") or []
                        for ev in trace:
                            payload = {"type": "node_end", "node": ev.node if hasattr(ev, "node") else ev.get("node"), "ms": ev.ms if hasattr(ev, "ms") else ev.get("ms")}
                            yield f"data: {_json.dumps(payload)}\n\n"
                        if updates.get("evidence"):
                            for ev in updates["evidence"]:
                                payload = {"type": "evidence", "item": ev.model_dump() if hasattr(ev, "model_dump") else ev}
                                yield f"data: {_json.dumps(payload)}\n\n"
                        if updates.get("answer"):
                            yield f"data: {_json.dumps({'type': 'answer', 'text': updates['answer']})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_services(settings: Settings):
    import json as _json
    from app.tools import Services
    from app.tools.emails import EmailIndex
    from app.tools.database import Database

    # Emails
    emails_path = os.path.join(os.path.dirname(__file__), "..", "data", "emails.json")
    with open(emails_path) as f:
        emails = _json.load(f)
    email_index = EmailIndex(emails, default_embedder)

    # Database — read-only, with allowlist and forbidden-column enforcement
    db_path = settings.database_path
    database = (
        Database(
            db_path,
            allowed_tables=settings.sql_allowed_tables or None,
            forbidden_columns=settings.sql_forbidden_columns or None,
            max_rows=100,
        )
        if os.path.exists(db_path)
        else None
    )

    # Company site
    company_site = None
    if settings.company_site_urls:
        from app.tools.company_site import CompanySiteIndex
        company_site = CompanySiteIndex.from_urls(settings.company_site_urls, default_embedder)

    return Services(emails=email_index, database=database, company_site=company_site)


def _build_llm(settings: Settings):
    from app.llm import make_llm
    return make_llm(settings)


# ---------------------------------------------------------------------------
# WSGI entry-point  (not executed during test imports)
# ---------------------------------------------------------------------------

app = create_app()  # noqa: E402 — module-level singleton for uvicorn
