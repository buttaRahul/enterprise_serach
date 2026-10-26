"""SQL repair subgraph — link_schema → generate_sql → guard → execute → classify → critic."""
from __future__ import annotations
import asyncio
import hashlib
import re
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.graph.prompts import CRITIC_PROMPT, GENERATE_SQL_PROMPT


# ── State ─────────────────────────────────────────────────────────────────────

class SQLState(TypedDict, total=False):
    question: str
    step_id: str
    context: str
    workspace_id: str
    linked_tables: list[str]
    schemas: str
    sql: str
    attempts: int
    first_try_ok: bool
    last_error_kind: Optional[str]
    last_error: Optional[str]
    signatures: list[str]
    critique: Optional[str]
    excluded_tables: list[str]
    rows: list[dict]
    status: Optional[Literal["ok", "empty", "error"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _sig(kind: str, error: str, sql: str) -> str:
    sql_norm = re.sub(r"'[^']*'", "?", re.sub(r"\s+", " ", sql.lower().strip()))
    raw = f"{kind}|{_norm(error)}|{sql_norm}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _tables_in_sql(sql: str) -> list[str]:
    try:
        import sqlglot
        import sqlglot.expressions as exp
        tree = sqlglot.parse_one(sql)
        return [t.name.lower() for t in tree.find_all(exp.Table)]
    except Exception:
        return []


class SQLDraft(BaseModel):
    sql: str
    rationale: str


def _classify_db_error(msg: str) -> str:
    m = msg.lower()
    if any(p in m for p in ["no such column", "unknown column", "does not exist"]):
        return "unknown_column"
    if any(p in m for p in ["no such table", "doesn't exist", "relation"]):
        return "unknown_table"
    if any(p in m for p in ["syntax error", 'near "', "parseerror"]):
        return "syntax"
    if any(p in m for p in ["interrupted", "timeout", "canceling statement"]):
        return "timeout"
    if any(p in m for p in ["permission", "denied", "readonly", "read-only"]):
        return "permission"
    if any(p in m for p in ["datatype mismatch", "cannot compare", "invalid input syntax"]):
        return "type_mismatch"
    return "unknown"


# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_link_schema_node(database):
    async def link_schema_node(state: SQLState) -> dict:
        from app.graph.sql.link import link_schema
        excluded = state.get("excluded_tables") or []
        linked, schemas = await asyncio.to_thread(
            link_schema, state["question"], database, excluded
        )
        schemas_str = "\n\n".join(schemas.values())
        return {"linked_tables": linked, "schemas": schemas_str}
    return link_schema_node


def make_generate_sql_node(llm, settings, feedback_store=None):
    async def generate_sql_node(state: SQLState) -> dict:
        import asyncio
        dialect = "sqlite"
        critique = state.get("critique")
        context = state.get("context", "")
        sql = state.get("sql", "")
        kind = state.get("last_error_kind", "")
        error = state.get("last_error", "")
        attempts = state.get("attempts", 0)

        critique_block = ""
        if critique and sql:
            critique_block = (
                f"\nPrevious SQL failed: {sql}\n"
                f"Error ({kind}): {error}\n"
                f"Fix plan: {critique}\n"
            )

        context_block = f"\nEarlier result: {context}\n" if context else ""

        # Few-shot from positive feedback
        fewshot_block = ""
        corrections_block = ""
        workspace_id = state.get("workspace_id", "")
        if feedback_store and workspace_id and attempts == 0:
            try:
                positives = await asyncio.to_thread(
                    feedback_store.list_positives, workspace_id, 10
                )
                if positives:
                    examples = "\n".join(
                        f"Q: {r.question}\nSQL: {r.sql}" for r in positives[:3]
                    )
                    fewshot_block = f"\nExamples from users in this workspace:\n{examples}\n"

                corrections = await asyncio.to_thread(
                    feedback_store.list_corrections, workspace_id, 10
                )
                if corrections:
                    corr_lines = "\n".join(
                        f"Q: {r.question} | Wrong SQL: {r.sql} | Correction: {r.correction}"
                        for r in corrections[:3]
                    )
                    corrections_block = f"\nCorrections from users:\n{corr_lines}\n"
            except Exception:
                pass

        prompt = GENERATE_SQL_PROMPT.format(
            dialect=dialect,
            schemas=state.get("schemas", ""),
            corrections_block=corrections_block,
            fewshot_block=fewshot_block,
            context_block=context_block,
            critique_block=critique_block,
            question=state["question"],
        )
        msgs = [SystemMessage(content=prompt), HumanMessage(content=state["question"])]
        draft: SQLDraft = await llm.structured(SQLDraft, msgs, name="generate_sql")

        new_attempts = attempts + 1
        return {
            "sql": draft.sql,
            "attempts": new_attempts,
            "first_try_ok": new_attempts == 1,
            "critique": None,
        }
    return generate_sql_node


def make_guard_node(settings):
    async def guard_node(state: SQLState) -> dict:
        from app.tools.database import UnsafeQueryError, make_safe
        try:
            safe = make_safe(
                state["sql"],
                dialect="sqlite",
                max_rows=100,
                allowed_tables=settings.sql_allowed_tables or None,
                forbidden_columns=settings.sql_forbidden_columns or None,
            )
            return {"sql": safe, "last_error_kind": None, "last_error": None}
        except UnsafeQueryError as exc:
            return {"last_error_kind": f"guard_{exc.kind}", "last_error": str(exc)}
    return guard_node


def route_after_guard(state: SQLState) -> str:
    return "classify" if state.get("last_error_kind") else "execute"


def make_execute_node(database):
    async def execute_node(state: SQLState) -> dict:
        from app.tools.database import SQLQueryInput
        try:
            result = await asyncio.to_thread(
                database.query, SQLQueryInput(sql=state["sql"])
            )
            rows = result["rows"]
            return {
                "rows": rows,
                "status": "ok" if rows else "empty",
                "last_error_kind": None,
                "last_error": None,
            }
        except Exception as exc:
            kind = _classify_db_error(str(exc))
            return {"last_error_kind": kind, "last_error": str(exc), "status": "error"}
    return execute_node


def route_after_execute(state: SQLState) -> str:
    return "classify" if state.get("last_error_kind") else END


def make_classify_node(settings):
    async def classify_node(state: SQLState) -> dict:
        kind = state.get("last_error_kind", "unknown")
        error = state.get("last_error", "")
        sql = state.get("sql", "")
        sigs = list(state.get("signatures") or [])
        attempts = state.get("attempts", 1)
        max_attempts = getattr(settings, "sql_max_attempts", 3)

        terminal = {
            "permission", "timeout",
            "guard_write_op", "guard_not_select", "guard_multi_statement",
        }
        if kind in terminal:
            return {"status": "error"}

        sig = _sig(kind, error, sql)
        if sig in sigs or attempts >= max_attempts:
            return {"status": "error"}

        sigs.append(sig)
        return {"signatures": sigs, "status": None}
    return classify_node


def route_after_classify(state: SQLState) -> str:
    if state.get("status") == "error":
        return END
    kind = state.get("last_error_kind", "")
    if kind in ("unknown_table", "guard_table_not_allowed"):
        return "link_schema"
    return "critic"


def make_critic_node(llm):
    async def critic_node(state: SQLState) -> dict:
        msgs = [
            SystemMessage(content=CRITIC_PROMPT.format(
                question=state["question"],
                sql=state.get("sql", ""),
                kind=state.get("last_error_kind", ""),
                error=state.get("last_error", ""),
                schemas=state.get("schemas", ""),
            )),
            HumanMessage(content="Provide a fix plan."),
        ]
        critique = await llm.text(msgs, name="critic")
        excluded = list(state.get("excluded_tables") or [])
        if state.get("last_error_kind") == "unknown_table":
            for t in _tables_in_sql(state.get("sql", "")):
                if t not in excluded:
                    excluded.append(t)
        return {"critique": critique, "excluded_tables": excluded}
    return critic_node


# ── Build ─────────────────────────────────────────────────────────────────────

def build_sql_subgraph(database, llm, settings, feedback_store=None):
    g = StateGraph(SQLState)
    g.add_node("link_schema", make_link_schema_node(database))
    g.add_node("generate_sql", make_generate_sql_node(llm, settings, feedback_store))
    g.add_node("guard", make_guard_node(settings))
    g.add_node("execute", make_execute_node(database))
    g.add_node("classify", make_classify_node(settings))
    g.add_node("critic", make_critic_node(llm))

    g.add_edge(START, "link_schema")
    g.add_edge("link_schema", "generate_sql")
    g.add_edge("generate_sql", "guard")
    g.add_conditional_edges("guard", route_after_guard, ["execute", "classify"])
    g.add_conditional_edges("execute", route_after_execute, ["classify", END])
    g.add_conditional_edges("classify", route_after_classify, ["link_schema", "critic", END])
    g.add_edge("critic", "generate_sql")

    return g.compile()
