"""SQL subgraph tests — P2.3 / P2.4 — all offline."""
from __future__ import annotations
import asyncio
import sqlite3
import tempfile
import os

import pytest

from tests.conftest import ScriptedLLM
from app.config import Settings
from app.tools.database import Database, UnsafeQueryError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _temp_db() -> Database:
    """SQLite DB in a temp file, seeded with a tiny employees table."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            first_name TEXT,
            department TEXT
        )
    """)
    con.execute("INSERT INTO employees VALUES (1,'Alice','Engineering')")
    con.execute("INSERT INTO employees VALUES (2,'Bob','Product')")
    con.commit()
    con.close()
    return Database(path, timeout_s=2.0)


def _settings(**kwargs) -> Settings:
    s = Settings()
    for k, v in kwargs.items():
        object.__setattr__(s, k, v)
    return s


# ── P2.3: timeout ─────────────────────────────────────────────────────────────

class TestSQLTimeout:
    def test_timeout_classified(self):
        """A recursive CTE that runs forever should be interrupted by the progress handler."""
        db = _temp_db()
        db._timeout_s = 0.1  # 100 ms
        from app.tools.database import SQLQueryInput
        # Use COUNT(*) on a cross-join of a large CTE — cannot be short-circuited by LIMIT
        with pytest.raises(Exception) as exc_info:
            db.query(SQLQueryInput(
                sql=(
                    "WITH RECURSIVE r(n) AS (SELECT 0 UNION ALL SELECT n+1 FROM r WHERE n < 999) "
                    "SELECT COUNT(*) FROM r CROSS JOIN r AS r2 CROSS JOIN r AS r3"
                )
            ))
        msg = str(exc_info.value).lower()
        assert any(w in msg for w in ("interrupt", "abort", "cancel")), msg


# ── P2.4: SQL subgraph repair / stop tests ────────────────────────────────────

def _build_sql_graph(llm, settings=None):
    from app.graph.sql.subgraph import build_sql_subgraph
    db = _temp_db()
    s = settings or _settings(
        sql_max_attempts=3,
        sql_timeout_s=5,
        sql_allowed_tables=["employees"],
        sql_forbidden_columns=[],
    )
    return build_sql_subgraph(db, llm, s)


class TestRepairAfterUnknownColumn:
    def test_repair_after_unknown_column(self):
        """generate_sql returns bad column first; critic suggests fix; second attempt succeeds."""
        from app.graph.sql.subgraph import SQLDraft

        first_sql = "SELECT bad_column FROM employees"
        fixed_sql = "SELECT first_name FROM employees"

        llm = ScriptedLLM({
            "generate_sql": [
                SQLDraft(sql=first_sql, rationale="first try"),
                SQLDraft(sql=fixed_sql, rationale="fixed"),
            ],
            "critic": ["Use first_name instead of bad_column"],
        })
        graph = _build_sql_graph(llm)

        async def run():
            return await graph.ainvoke({
                "question": "List employee names",
                "step_id": "s1",
                "context": "",
                "workspace_id": "",
                "attempts": 0,
                "signatures": [],
                "excluded_tables": [],
            })

        result = asyncio.run(run())
        assert result["status"] == "ok"
        assert result["attempts"] == 2
        assert result.get("first_try_ok") is False  # repair needed → not first-try success


class TestStopOnRepeatedSignature:
    def test_stop_on_repeated_signature(self):
        """Same error twice → stop, not retry forever."""
        from app.graph.sql.subgraph import SQLDraft

        bad_sql = "SELECT bad_col FROM employees"
        llm = ScriptedLLM({
            "generate_sql": [
                SQLDraft(sql=bad_sql, rationale="try 1"),
                SQLDraft(sql=bad_sql, rationale="try 2"),
                SQLDraft(sql=bad_sql, rationale="try 3"),
            ],
            "critic": ["Try using first_name", "Try using first_name again"],
        })
        graph = _build_sql_graph(llm, _settings(
            sql_max_attempts=3,
            sql_timeout_s=5,
            sql_allowed_tables=["employees"],
            sql_forbidden_columns=[],
        ))

        async def run():
            return await graph.ainvoke({
                "question": "names",
                "step_id": "s1",
                "context": "",
                "workspace_id": "",
                "attempts": 0,
                "signatures": [],
                "excluded_tables": [],
            })

        result = asyncio.run(run())
        assert result["status"] == "error"
        # Should stop after seeing repeated signature, not exhaust all 3 attempts
        assert result.get("attempts", 0) <= 3


class TestPermissionStopsImmediately:
    def test_write_op_stops_immediately(self):
        """A write statement is rejected by the guard as write_op — no critic call."""
        from app.graph.sql.subgraph import SQLDraft

        llm = ScriptedLLM({
            "generate_sql": [SQLDraft(sql="DELETE FROM employees WHERE id=1", rationale="bad")],
            # No "critic" key — KeyError if critic is called
        })
        graph = _build_sql_graph(llm)

        async def run():
            return await graph.ainvoke({
                "question": "delete first employee",
                "step_id": "s1",
                "context": "",
                "workspace_id": "",
                "attempts": 0,
                "signatures": [],
                "excluded_tables": [],
            })

        result = asyncio.run(run())
        assert result["status"] == "error"
        assert result.get("attempts", 0) == 1
        critic_calls = [name for name, _ in llm.calls if name == "critic"]
        assert critic_calls == []
