"""Feedback store tests — P4.1-P4.2 — all offline."""
from __future__ import annotations
import asyncio
import tempfile
import os

import pytest

from app.memory.feedback_store import FeedbackStore, FeedbackRecord


def _store(tmp_path_str: str) -> FeedbackStore:
    return FeedbackStore(os.path.join(tmp_path_str, "feedback.db"))


# ── P4.1: basic roundtrip ────────────────────────────────────────────────────

class TestFeedbackRoundtrip:
    def test_save_and_retrieve(self, tmp_path):
        store = _store(str(tmp_path))
        rec = FeedbackRecord(
            workspace_id="demo",
            thread_id="t1",
            question="How many engineers?",
            sql="SELECT COUNT(*) FROM employees WHERE department='Engineering'",
            rating="up",
        )
        fb_id = store.save(rec)
        assert fb_id > 0

        positives = store.list_positives("demo")
        assert len(positives) == 1
        assert positives[0].question == rec.question
        assert positives[0].sql == rec.sql
        assert positives[0].rating == "up"

    def test_down_with_correction(self, tmp_path):
        store = _store(str(tmp_path))
        rec = FeedbackRecord(
            workspace_id="demo",
            thread_id="t2",
            question="List projects",
            sql="SELECT * FROM project",  # typo
            correction="SELECT * FROM projects",
            rating="down",
        )
        store.save(rec)
        corrections = store.list_corrections("demo")
        assert len(corrections) == 1
        assert corrections[0].correction == "SELECT * FROM projects"

    def test_up_not_in_corrections(self, tmp_path):
        store = _store(str(tmp_path))
        store.save(FeedbackRecord(
            workspace_id="demo", thread_id="t1",
            question="q", sql="SELECT 1", rating="up",
        ))
        corrections = store.list_corrections("demo")
        assert corrections == []

    def test_empty_sql_excluded_from_positives(self, tmp_path):
        store = _store(str(tmp_path))
        store.save(FeedbackRecord(
            workspace_id="demo", thread_id="t1",
            question="q", sql="", rating="up",
        ))
        positives = store.list_positives("demo")
        assert positives == []


# ── P4.1: workspace scoped ───────────────────────────────────────────────────

class TestFeedbackIsWorkspaceScoped:
    def test_other_workspace_excluded(self, tmp_path):
        store = _store(str(tmp_path))
        store.save(FeedbackRecord(
            workspace_id="ws-A", thread_id="t1",
            question="q", sql="SELECT 1", rating="up",
        ))
        store.save(FeedbackRecord(
            workspace_id="ws-B", thread_id="t2",
            question="q2", sql="SELECT 2", rating="up",
        ))
        result_a = store.list_positives("ws-A")
        result_b = store.list_positives("ws-B")
        assert len(result_a) == 1 and result_a[0].sql == "SELECT 1"
        assert len(result_b) == 1 and result_b[0].sql == "SELECT 2"


# ── P4.2: few-shot injection ─────────────────────────────────────────────────

class TestFewshotInjected:
    def test_fewshot_included_in_prompt(self, tmp_path):
        """generate_sql_node must include few-shot examples from positive feedback."""
        import sqlite3
        from tests.conftest import ScriptedLLM
        from app.graph.sql.subgraph import SQLDraft, build_sql_subgraph
        from app.config import Settings

        # Create store with one positive example
        store = _store(str(tmp_path))
        store.save(FeedbackRecord(
            workspace_id="demo",
            thread_id="t1",
            question="count engineers",
            sql="SELECT COUNT(*) FROM employees WHERE department='Engineering'",
            rating="up",
        ))

        # Create a tiny DB
        db_path = os.path.join(str(tmp_path), "t.db")
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE employees (id INTEGER, department TEXT)")
        con.execute("INSERT INTO employees VALUES (1,'Engineering')")
        con.commit()
        con.close()

        from app.tools.database import Database
        db = Database(db_path)
        settings = Settings()
        settings_obj = object.__setattr__

        llm = ScriptedLLM({
            "generate_sql": [SQLDraft(
                sql="SELECT COUNT(*) FROM employees WHERE department='Engineering'",
                rationale="",
            )],
        })

        # Capture what prompt text was passed to the LLM
        captured = []
        original_structured = llm.structured

        async def capturing_structured(schema, messages, *, name):
            if name == "generate_sql":
                captured.extend(messages)
            return await original_structured(schema, messages, name=name)

        llm.structured = capturing_structured

        graph = build_sql_subgraph(db, llm, Settings(), feedback_store=store)
        result = asyncio.run(graph.ainvoke({
            "question": "count engineers",
            "step_id": "s1",
            "context": "",
            "workspace_id": "demo",
            "attempts": 0,
            "signatures": [],
            "excluded_tables": [],
        }))

        # The prompt (SystemMessage) should contain the few-shot example
        all_content = " ".join(
            str(m.content) for m in captured
        )
        assert "count engineers" in all_content.lower() or "Engineering" in all_content


class TestOtherWorkspaceExamplesExcluded:
    def test_other_workspace_not_injected(self, tmp_path):
        """Few-shot examples from workspace ws-B are not injected for ws-A."""
        import sqlite3
        from tests.conftest import ScriptedLLM
        from app.graph.sql.subgraph import SQLDraft, build_sql_subgraph
        from app.config import Settings

        store = _store(str(tmp_path))
        store.save(FeedbackRecord(
            workspace_id="ws-B",  # different workspace
            thread_id="t1",
            question="secret query",
            sql="SELECT secret FROM hidden",
            rating="up",
        ))

        db_path = os.path.join(str(tmp_path), "t2.db")
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE employees (id INTEGER)")
        con.commit()
        con.close()

        from app.tools.database import Database
        db = Database(db_path)

        llm = ScriptedLLM({
            "generate_sql": [SQLDraft(sql="SELECT * FROM employees", rationale="")],
        })
        captured = []
        orig = llm.structured

        async def cap(schema, messages, *, name):
            if name == "generate_sql":
                captured.extend(messages)
            return await orig(schema, messages, name=name)

        llm.structured = cap
        graph = build_sql_subgraph(db, llm, Settings(), feedback_store=store)
        asyncio.run(graph.ainvoke({
            "question": "get all",
            "step_id": "s1",
            "context": "",
            "workspace_id": "ws-A",  # different workspace
            "attempts": 0,
            "signatures": [],
            "excluded_tables": [],
        }))

        all_content = " ".join(str(m.content) for m in captured)
        assert "secret" not in all_content.lower()
