"""Tests for tools — all offline."""
from __future__ import annotations
import pytest

from tests.conftest import hashing_embedder, SAMPLE_EMAILS
from app.tools.emails import EmailIndex, EmailSearchInput
from app.tools.database import Database, UnsafeQueryError, make_safe


# ── Email search ──────────────────────────────────────────────────────────────

class TestEmailSearch:
    def setup_method(self):
        self.idx = EmailIndex(SAMPLE_EMAILS, hashing_embedder)

    def test_sender_filter_exact(self):
        r = self.idx.search(EmailSearchInput(sender="rahul@company.com"))
        assert all("rahul" in e["sender"] for e in r["emails"])
        assert r["count"] >= 1

    def test_sender_filter_partial(self):
        r = self.idx.search(EmailSearchInput(sender="ramesh"))
        assert r["count"] >= 1

    def test_before_filter(self):
        r = self.idx.search(EmailSearchInput(sent_before="2025-02-20"))
        for e in r["emails"]:
            assert e["timestamp"] < "2025-02-20"

    def test_after_filter(self):
        r = self.idx.search(EmailSearchInput(sent_after="2025-02-20"))
        for e in r["emails"]:
            d = e["timestamp"][:10]
            assert d > "2025-02-20"

    def test_on_filter(self):
        r = self.idx.search(EmailSearchInput(sent_on="2025-02-21"))
        assert r["count"] >= 1
        for e in r["emails"]:
            assert e["timestamp"].startswith("2025-02-21")

    def test_content_query_returns_results(self):
        r = self.idx.search(EmailSearchInput(content_query="pull request"))
        assert r["count"] >= 1

    def test_no_params_raises(self):
        with pytest.raises(ValueError):
            self.idx.search(EmailSearchInput())

    def test_empty_result_on_unknown_sender(self):
        r = self.idx.search(EmailSearchInput(sender="nobody@nowhere.com"))
        assert r["count"] == 0

    def test_top_k_respected(self):
        r = self.idx.search(EmailSearchInput(content_query="email", top_k=2))
        assert len(r["emails"]) <= 2


# ── make_safe ─────────────────────────────────────────────────────────────────

class TestMakeSafe:
    def test_allows_simple_select(self):
        sql = make_safe("SELECT * FROM employees", allowed_tables=["employees"])
        assert "LIMIT" in sql.upper()

    def test_rejects_insert(self):
        with pytest.raises(UnsafeQueryError) as exc:
            make_safe("INSERT INTO employees VALUES (1,'x')", allowed_tables=["employees"])
        assert exc.value.kind == "write_op"

    def test_rejects_multi_statement(self):
        with pytest.raises(UnsafeQueryError) as exc:
            make_safe("SELECT 1; DROP TABLE employees", allowed_tables=["employees"])
        assert exc.value.kind == "multi_statement"

    def test_rejects_unlisted_table(self):
        with pytest.raises(UnsafeQueryError) as exc:
            make_safe("SELECT * FROM salaries", allowed_tables=["employees", "projects"])
        assert exc.value.kind == "table_not_allowed"

    def test_rejects_forbidden_column(self):
        with pytest.raises(UnsafeQueryError) as exc:
            make_safe("SELECT salary FROM employees",
                      allowed_tables=["employees"], forbidden_columns=["salary"])
        assert exc.value.kind == "forbidden_column"

    def test_injects_limit(self):
        sql = make_safe("SELECT id FROM employees", max_rows=5, allowed_tables=["employees"])
        assert "5" in sql

    def test_cte_not_flagged_as_table(self):
        sql = make_safe(
            "WITH t AS (SELECT 1) SELECT * FROM t",
            allowed_tables=["employees"],
        )
        assert sql  # should not raise

    def test_empty_allowed_tables_permits_all(self):
        sql = make_safe("SELECT * FROM employees", allowed_tables=None)
        assert sql


# ── Database ──────────────────────────────────────────────────────────────────

class TestDatabase:
    def test_schema_returns_ddl(self, database):
        schema = database.schema(["employees"])
        assert "employees" in schema
        assert "first_name" in schema["employees"]

    def test_query_returns_rows(self, database):
        from app.tools.database import SQLQueryInput
        result = database.query(SQLQueryInput(sql="SELECT * FROM employees"))
        assert result["rows"]
        assert "first_name" in result["rows"][0]

    def test_query_rejects_write(self, database):
        from app.tools.database import SQLQueryInput
        with pytest.raises(UnsafeQueryError):
            database.query(SQLQueryInput(sql="DELETE FROM employees"))

    def test_allowlist_blocks_unlisted_table(self, tmp_path):
        """salaries table is not in the allowlist — make_safe should reject it."""
        with pytest.raises(UnsafeQueryError) as exc:
            make_safe(
                "SELECT * FROM salaries",
                allowed_tables=["employees", "projects"],
            )
        assert exc.value.kind == "table_not_allowed"
