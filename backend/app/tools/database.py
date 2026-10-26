from __future__ import annotations
import re
import time
from typing import Optional

import sqlglot
import sqlglot.expressions as exp
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text, inspect as sa_inspect


class UnsafeQueryError(Exception):
    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind


class SQLQueryInput(BaseModel):
    sql: str = Field(..., description="A SQL SELECT statement to execute")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def make_safe(
    sql: str,
    dialect: str = "sqlite",
    max_rows: int = 100,
    allowed_tables: Optional[list[str]] = None,
    forbidden_columns: Optional[list[str]] = None,
) -> str:
    """
    Parse and validate a SQL statement via AST.

    Raises UnsafeQueryError with a typed .kind if the query violates any rule.
    Returns the sanitised SQL string (LIMIT injected or capped to max_rows).
    """
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise UnsafeQueryError("multi_statement", "Only a single statement is allowed.")

    try:
        statements = sqlglot.parse(sql, dialect=dialect)
    except Exception as exc:
        raise UnsafeQueryError("parse_error", f"Could not parse SQL: {exc}") from exc

    if not statements:
        raise UnsafeQueryError("parse_error", "Empty SQL statement.")

    if len(statements) > 1:
        raise UnsafeQueryError("multi_statement", "Only a single statement is allowed.")

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        kind = "write_op" if isinstance(tree, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create)) else "not_select"
        raise UnsafeQueryError(kind, "Only SELECT statements are allowed.")

    # Reject SELECT ... INTO <table> (SQL Server / PostgreSQL CREATE TABLE AS)
    if tree.args.get("into"):
        raise UnsafeQueryError("write_op", "SELECT INTO is not allowed.")

    cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}

    if allowed_tables:
        allowed_lower = {t.lower() for t in allowed_tables}
        for tbl in tree.find_all(exp.Table):
            name = tbl.name.lower()
            if name and name not in cte_names and name not in allowed_lower:
                raise UnsafeQueryError("table_not_allowed", f"Table '{tbl.name}' is not in the allowed list.")

    if forbidden_columns:
        forbidden_lower = {c.lower() for c in forbidden_columns}
        for col in tree.find_all(exp.Column):
            if col.name.lower() in forbidden_lower:
                raise UnsafeQueryError("forbidden_column", f"Column '{col.name}' is not allowed.")

    # Inject or cap LIMIT
    existing = tree.args.get("limit")
    if existing is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        try:
            val = int(existing.expression.this)
        except Exception:
            val = max_rows + 1
        if val > max_rows:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return tree.sql(dialect=dialect)


class Database:
    """Read-only SQLite (or other SQLAlchemy) database wrapper."""

    def __init__(
        self,
        path_or_url: str,
        dialect: str = "sqlite",
        timeout_s: float = 5.0,
        allowed_tables: list[str] | None = None,
        forbidden_columns: list[str] | None = None,
        max_rows: int = 100,
    ):
        # Accept a full SQLAlchemy URL (already has "://") or a plain file path
        if "://" in str(path_or_url):
            url = str(path_or_url)
            engine_kwargs: dict = {}
        elif dialect == "sqlite":
            url = f"sqlite:///file:{path_or_url}?mode=ro&uri=true"
            engine_kwargs = {"connect_args": {"check_same_thread": False}}
        else:
            url = path_or_url
            engine_kwargs = {}

        self.engine = create_engine(url, **engine_kwargs)
        self._engine = self.engine  # keep private alias for internal callers
        self._dialect = dialect
        self._timeout_s = timeout_s
        self._allowed_tables = allowed_tables or []
        self._forbidden_columns = forbidden_columns or []
        self._max_rows = max_rows

    def schema(self, tables: Optional[list[str]] = None) -> str:
        """
        Return DDL for all (or the given) tables as a single string.

        Used both by the graph (tables=[...]) and the tests (no args).
        """
        inspector = sa_inspect(self.engine)
        all_tables = inspector.get_table_names()
        target = [t for t in all_tables if t in tables] if tables else all_tables
        parts: list[str] = []
        for tbl in target:
            cols = inspector.get_columns(tbl)
            fks = inspector.get_foreign_keys(tbl)
            fk_map = {
                fk["constrained_columns"][0]: fk["referred_table"]
                for fk in fks if fk["constrained_columns"]
            }
            col_strs = []
            for c in cols:
                line = f"  {c['name']} {c['type']}"
                if not c.get("nullable", True):
                    line += " NOT NULL"
                if c["name"] in fk_map:
                    line += f" REFERENCES {fk_map[c['name']]}"
                col_strs.append(line)
            parts.append(f"CREATE TABLE {tbl}(\n" + ",\n".join(col_strs) + "\n);")
        return "\n\n".join(parts)

    def query(self, inp: SQLQueryInput) -> dict:
        """Execute a safe SELECT and return rows + row_count."""
        safe_sql = make_safe(
            inp.sql,
            dialect=self._dialect,
            max_rows=self._max_rows,
            allowed_tables=self._allowed_tables or None,
            forbidden_columns=self._forbidden_columns or None,
        )
        with self.engine.connect() as conn:
            if self._dialect == "sqlite" and self._timeout_s > 0:
                deadline = time.monotonic() + self._timeout_s
                raw = conn.connection.dbapi_connection

                def _abort():
                    return 1 if time.monotonic() > deadline else 0

                raw.set_progress_handler(_abort, 100)
            try:
                rows = conn.execute(text(safe_sql)).mappings().all()
            finally:
                if self._dialect == "sqlite" and self._timeout_s > 0:
                    raw.set_progress_handler(None, 0)
        result = [dict(r) for r in rows]
        return {"rows": result, "row_count": len(result)}

    def query_tool(self, sql: str) -> str:
        """Adapter for LangChain StructuredTool (returns JSON string)."""
        import json
        return json.dumps(self.query(SQLQueryInput(sql=sql)), default=str)
