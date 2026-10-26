from __future__ import annotations
import re
import textwrap
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
    Returns the sanitised SQL string (with LIMIT injected if missing).
    """
    # Strip trailing semicolons to check for multi-statement
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

    # Collect CTE names so they are not flagged as disallowed tables
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

    # Inject LIMIT if missing
    if tree.args.get("limit") is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return tree.sql(dialect=dialect)


class Database:
    """Read-only SQLite (or other SQLAlchemy) database wrapper."""

    def __init__(
        self,
        path: str,
        dialect: str = "sqlite",
        timeout_s: float = 5.0,
        allowed_tables: list[str] | None = None,
        forbidden_columns: list[str] | None = None,
        max_rows: int = 100,
    ):
        if dialect == "sqlite":
            url = f"sqlite:///file:{path}?mode=ro&uri=true"
            engine_kwargs = {"connect_args": {"check_same_thread": False}}
        else:
            url = path
            engine_kwargs = {}
        self._engine = create_engine(url, **engine_kwargs)
        self._dialect = dialect
        self._timeout_s = timeout_s
        self._allowed_tables = allowed_tables or []
        self._forbidden_columns = forbidden_columns or []
        self._max_rows = max_rows

    def schema(self, tables: Optional[list[str]] = None) -> dict[str, str]:
        """
        Return a dict mapping table_name -> DDL-like column description string.
        Optionally filter to the given table names.
        """
        inspector = sa_inspect(self._engine)
        result: dict[str, str] = {}
        all_tables = inspector.get_table_names()
        target = [t for t in all_tables if t in tables] if tables else all_tables
        for tbl in target:
            cols = inspector.get_columns(tbl)
            fks = inspector.get_foreign_keys(tbl)
            fk_map = {fk["constrained_columns"][0]: fk["referred_table"] for fk in fks if fk["constrained_columns"]}
            col_strs = []
            for c in cols:
                line = f"  {c['name']} {c['type']}"
                if not c.get("nullable", True):
                    line += " NOT NULL"
                if c["name"] in fk_map:
                    line += f" REFERENCES {fk_map[c['name']]}"
                col_strs.append(line)
            result[tbl] = f"CREATE TABLE {tbl} (\n" + ",\n".join(col_strs) + "\n);"
        return result

    def query(self, inp: SQLQueryInput) -> dict:
        """Execute a safe SELECT and return rows as a list of dicts."""
        safe_sql = make_safe(
            inp.sql,
            dialect=self._dialect,
            max_rows=self._max_rows,
            allowed_tables=self._allowed_tables or None,
            forbidden_columns=self._forbidden_columns or None,
        )
        with self._engine.connect() as conn:
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
            return {"rows": [dict(r) for r in rows]}

    def query_tool(self, sql: str) -> str:
        """Adapter for LangChain StructuredTool."""
        import json
        result = self.query(SQLQueryInput(sql=sql))
        return json.dumps(result, default=str)
