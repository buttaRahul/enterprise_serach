import pytest
from sqlalchemy import exc

from app.config import BACKEND_DIR
from app.tools.database import Database, SQLQueryInput, UnsafeQueryError, make_safe

RO_URL = f"sqlite:///file:{BACKEND_DIR / 'data' / 'demo.db'}?mode=ro&uri=true"


@pytest.fixture(scope="module")
def db():
    return Database(RO_URL, max_rows=5)


@pytest.mark.parametrize("sql", [
    "DROP TABLE employees",
    "DELETE FROM employees",
    "UPDATE employees SET title = 'CEO'",
    "INSERT INTO employees (id) VALUES (99)",
    "SELECT 1; DROP TABLE employees",
    "SELECT * INTO backup FROM employees",
    "PRAGMA writable_schema = 1",
    "not sql at all ((",
])
def test_rejects_non_select(sql):
    with pytest.raises(UnsafeQueryError):
        make_safe(sql, "sqlite", 50)


def test_adds_and_caps_limit():
    assert make_safe("SELECT * FROM employees", "sqlite", 50).endswith("LIMIT 50")
    assert make_safe("SELECT * FROM employees LIMIT 10", "sqlite", 50).endswith("LIMIT 10")
    assert make_safe("SELECT * FROM employees LIMIT 1000", "sqlite", 50).endswith("LIMIT 50")


def test_schema_and_query(db):
    assert "employees(" in db.schema() and "projects(" in db.schema()
    res = db.query(SQLQueryInput(sql="SELECT first_name FROM employees WHERE department = 'DevOps' ORDER BY id"))
    assert [r["first_name"] for r in res["rows"]] == ["Ramu", "Roshan"]
    assert db.query(SQLQueryInput(sql="SELECT * FROM employees"))["row_count"] == 5   # max_rows cap


def test_connection_itself_is_read_only(db):
    # Even if the parser guard were bypassed, the SQLite connection is opened read-only.
    from sqlalchemy import text
    with pytest.raises(exc.OperationalError):
        with db.engine.begin() as conn:
            conn.execute(text("DELETE FROM employees"))
