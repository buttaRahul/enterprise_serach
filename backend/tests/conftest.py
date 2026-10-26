"""Shared test fixtures and fakes (all offline — no network, no API keys)."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import tempfile
from typing import Any

import numpy as np
import pytest

from app.graph.state import Plan, Step, Evidence, ToolResult


# ---------------------------------------------------------------------------
# Deterministic embedder (no network)
# ---------------------------------------------------------------------------

def hashing_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic fake embedder — same text always gives same vector."""
    dim = 64
    result = []
    for t in texts:
        raw = hashlib.sha256(t.encode()).digest()
        arr = np.frombuffer(raw * (dim // 32 + 1), dtype=np.uint8)[:dim].astype(np.float32)
        norm = np.linalg.norm(arr)
        result.append(arr / (norm + 1e-9))
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# ScriptedLLM — queued responses per call name
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """
    Returns queued responses per call name; records every call for assertions.

    Usage:
        llm = ScriptedLLM({
            "understand": [Plan(standalone_query="...", intent="search", steps=[...])],
            "synthesize": ["The answer is 42."],
        })
    """

    def __init__(self, script: dict[str, list]):
        self.script: dict[str, list] = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, list]] = []

    async def structured(self, schema, messages, *, name: str):
        self.calls.append((name, messages))
        out = self.script[name].pop(0)
        assert isinstance(out, schema), f"ScriptedLLM {name}: expected {schema.__name__}, got {type(out).__name__}"
        return out

    async def text(self, messages, *, name: str) -> str:
        self.calls.append((name, messages))
        return self.script[name].pop(0)


# ---------------------------------------------------------------------------
# ScriptedChatModel — LangChain-compatible fake for v1 tests
# ---------------------------------------------------------------------------

class _FakeAIMessage:
    def __init__(self, content: str):
        self.content = content


def tool_call(name: str, args: dict) -> dict:
    """Build a fake LangChain tool-call dict."""
    return {"name": name, "args": args, "id": f"call_{name}"}


class ScriptedChatModel:
    """
    Minimal LangChain ChatModel fake that returns scripted strings.
    Not async — used only for synchronous v1 agent tests.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def bind_tools(self, tools):  # noqa: ARG002
        return self

    def invoke(self, messages, **_):
        return _FakeAIMessage(self._responses.pop(0))

    def with_structured_output(self, schema):
        parent = self

        class _Bound:
            def invoke(self, messages, **__):
                raw = parent._responses.pop(0)
                if isinstance(raw, str):
                    return schema(**json.loads(raw))
                return raw

        return _Bound()


# ---------------------------------------------------------------------------
# In-memory SQLite fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_db(tmp_path):
    """Seed a temporary SQLite database and return its path."""
    from scripts.seed_demo_db import seed
    db_path = str(tmp_path / "demo.db")
    seed(db_path)
    return db_path


@pytest.fixture
def database(demo_db):
    from app.tools.database import Database
    return Database(demo_db)


# ---------------------------------------------------------------------------
# Email fixtures
# ---------------------------------------------------------------------------

SAMPLE_EMAILS = [
    {"id": "e00", "timestamp": "2025-02-21 09:15:32", "sender": "rahul@company.com",
     "content": "Pull Request #10 has been raised for the latest bug fix on validation error."},
    {"id": "e01", "timestamp": "2025-02-21 10:05:14", "sender": "rajesh@company.com",
     "content": "There is an issue with the API response time. Ticket #5678 raised."},
    {"id": "e02", "timestamp": "2025-02-20 11:32:45", "sender": "ram@company.com",
     "content": "Assigning task #11 to Rahul for UI improvements. Deadline is Friday."},
    {"id": "e03", "timestamp": "2025-02-19 17:44:12", "sender": "roshan@company.com",
     "content": "Assigning ticket #15 to the DevOps team for infrastructure scaling."},
    {"id": "e04", "timestamp": "2025-02-18 19:30:15", "sender": "ramesh@company.com",
     "content": "Research obstacle detection in autonomous cars for the upcoming project."},
]


@pytest.fixture
def email_index():
    from app.tools.emails import EmailIndex
    return EmailIndex(SAMPLE_EMAILS, hashing_embedder)


@pytest.fixture
def services(email_index, database):
    from app.tools import Services
    return Services(emails=email_index, database=database)


@pytest.fixture
def services_no_db(email_index):
    from app.tools import Services
    return Services(emails=email_index, database=None, company_site=None)
