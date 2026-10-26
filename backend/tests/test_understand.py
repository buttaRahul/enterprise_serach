"""Tests for the understand node — offline."""
from __future__ import annotations
import asyncio

import pytest

from tests.conftest import ScriptedLLM
from app.graph.nodes.understand import make_understand, route_after_understand
from app.graph.state import Plan, Step, turn_input
from app.tools import Services
from app.tools.emails import EmailIndex
from tests.conftest import hashing_embedder, SAMPLE_EMAILS


def _services(with_db=False, with_site=False):
    svc = Services(emails=EmailIndex(SAMPLE_EMAILS, hashing_embedder))
    if with_db:
        # mock — just set a truthy value
        svc.database = object()
    if with_site:
        svc.company_site = object()
    return svc


class TestUnderstandDropsUnavailableSources:
    def test_drops_site_when_not_configured(self):
        plan = Plan(
            standalone_query="company products",
            intent="search",
            steps=[
                Step(id="s1", source="site", query="products"),
                Step(id="s2", source="email", query="products"),
            ],
        )
        llm = ScriptedLLM({"understand": [plan]})
        understand = make_understand(llm, _services(with_db=False, with_site=False))

        async def run():
            return await understand(turn_input("company products"))

        result = asyncio.run(run())
        sources = [s.source for s in result["plan"].steps]
        assert "site" not in sources
        assert "email" in sources

    def test_drops_sql_when_no_db(self):
        plan = Plan(
            standalone_query="employees",
            intent="search",
            steps=[
                Step(id="s1", source="sql", query="employees"),
                Step(id="s2", source="email", query="team"),
            ],
        )
        llm = ScriptedLLM({"understand": [plan]})
        understand = make_understand(llm, _services())

        async def run():
            return await understand(turn_input("employees"))

        result = asyncio.run(run())
        sources = [s.source for s in result["plan"].steps]
        assert "sql" not in sources

    def test_keeps_sql_when_db_configured(self):
        plan = Plan(
            standalone_query="employees",
            intent="search",
            steps=[Step(id="s1", source="sql", query="employees")],
        )
        llm = ScriptedLLM({"understand": [plan]})
        understand = make_understand(llm, _services(with_db=True))

        async def run():
            return await understand(turn_input("employees"))

        result = asyncio.run(run())
        sources = [s.source for s in result["plan"].steps]
        assert "sql" in sources


class TestUnderstandSplitsWaves:
    def test_wave0_and_pending(self):
        plan = Plan(
            standalone_query="who sent the security email and what dept",
            intent="search",
            steps=[
                Step(id="s1", source="email", query="security patch"),
                Step(id="s2", source="sql", query="dept", depends_on="s1"),
            ],
        )
        llm = ScriptedLLM({"understand": [plan]})
        understand = make_understand(llm, _services(with_db=True))

        async def run():
            return await understand(turn_input("..."))

        result = asyncio.run(run())
        plan_out = result["plan"]
        pending = result["pending_steps"]

        wave0 = [s for s in plan_out.steps if not s.depends_on]
        assert len(wave0) == 1
        assert wave0[0].id == "s1"
        assert len(pending) == 1
        assert pending[0].id == "s2"


class TestBadDateBecomesErrorResult:
    def test_invalid_date_creates_tool_result(self):
        plan = Plan(
            standalone_query="emails",
            intent="search",
            steps=[
                Step(id="s1", source="email", query="emails", sent_before="not-a-date"),
            ],
        )
        llm = ScriptedLLM({"understand": [plan]})
        understand = make_understand(llm, _services())

        async def run():
            return await understand(turn_input("emails"))

        result = asyncio.run(run())
        bad_results = result.get("results", [])
        assert any(r.error_kind == "bad_args" for r in bad_results)
        # Step with bad date is dropped from plan
        assert all(s.sent_before != "not-a-date" for s in result["plan"].steps)
