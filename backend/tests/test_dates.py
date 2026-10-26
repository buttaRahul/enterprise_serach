"""Tests for the date-hint extractor — offline."""
from __future__ import annotations
from datetime import date

import pytest

from app.graph.dates import extract_date_hints


TODAY = date(2025, 2, 25)


def hints(query: str) -> dict:
    return extract_date_hints(query, today=TODAY)


class TestDateHints:
    def test_iso_date(self):
        h = hints("emails before 2025-02-18")
        assert "2025-02-18" in h["dates"]
        assert h["relations"][h["dates"].index("2025-02-18")] == "before"

    def test_dd_mm_yyyy(self):
        h = hints("emails on 18-02-2025")
        assert "2025-02-18" in h["dates"]

    def test_named_month_full(self):
        h = hints("emails before 18th February 2025")
        assert "2025-02-18" in h["dates"]
        assert h["relations"][0] == "before"

    def test_named_month_abbrev(self):
        h = hints("after Feb 18, 2025")
        assert "2025-02-18" in h["dates"]
        assert h["relations"][0] == "after"

    def test_named_month_dmy(self):
        h = hints("18 Feb 2025")
        assert "2025-02-18" in h["dates"]

    def test_earlier_than(self):
        h = hints("emails earlier than 18th February 2025")
        assert h["relations"][0] == "before"

    def test_since(self):
        h = hints("since 2025-02-10")
        assert h["relations"][0] == "after"

    def test_today(self):
        h = hints("emails sent today")
        assert TODAY.isoformat() in h["dates"]

    def test_yesterday(self):
        h = hints("emails from yesterday")
        from datetime import timedelta
        assert (TODAY - timedelta(days=1)).isoformat() in h["dates"]

    def test_last_week(self):
        h = hints("emails from last week")
        assert len(h["dates"]) >= 1

    def test_no_dates(self):
        h = hints("find emails about pull requests")
        assert h["dates"] == []

    def test_on_relation(self):
        h = hints("emails on 2025-02-21")
        assert h["relations"][0] == "on"


class TestChitchatDetection:
    def test_hello_is_chitchat(self):
        from app.graph.nodes.preprocess import _is_chitchat
        assert _is_chitchat("hello")
        assert _is_chitchat("hi there")
        assert _is_chitchat("thanks!")

    def test_long_message_not_chitchat(self):
        from app.graph.nodes.preprocess import _is_chitchat
        assert not _is_chitchat("find emails about pull requests from last week")

    def test_greeting_too_long(self):
        from app.graph.nodes.preprocess import _is_chitchat
        assert not _is_chitchat("hello can you find me some emails about infrastructure")
