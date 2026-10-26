import pytest

from app.config import BACKEND_DIR
from app.tools.emails import EmailIndex, EmailSearchInput


@pytest.fixture
def index(embedder):
    return EmailIndex.from_json(BACKEND_DIR / "data" / "emails.json", embedder, min_score=0.1)


def search(index, **kw):
    return index.search(EmailSearchInput(**kw))


def test_requires_a_filter(index):
    with pytest.raises(ValueError):
        search(index)


def test_sender_matches_address_or_name(index):
    assert [e["sender"] for e in search(index, sender="RAMESH@company.com")["emails"]] == ["ramesh@company.com"]
    assert {e["sender"] for e in search(index, sender="raghu")["emails"]} == {"raghu@company.com"}


def test_date_filters_are_strict(index):
    on = search(index, sent_on="2025-02-19")["emails"]
    before = search(index, sent_before="2025-02-19")["emails"]
    after = search(index, sent_after="2025-02-20")["emails"]
    assert on and all(e["timestamp"].startswith("2025-02-19") for e in on)
    assert before and all(e["timestamp"] < "2025-02-19" for e in before)
    assert after and all(e["timestamp"] >= "2025-02-21" for e in after)


def test_filters_are_anded_without_fallback(index):
    # ramesh only sent on 2025-02-18: combining with another date must return nothing, not all emails
    assert search(index, sender="ramesh", sent_on="2025-02-21")["count"] == 0
    assert search(index, sender="nobody@company.com", content_query="pull request")["count"] == 0


def test_content_ranking_and_top_k(index):
    res = search(index, content_query="pull request review", top_k=3)["emails"]
    assert 1 <= len(res) <= 3
    assert all("pull request" in e["content"].lower() for e in res)
    assert [e["score"] for e in res] == sorted((e["score"] for e in res), reverse=True)


def test_bad_date_is_reported(index):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        search(index, sent_on="18th Feb")


def test_embeddings_computed_once(index):
    calls = []
    inner = index.embedder
    index.embedder = lambda texts: calls.append(len(texts)) or inner(texts)
    search(index, content_query="deployment")
    search(index, content_query="security")
    assert calls.count(len(index.emails)) == 1   # corpus embedded once; then one call per query
