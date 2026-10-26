from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import run_agent
from app.config import BACKEND_DIR
from app.main import create_app
from app.tools import Services, build_tools
from app.tools.company_site import CompanySiteIndex
from app.tools.database import Database
from app.tools.emails import EmailIndex
from tests.conftest import ScriptedChatModel, hashing_embedder, tool_call

RO_URL = f"sqlite:///file:{BACKEND_DIR / 'data' / 'demo.db'}?mode=ro&uri=true"


def services():
    pages = {"https://example.com/": "Acme builds computer vision products.\n\nWe also offer NLP consulting."}
    return Services(
        emails=EmailIndex.from_json(BACKEND_DIR / "data" / "emails.json", hashing_embedder, min_score=0.1),
        company_site=CompanySiteIndex(list(pages), hashing_embedder, fetch=pages.__getitem__, chunk_size=60,
                                      chunk_overlap=0),
        database=Database(RO_URL),
        web=lambda args: {"results": [{"title": "Fixing slow APIs", "url": "https://example.org/a", "snippet": "..."}]},
    )


def test_all_tools_registered():
    assert [t.name for t in build_tools(services())] == [
        "email_search", "company_website_search", "database_query", "web_search"]


def test_multi_step_tool_use_and_trace():
    llm = ScriptedChatModel(responses=[
        tool_call("email_search", {"content_query": "API response time issue"}, "c1"),
        tool_call("web_search", {"query": "reduce API response time"}, "c2"),
        AIMessage(content="Rajesh reported slow API responses; see [Fixing slow APIs](https://example.org/a)."),
    ], calls=[])
    out = run_agent("find emails about API issues and search the web for fixes", llm, build_tools(services()))
    assert [t["tool"] for t in out["trace"]] == ["email_search", "web_search"]
    assert all(t["status"] == "ok" for t in out["trace"])
    assert out["trace"][0]["result"]["emails"][0]["sender"] == "rajesh@company.com"
    assert "Fixing slow APIs" in out["answer"]
    # tool results were fed back to the model
    assert isinstance(llm.calls[1][-1], ToolMessage)


def test_tool_errors_are_returned_to_model():
    llm = ScriptedChatModel(responses=[
        tool_call("database_query", {"sql": "DROP TABLE employees"}),
        AIMessage(content="I can only run read-only queries."),
    ], calls=[])
    out = run_agent("delete all employees", llm, build_tools(services()))
    assert out["trace"][0]["status"] == "error"
    assert "Only SELECT" in out["trace"][0]["result"]["error"]


def test_company_site_rag():
    llm = ScriptedChatModel(responses=[
        tool_call("company_website_search", {"query": "NLP consulting"}),
        AIMessage(content="They offer NLP consulting."),
    ], calls=[])
    out = run_agent("does the company do NLP?", llm, build_tools(services()))
    assert "NLP consulting" in out["trace"][0]["result"]["passages"][0]["text"]


def test_step_budget():
    llm = ScriptedChatModel(responses=[tool_call("email_search", {"sender": "ram"}, f"c{i}") for i in range(2)]
                            + [AIMessage(content="final")], calls=[])
    out = run_agent("loop", llm, build_tools(services()), max_steps=2)
    assert out["truncated"] and out["answer"] == "final" and len(out["trace"]) == 2


def test_api_endpoint():
    llm = ScriptedChatModel(responses=[
        tool_call("database_query", {"sql": "SELECT count(*) AS n FROM employees WHERE department='Security'"}),
        AIMessage(content="There are 2 people in Security."),
    ], calls=[])
    with TestClient(create_app(services=services(), llm=llm)) as client:
        assert client.get("/api/health").json()["tools"][0] == "email_search"
        body = client.post("/api/search", json={"query": "how many people are in security?"}).json()
        assert body["answer"] == "There are 2 people in Security."
        assert body["trace"][0]["result"]["rows"] == [{"n": 2}]
        assert client.post("/api/search", json={"query": ""}).status_code == 422
