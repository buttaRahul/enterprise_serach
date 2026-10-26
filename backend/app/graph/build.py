"""Build and compile the main LangGraph."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.check_citations import check_citations
from app.graph.nodes.clarify import clarify
from app.graph.nodes.preprocess import preprocess, route_after_preprocess
from app.graph.nodes.reflect import make_reflect, route_after_reflect
from app.graph.nodes.respond_direct import make_respond_direct
from app.graph.nodes.retrieve import (
    make_email_retrieve,
    make_site_retrieve,
    make_sql_answer,
    make_web_retrieve,
)
from app.graph.nodes.summarize import make_summarize
from app.graph.nodes.synthesize import make_synthesize
from app.graph.nodes.understand import make_understand, route_after_understand
from app.graph.state import GraphState


def build_graph(services, llm, checkpointer=None, settings=None, feedback_store=None):
    """Compile and return the main StateGraph."""
    if settings is None:
        from app.config import settings as _settings
        settings = _settings

    available_sources: list[str] = ["email"]
    if services.database:
        available_sources.append("sql")
    if services.company_site:
        available_sources.append("site")
    available_sources.append("web")

    g = StateGraph(GraphState)

    g.add_node("preprocess", preprocess)
    g.add_node("summarize", make_summarize(llm))
    g.add_node("understand", make_understand(llm, services, max_steps=settings.max_steps))
    g.add_node("clarify", clarify)
    g.add_node("respond_direct", make_respond_direct(llm))
    g.add_node("email_retrieve", make_email_retrieve(services))
    g.add_node("site_retrieve", make_site_retrieve(services))
    g.add_node("web_retrieve", make_web_retrieve(services))
    g.add_node("sql_answer", make_sql_answer(services, llm, settings, feedback_store))
    g.add_node("reflect", make_reflect(llm, available_sources))
    g.add_node("synthesize", make_synthesize(llm, settings.max_evidence_chars))
    g.add_node("check_citations", check_citations)

    g.add_edge(START, "preprocess")
    g.add_conditional_edges(
        "preprocess", route_after_preprocess, ["understand", "summarize", "respond_direct"]
    )
    g.add_edge("summarize", "understand")
    g.add_conditional_edges(
        "understand", route_after_understand,
        ["clarify", "respond_direct", "email_retrieve", "site_retrieve", "web_retrieve", "sql_answer", "synthesize"],
    )
    g.add_edge("clarify", "understand")

    for node in ["email_retrieve", "site_retrieve", "web_retrieve", "sql_answer"]:
        g.add_edge(node, "reflect")

    g.add_conditional_edges(
        "reflect", route_after_reflect,
        ["synthesize", "email_retrieve", "site_retrieve", "web_retrieve", "sql_answer"],
    )
    g.add_edge("synthesize", "check_citations")
    g.add_edge("check_citations", END)
    g.add_edge("respond_direct", END)

    return g.compile(checkpointer=checkpointer)
