from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.tools.emails import EmailIndex
from app.tools.database import Database
from app.tools.company_site import CompanySiteIndex


@dataclass
class Services:
    emails: EmailIndex
    database: Optional[Database] = None
    company_site: Optional[CompanySiteIndex] = None
    web: Optional[Callable] = None  # callable(WebSearchInput) -> dict, or None to use default


def build_tools(services: Services) -> list:
    """Return a list of LangChain StructuredTools for the v1 ReAct agent."""
    from langchain_core.tools import StructuredTool
    from app.tools.emails import EmailSearchInput
    from app.tools.database import SQLQueryInput
    from app.tools.web import WebSearchInput

    tools = []

    tools.append(
        StructuredTool.from_function(
            func=services.emails.search_tool,
            name="email_search",
            description="Search company emails by content, sender, or date.",
            args_schema=EmailSearchInput,
        )
    )

    if services.company_site:
        tools.append(
            StructuredTool.from_function(
                func=services.company_site.search_tool,
                name="company_website_search",
                description="Search the company website for information about products, services, or the team.",
            )
        )

    if services.database:
        tools.append(
            StructuredTool.from_function(
                func=services.database.query_tool,
                name="database_query",
                description="Run a read-only SQL SELECT on the company database.",
                args_schema=SQLQueryInput,
            )
        )

    web_fn = _make_web_fn(services.web)
    tools.append(
        StructuredTool.from_function(
            func=web_fn,
            name="web_search",
            description="Search the public internet for general or external knowledge.",
            args_schema=WebSearchInput,
        )
    )

    return tools


def _make_web_fn(custom_web: Callable | None):
    from app.tools.web import WebSearchInput, web_search as _default_web

    def _fn(**kwargs) -> str:
        inp = WebSearchInput(**kwargs)
        result = custom_web(inp) if custom_web else _default_web(inp)
        return json.dumps(result, default=str)

    return _fn


def to_json(obj) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str)
    return json.dumps(obj, default=str)
