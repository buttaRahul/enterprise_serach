from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from app.tools.emails import EmailIndex
from app.tools.database import Database
from app.tools.company_site import CompanySiteIndex
from app.tools.web import web_search as _web_search


@dataclass
class Services:
    emails: EmailIndex
    database: Optional[Database] = None
    company_site: Optional[CompanySiteIndex] = None


def build_tools(services: Services) -> list:
    """Return a list of LangChain StructuredTools for the ReAct agent."""
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

    if services.database:
        tools.append(
            StructuredTool.from_function(
                func=services.database.query_tool,
                name="db_query",
                description="Run a read-only SQL SELECT on the company database (employees, projects).",
                args_schema=SQLQueryInput,
            )
        )

    if services.company_site:
        tools.append(
            StructuredTool.from_function(
                func=services.company_site.search_tool,
                name="site_search",
                description="Search the company website for information about products, services, or the team.",
            )
        )

    tools.append(
        StructuredTool.from_function(
            func=_web_search_tool,
            name="web_search",
            description="Search the public internet for general or external knowledge.",
            args_schema=WebSearchInput,
        )
    )

    return tools


def _web_search_tool(query: str) -> str:
    from app.tools.web import WebSearchInput
    result = _web_search(WebSearchInput(query=query))
    return str(result)


def to_json(obj) -> str:
    import json
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str)
    return json.dumps(obj, default=str)
