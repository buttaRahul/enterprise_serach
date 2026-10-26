from __future__ import annotations
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for the public web")
    max_results: int = Field(4, description="Maximum number of results")


def web_search(inp: WebSearchInput) -> dict:
    """Search the public web via DuckDuckGo. Returns {"results": [{title, url, snippet}]}."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(inp.query, max_results=inp.max_results))
        results = [
            {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
            for h in hits
        ]
        return {"results": results}
    except Exception as exc:
        return {"results": [], "error": str(exc)}
