"""v1 ReAct agent — kept as the eval baseline behind AGENT_MODE=react."""
from __future__ import annotations
import time
from typing import Any

from langchain_core.messages import SystemMessage
from langchain.agents import AgentType, initialize_agent


SYSTEM_PROMPT = """\
You are an enterprise search assistant. You have access to tools for searching
company emails, the employee/project database, the company website, and the web.

For each query:
1. Identify which tool(s) to use.
2. Call the tool(s) with appropriate inputs.
3. Synthesise the results into a concise, factual answer.

Always cite sources where possible. Never make up information not found in the tools.
"""


def run_agent(query: str, services, llm, *, config: dict | None = None) -> dict:
    """
    Run the v1 ReAct agent.

    Returns {"answer": str, "trace": list[dict]}.
    """
    from app.tools import build_tools

    tools = build_tools(services)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    t0 = time.perf_counter()
    try:
        resp: dict[str, Any] = agent.invoke(query)
    except Exception as exc:
        return {"answer": f"Error: {exc}", "trace": []}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    trace = [
        {
            "node": "react",
            "tool": step[0].tool,
            "tool_input": step[0].tool_input,
            "ms": elapsed_ms,
        }
        for step in resp.get("intermediate_steps", [])
    ]

    return {"answer": resp.get("output", ""), "trace": trace}
