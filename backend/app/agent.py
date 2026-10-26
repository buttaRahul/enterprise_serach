"""v1 ReAct agent — kept as the eval baseline behind AGENT_MODE=react."""
from __future__ import annotations
import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool


SYSTEM_PROMPT = """\
You are an enterprise search assistant. You have access to tools for searching
company emails, the employee/project database, the company website, and the web.

For each query:
1. Identify which tool(s) to use.
2. Call the tool(s) with appropriate inputs.
3. Synthesise the results into a concise, factual answer.

Always cite sources where possible. Never make up information not found in the tools.
"""

_MAX_TURNS = 6


def run_agent(query: str, services, llm, *, config: dict | None = None) -> dict:
    """
    Run the v1 ReAct agent using a simple tool-calling loop over langchain_core.

    Returns {"answer": str, "trace": list[dict]}.
    """
    from app.tools import build_tools

    lc_tools: list[StructuredTool] = build_tools(services)
    tool_map = {t.name: t for t in lc_tools}

    # llm here is our LLMProtocol wrapper; get the underlying chat model
    chat = llm._model if hasattr(llm, "_model") else llm
    bound = chat.bind_tools(lc_tools)

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    t0 = time.perf_counter()
    trace: list[dict] = []

    for _ in range(_MAX_TURNS):
        try:
            response: AIMessage = bound.invoke(messages)
        except Exception as exc:
            return {"answer": f"Error: {exc}", "trace": trace}

        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool is None:
                result = f"Unknown tool: {tc['name']}"
            else:
                try:
                    result = tool.invoke(tc["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"

            trace.append({
                "node": "react",
                "tool": tc["name"],
                "tool_input": tc["args"],
                "ms": int((time.perf_counter() - t0) * 1000),
            })
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            answer = msg.content
            break

    return {"answer": answer, "trace": trace}
