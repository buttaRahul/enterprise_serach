"""v1 ReAct agent — kept as the eval baseline behind AGENT_MODE=react."""
from __future__ import annotations
import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


SYSTEM_PROMPT = """\
You are an enterprise search assistant. You have access to tools for searching
company emails, the employee/project database, the company website, and the web.

For each query:
1. Identify which tool(s) to use.
2. Call the tool(s) with appropriate inputs.
3. Synthesise the results into a concise, factual answer.

Always cite sources where possible. Never make up information not found in the tools.
"""

_DEFAULT_MAX_STEPS = 6


def run_agent(
    query: str,
    llm,
    tools: list,
    *,
    max_steps: int = _DEFAULT_MAX_STEPS,
    config: dict | None = None,
) -> dict:
    """
    Run the v1 ReAct agent using a langchain_core tool-calling loop.

    llm may be our LLMProtocol wrapper (has .chat) or a raw LangChain chat model.
    Returns {"answer": str, "trace": list[dict], "truncated": bool}.
    """
    from app.tools.database import UnsafeQueryError

    # Accept both our wrapper (llm.chat) and a bare chat model
    chat = getattr(llm, "chat", llm)
    bound = chat.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    t0 = time.perf_counter()
    trace: list[dict] = []
    truncated = False

    # max_steps caps the number of tool calls, not LLM turns
    while True:
        response: AIMessage = bound.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool = tool_map.get(tc["name"])
            ms = int((time.perf_counter() - t0) * 1000)

            if tool is None:
                status = "error"
                result: dict = {"error": f"Unknown tool: {tc['name']}"}
            else:
                try:
                    raw = tool.invoke(tc["args"])
                    result = json.loads(raw) if isinstance(raw, str) else raw
                    status = "ok"
                except UnsafeQueryError as exc:
                    status = "error"
                    result = {"error": str(exc)}
                    raw = json.dumps(result)
                except Exception as exc:
                    status = "error"
                    result = {"error": str(exc)}
                    raw = json.dumps(result)

            trace.append({
                "tool": tc["name"],
                "tool_input": tc["args"],
                "status": status,
                "result": result,
                "ms": ms,
            })
            messages.append(ToolMessage(
                content=raw if isinstance(raw, str) else json.dumps(raw, default=str),
                tool_call_id=tc["id"],
            ))

        if len(trace) >= max_steps:
            truncated = True
            break

    if truncated:
        # One final LLM call to get a text answer after budget exhaustion
        final: AIMessage = bound.invoke(messages)
        answer = final.content if isinstance(final.content, str) else str(final.content)
    else:
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

    return {"answer": answer, "trace": trace, "truncated": truncated}
