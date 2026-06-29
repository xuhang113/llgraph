"""search_code_parallel 同轮二次调用应被拦截。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from llgraph.core.tool_execution_context import (
    count_tool_results_since_user,
    set_tool_execution_messages,
)
from llgraph.core.code_index_tools import create_code_index_tools


def test_count_tool_results_since_user():
    msgs = [
        HumanMessage(content="q"),
        ToolMessage(content="ok", tool_call_id="1", name="search_code_parallel"),
    ]
    assert count_tool_results_since_user(msgs, "search_code_parallel") == 1


def test_second_parallel_search_blocked_in_same_turn(tmp_path: Path):
    tools = {t.name: t for t in create_code_index_tools(tmp_path)}
    parallel = tools["search_code_parallel"]
    prior = [
        HumanMessage(content="find auth"),
        ToolMessage(content="ok", tool_call_id="c1", name="search_code_parallel"),
    ]
    set_tool_execution_messages(prior)
    out = parallel.invoke({"query": "x", "top_k": 1})
    assert "拦截" in out
