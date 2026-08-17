"""单工具 invoke 计时。"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolNode

from llgraph.core.tool_invoke_timing import (
    attach_tool_timings_to_output,
    read_tool_message_elapsed,
    record_tool_timing,
    reset_tool_timings,
    wrap_tool_node_with_timing,
)


def test_attach_tool_timings_per_call_id() -> None:
    reset_tool_timings()
    record_tool_timing("call_a", 0.12)
    record_tool_timing("call_b", 0.34)
    out = attach_tool_timings_to_output(
        {
            "messages": [
                ToolMessage(content="a", tool_call_id="call_a", name="grep_files"),
                ToolMessage(content="b", tool_call_id="call_b", name="read_file"),
            ]
        }
    )
    msgs = out["messages"]
    assert read_tool_message_elapsed(msgs[0]) == 0.12
    assert read_tool_message_elapsed(msgs[1]) == 0.34


def test_wrap_tool_node_run_one_records_per_call() -> None:
    inner = ToolNode([])

    def slow_run(call, input_type, tool_runtime):
        if call["id"] == "t1":
            time.sleep(0.04)
        return ToolMessage(content="ok", tool_call_id=call["id"], name="x")

    inner._run_one = slow_run  # type: ignore[method-assign]
    wrap_tool_node_with_timing(inner)

    reset_tool_timings()
    inner._run_one({"id": "t1", "name": "x", "args": {}}, "list", MagicMock())
    inner._run_one({"id": "t2", "name": "x", "args": {}}, "list", MagicMock())

    out = attach_tool_timings_to_output(
        {
            "messages": [
                ToolMessage(content="1", tool_call_id="t1", name="x"),
                ToolMessage(content="2", tool_call_id="t2", name="x"),
            ]
        }
    )
    t1 = read_tool_message_elapsed(out["messages"][0])
    t2 = read_tool_message_elapsed(out["messages"][1])
    assert t1 is not None and t2 is not None
    assert t1 > t2
