"""工具节点响应 Web Stop。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from llgraph.context.runtime_context import set_active_thread_id
from llgraph.core.react_tools import build_tool_node
from llgraph.core.tools import get_agent_tools
from llgraph.console.runtime.agent_service import (
    clear_agent_cancel,
    is_agent_cancel_requested,
    request_agent_cancel,
)


def test_tool_node_skips_invoke_when_cancel_requested() -> None:
    tid = "cli-tool-cancel"
    clear_agent_cancel(tid)
    from llgraph.console.runtime import agent_service

    with agent_service._ACTIVE_AGENT_CHATS_LOCK:
        agent_service._ACTIVE_AGENT_CHATS.add(tid)
    set_active_thread_id(tid)
    try:
        assert request_agent_cancel(tid) is True
        node = build_tool_node(get_agent_tools(), workspace=None)
        state = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(
                    content="plan",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "grep_files",
                            "args": {"pattern": "foo", "path": "."},
                        }
                    ],
                ),
            ]
        }
        out = node.invoke(state, config={"configurable": {"thread_id": tid}})
        msgs = out.get("messages") or []
        assert len(msgs) == 1
        assert msgs[0].content == "[llgraph] 用户已停止当前生成。"
        assert msgs[0].tool_call_id == "call_1"
    finally:
        set_active_thread_id(None)
        clear_agent_cancel(tid)
        with agent_service._ACTIVE_AGENT_CHATS_LOCK:
            agent_service._ACTIVE_AGENT_CHATS.discard(tid)
