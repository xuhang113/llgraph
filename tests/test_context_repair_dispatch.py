"""tool_call_id 修链测试。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import (
    _canonical_tool_call_id,
    rebuild_provider_safe_messages,
)


def test_canonical_tool_call_id_matches_kimi_variants() -> None:
    assert _canonical_tool_call_id("functions_read_files_2") == "functions_read_files_2"
    assert _canonical_tool_call_id("functions.read_files:2") == "functions_read_files_2"


def test_rebuild_does_not_patch_missing_read_files_result() -> None:
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_files",
                "args": {"paths": ["a.md"]},
                "id": "functions_read_files_2",
                "type": "tool_call",
            }
        ],
    )
    tool = ToolMessage(
        content="file body",
        tool_call_id="functions.read_files:2",
        name="read_files",
    )
    safe, report = rebuild_provider_safe_messages([HumanMessage(content="hi"), ai, tool])
    assert report.patched_tool_results == 0
    tool_msgs = [m for m in safe if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "未完成" not in str(tool_msgs[0].content)
