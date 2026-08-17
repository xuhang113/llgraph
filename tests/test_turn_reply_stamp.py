"""轮末答复写回 messages（避免仅占位符）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import (
    TOOL_ASSISTANT_DISPATCH_TEXT,
    stamp_turn_reply_on_messages,
)


def test_stamp_turn_reply_replaces_placeholder_on_last_ai() -> None:
    messages = [
        HumanMessage(content="review 分支"),
        AIMessage(content=TOOL_ASSISTANT_DISPATCH_TEXT, tool_calls=[{"name": "grep_files", "args": {}, "id": "1"}]),
        ToolMessage(content="hit", tool_call_id="1", name="grep_files"),
        AIMessage(content=TOOL_ASSISTANT_DISPATCH_TEXT),
    ]
    out, changed = stamp_turn_reply_on_messages(messages, "## Review 结论\n\n1. 问题 A")
    assert changed is True
    last = out[-1]
    assert isinstance(last, AIMessage)
    assert "Review 结论" in str(last.content)
    meta = (last.additional_kwargs or {}).get("llgraph") or {}
    assert meta.get("turn_reply_text")


def test_stamp_turn_reply_skips_when_real_content_exists() -> None:
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="已有正文"),
    ]
    out, changed = stamp_turn_reply_on_messages(messages, "新正文")
    assert changed is False
    assert out[-1].content == "已有正文"
