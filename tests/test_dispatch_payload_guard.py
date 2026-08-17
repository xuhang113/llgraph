"""出站 payload 校验与 thinking 占位修复。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from llgraph.context.chat_history_repair import (
    THINKING_ONLY_DISPATCH_TEXT,
    TOOL_ASSISTANT_DISPATCH_TEXT,
)
from llgraph.core.dispatch_payload_guard import validate_and_repair_formatted_messages
from llgraph.core.gateway_kimi_patch import inject_reasoning_into_formatted_messages


def test_repair_null_assistant_content() -> None:
    formatted = [{"role": "assistant", "content": None}]
    repaired, issues = validate_and_repair_formatted_messages(formatted)
    assert issues
    content = repaired[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == TOOL_ASSISTANT_DISPATCH_TEXT


def test_repair_whitespace_only_text_block() -> None:
    formatted = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": " "},
                {"type": "tool_use", "name": "grep", "input": {}, "id": "g1"},
            ],
        }
    ]
    repaired, issues = validate_and_repair_formatted_messages(formatted)
    blocks = repaired[0]["content"]
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert text_blocks
    assert text_blocks[-1]["text"] == TOOL_ASSISTANT_DISPATCH_TEXT


def test_repair_thinking_only_adds_text() -> None:
    formatted = [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "only think"}],
            "reasoning_content": "only think",
        }
    ]
    repaired, issues = validate_and_repair_formatted_messages(formatted)
    blocks = repaired[0]["content"]
    assert any(b.get("type") == "text" and b.get("text") == THINKING_ONLY_DISPATCH_TEXT for b in blocks)


def test_kimi_inject_then_guard_produces_valid_payload() -> None:
    ai = AIMessage(
        content="",
        additional_kwargs={
            "llgraph": {"thinking_text": "内部推理链"},
        },
        tool_calls=[{"id": "t1", "name": "grep_files", "args": {"pattern": "x"}}],
    )
    source = [HumanMessage(content="hi"), ai]
    formatted = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
    ]
    inject_reasoning_into_formatted_messages(source, formatted)
    repaired, _ = validate_and_repair_formatted_messages(formatted)
    content = repaired[1]["content"]
    assert isinstance(content, list)
    assert any(b.get("type") == "thinking" for b in content)
    assert any(
        b.get("type") == "text"
        and b.get("text") in (TOOL_ASSISTANT_DISPATCH_TEXT, THINKING_ONLY_DISPATCH_TEXT)
        for b in content
    )
