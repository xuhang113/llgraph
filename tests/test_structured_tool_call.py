"""结构化 tool_use 入站校验单测。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from llgraph.adapters.inbound import UnstructuredToolCallError, normalize_ai_response
from llgraph.adapters.inbound.profile import InboundAdapterProfile


def test_structured_tool_calls_passes() -> None:
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "plan"},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "grep_files",
                "input": {"pattern": "x", "path": "."},
            },
        ],
        tool_calls=[
            {
                "name": "grep_files",
                "args": {"pattern": "x", "path": "."},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    out = normalize_ai_response(
        msg,
        None,
        "deepseek-v4-pro",
        profile=InboundAdapterProfile(parse_kimi_native_tool_tokens=False),
    )
    assert out.tool_calls


def test_xml_in_content_raises_without_structured_calls() -> None:
    msg = AIMessage(
        content=(
            "<tool_calls><tool_call name=\"grep_files\">"
            "<tool_call_args>{\"pattern\": \"x\"}</tool_call_args></tool_call></tool_calls>"
        )
    )
    with pytest.raises(UnstructuredToolCallError, match="结构化"):
        normalize_ai_response(msg, None, "deepseek-v4-pro")


def test_kimi_token_in_content_raises() -> None:
    sample = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin_kimi|>functions.grep_files:0"
        "<|tool_call_argument_begin|>{\"pattern\": \"x\"}"
        "<|tool_call_end_kimi|>"
        "<|tool_calls_section_end|>"
    )
    msg = AIMessage(content=sample)
    with pytest.raises(UnstructuredToolCallError):
        normalize_ai_response(msg, None, "kimi-k2.6")


def test_plain_text_reply_ok() -> None:
    msg = AIMessage(content="这是最终答复，无工具。")
    out = normalize_ai_response(msg, None, "deepseek-v4-pro")
    assert out.content == "这是最终答复，无工具。"
