"""全模型入站 tool markup 剥离与展示清洗单测。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from llgraph.adapters.inbound import UnstructuredToolCallError, normalize_ai_response, strip_inbound_tool_call_markup
from llgraph.adapters.inbound.profile import InboundAdapterProfile
from llgraph.context.message_normalize import format_agent_chat_display_text

_KIMI_TOKEN_SAMPLE = (
    "<|tool_calls_section_begin|>"
    "<|tool_call_begin_kimi|>functions.grep_files:0"
    "<|tool_call_argument_begin|>{\"pattern\": \"x\"}"
    "<|tool_call_end_kimi|>"
    "<|tool_calls_section_end|>"
)

_PLAIN_SAMPLE = (
    'functions.grep_files:1{"path": ".", "pattern": "Foo"}'
    "【规划】检索实现类。"
)

_XML_QWEN_SAMPLE = (
    "【规划】并行检索。\n"
    "<tool_call>\n"
    "<function=glob_files>\n"
    "<parameter=path>\n"
    ".\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_strip_inbound_covers_kimi_plain_xml() -> None:
    assert strip_inbound_tool_call_markup(_KIMI_TOKEN_SAMPLE) == ""
    assert strip_inbound_tool_call_markup(_PLAIN_SAMPLE) == "【规划】检索实现类。"
    assert strip_inbound_tool_call_markup(_XML_QWEN_SAMPLE) == "【规划】并行检索。"


def test_format_agent_chat_strips_plan_and_tools() -> None:
    assert format_agent_chat_display_text(_PLAIN_SAMPLE) == ""
    assert format_agent_chat_display_text("正常结论\n【规划】中间步骤") == "正常结论"
    assert format_agent_chat_display_text(_XML_QWEN_SAMPLE) == ""


def test_plain_functions_in_content_raises_without_structured_calls() -> None:
    msg = AIMessage(content=_PLAIN_SAMPLE)
    with pytest.raises(UnstructuredToolCallError):
        normalize_ai_response(
            msg,
            None,
            "claude-sonnet-4-6",
            profile=InboundAdapterProfile(parse_kimi_native_tool_tokens=False),
        )


def test_xml_in_content_raises_without_structured_calls() -> None:
    msg = AIMessage(content=_XML_QWEN_SAMPLE)
    with pytest.raises(UnstructuredToolCallError):
        normalize_ai_response(
            msg,
            None,
            "gpt-4o",
            profile=InboundAdapterProfile(parse_kimi_native_tool_tokens=False),
        )


def test_kimi_token_raises_even_when_profile_enabled() -> None:
    msg = AIMessage(content=_KIMI_TOKEN_SAMPLE)
    with pytest.raises(UnstructuredToolCallError):
        normalize_ai_response(
            msg,
            None,
            "kimi-k2.6",
            profile=InboundAdapterProfile(parse_kimi_native_tool_tokens=True),
        )


def test_plain_strip_when_structured_tool_calls_present() -> None:
    leak = (
        'functions.grep_files:9{"pattern": "x"}'
        "最终答复正文。"
    )
    msg = AIMessage(
        content=leak,
        tool_calls=[
            {
                "name": "grep_files",
                "args": {"pattern": "x"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    out = normalize_ai_response(
        msg,
        None,
        "gpt-4o",
        profile=InboundAdapterProfile(parse_kimi_native_tool_tokens=False),
    )
    assert out.tool_calls
    assert "functions.grep_files" not in str(out.content)
    assert format_agent_chat_display_text(str(out.content)) == "最终答复正文。"
