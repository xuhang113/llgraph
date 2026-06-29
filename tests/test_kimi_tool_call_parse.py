"""Kimi 原生 tool call token 解析单测。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from llgraph.adapters.inbound import parse_kimi_native_tool_calls, strip_inbound_tool_call_markup, strip_kimi_tool_call_markup
from llgraph.adapters.inbound.kimi_native import (
    normalize_kimi_native_tool_calls,
    strip_plain_functions_tool_calls,
)
from llgraph.context.message_normalize import format_agent_chat_display_text
from llgraph.core.agent_turn import route_after_agent

# 与 Web 截图一致（tool_calls_section / tool_call_begin_kimi 无 redacted_ 前缀）
_SCREENSHOT_SAMPLE = (
    "<|tool_calls_section_begin|>"
    "<|tool_call_begin_kimi|>functions.glob_files:1"
    "<|tool_call_argument_begin|>{\"path\": \"dataw\", \"pattern\": \"/OWNER**\"}"
    "<|tool_call_end_kimi|>"
    "<|tool_call_begin_kimi|>functions.glob_files:2"
    "<|tool_call_argument_begin|>{\"path\": \"dataw\", \"pattern\": \"/README**\"}"
    "<|tool_call_end_kimi|>"
    "<|tool_calls_section_end|>"
)


def test_parse_kimi_native_tool_calls_from_screenshot() -> None:
    calls, text = parse_kimi_native_tool_calls(_SCREENSHOT_SAMPLE)
    assert text == ""
    assert len(calls) == 2
    assert calls[0]["name"] == "glob_files"
    assert calls[0]["id"] == "functions.glob_files:1"
    assert calls[0]["args"] == {"path": "dataw", "pattern": "/OWNER**"}
    assert calls[1]["args"]["pattern"] == "/README**"


def test_repair_kimi_message_routes_to_tools() -> None:
    msg = AIMessage(content=_SCREENSHOT_SAMPLE)
    repaired, changed = normalize_kimi_native_tool_calls(msg)
    assert changed
    assert len(repaired.tool_calls) == 2
    assert repaired.content in ("", [], "")
    state = {
        "messages": [AIMessage(content="q"), repaired],
        "remaining_steps": 10,
    }
    assert route_after_agent(state) == "tools"


def test_parse_redacted_kimi_tokens() -> None:
    raw = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.glob_files:0"
        "<|tool_call_argument_begin|>{\"path\": \".\"}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls, text = parse_kimi_native_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "glob_files"
    assert text == ""


def test_strip_kimi_markup_keeps_normal_text() -> None:
    raw = "前缀说明" + _SCREENSHOT_SAMPLE
    assert strip_kimi_tool_call_markup(raw) == "前缀说明"
    calls, text = parse_kimi_native_tool_calls(raw)
    assert len(calls) == 2
    assert text == "前缀说明"


_PLAIN_LEAK_SAMPLE = (
    'functions.grep_files:1{"path": "demo-order-service", "pattern": "class Foo"}'
    "【规划】已定位到 RPC 接口定义与核心实现类。"
    'functions.read_files:1{"paths": ["a/b.java"]}'
    "【规划】需要查清 repository 是否过滤已取消记录。"
)


def test_strip_plain_functions_tool_calls() -> None:
    assert strip_plain_functions_tool_calls(_PLAIN_LEAK_SAMPLE) == (
        "【规划】已定位到 RPC 接口定义与核心实现类。"
        "【规划】需要查清 repository 是否过滤已取消记录。"
    )
    assert strip_inbound_tool_call_markup(_PLAIN_LEAK_SAMPLE) == (
        "【规划】已定位到 RPC 接口定义与核心实现类。"
        "【规划】需要查清 repository 是否过滤已取消记录。"
    )
    assert format_agent_chat_display_text(_PLAIN_LEAK_SAMPLE) == ""


def test_parse_plain_functions_tool_calls() -> None:
    calls, text = parse_kimi_native_tool_calls(_PLAIN_LEAK_SAMPLE)
    assert len(calls) == 2
    assert calls[0]["name"] == "grep_files"
    assert calls[0]["args"]["path"] == "demo-order-service"
    assert calls[1]["name"] == "read_files"
    assert "【规划】" in text
    assert "functions.grep_files" not in text
