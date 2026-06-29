"""入站 adapter profile 与 normalize 单测。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from llgraph.adapters.inbound import UnstructuredToolCallError, normalize_ai_response, resolve_inbound_profile
from llgraph.adapters.inbound.profile import InboundAdapterProfile
from llgraph.adapters.inbound.kimi_native import normalize_kimi_native_tool_calls


def test_kimi_profile_still_resolves_native_flag() -> None:
    profile = resolve_inbound_profile(None, "kimi-k2.6")
    assert profile.parse_kimi_native_tool_tokens is True


def test_claude_profile_skips_kimi_parse_flag() -> None:
    profile = resolve_inbound_profile(None, "claude-sonnet-4-6")
    assert profile.parse_kimi_native_tool_tokens is False
    assert profile.repair_streaming_tool_calls is True


def test_normalize_rejects_kimi_token_markup() -> None:
    sample = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin_kimi|>functions.grep_files:0"
        "<|tool_call_argument_begin|>{\"pattern\": \"x\"}"
        "<|tool_call_end_kimi|>"
        "<|tool_calls_section_end|>"
    )
    msg = AIMessage(content=sample)
    with pytest.raises(UnstructuredToolCallError):
        normalize_ai_response(
            msg,
            None,
            "kimi-k2.6",
            profile=InboundAdapterProfile(parse_kimi_native_tool_tokens=True),
        )


def test_kimi_native_module_still_parses_for_unit_tests() -> None:
    sample = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin_kimi|>functions.list_directory:0"
        "<|tool_call_argument_begin|>{\"path\": \".\"}"
        "<|tool_call_end_kimi|>"
        "<|tool_calls_section_end|>"
    )
    msg = AIMessage(content=sample)
    repaired, changed = normalize_kimi_native_tool_calls(msg)
    assert changed
    assert repaired.tool_calls[0]["name"] == "list_directory"
