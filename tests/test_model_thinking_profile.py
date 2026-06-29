"""各模型 thinking 规格与 dispatch 覆盖。"""

from __future__ import annotations

import pytest

from llgraph.context.message_dispatch_profile import (
    MessageDispatchProfile,
    dispatch_preserves_thinking_on_outbound,
    resolve_dispatch_profile,
)
from llgraph.core.model_thinking import set_runtime_thinking
from llgraph.core.model_thinking_profile import (
    ThinkingOutboundMode,
    apply_thinking_dispatch_overrides,
    resolve_model_thinking_spec,
)


@pytest.mark.parametrize(
    ("model_id", "mode", "supports"),
    [
        ("kimi-k2.6", ThinkingOutboundMode.REASONING_CONTENT, True),
        ("deepseek-v4-flash", ThinkingOutboundMode.REASONING_CONTENT, True),
        ("glm-5", ThinkingOutboundMode.REASONING_CONTENT, True),
        ("deepseek-chat", ThinkingOutboundMode.REASONING_CONTENT, True),
        ("claude-sonnet-4-6", ThinkingOutboundMode.NATIVE_BLOCKS, True),
        ("gpt-5.4", ThinkingOutboundMode.NONE, False),
    ],
)
def test_thinking_spec(model_id: str, mode: ThinkingOutboundMode, supports: bool) -> None:
    spec = resolve_model_thinking_spec(model_id)
    assert spec.supports_thinking is supports
    assert spec.outbound_mode == mode


def test_thinking_dispatch_override_kimi() -> None:
    base = MessageDispatchProfile(
        expand_parallel_tool_rounds=False,
        patch_tool_ai_reasoning=False,
        strip_assistant_thinking_blocks=True,
        label="base",
    )
    spec = resolve_model_thinking_spec("kimi-k2.6")
    out = apply_thinking_dispatch_overrides(base, spec)
    assert out.patch_tool_ai_reasoning is True
    assert out.strip_assistant_thinking_blocks is False
    assert out.expand_parallel_tool_rounds is False
    assert dispatch_preserves_thinking_on_outbound(out) is True


def test_thinking_dispatch_override_claude_native() -> None:
    base = MessageDispatchProfile(
        strip_assistant_thinking_blocks=True,
        label="claude-sonnet-4-6",
    )
    spec = resolve_model_thinking_spec("claude-sonnet-4-6")
    out = apply_thinking_dispatch_overrides(base, spec)
    assert out.patch_tool_ai_reasoning is False
    assert out.strip_assistant_thinking_blocks is False
    assert dispatch_preserves_thinking_on_outbound(out) is True


def test_deepseek_v4_thinking_enabled_dispatch() -> None:
    set_runtime_thinking(True)
    try:
        profile = resolve_dispatch_profile(None, "deepseek-v4-flash", thinking_enabled=True)
        assert profile.patch_tool_ai_reasoning is True
        assert profile.strip_assistant_thinking_blocks is False
    finally:
        set_runtime_thinking(None)


def test_deepseek_v4_thinking_disabled_preserves_thinking_blocks() -> None:
    profile = resolve_dispatch_profile(None, "deepseek-v4-flash", thinking_enabled=False)
    assert profile.strip_assistant_thinking_blocks is False
    assert profile.patch_tool_ai_reasoning is False
