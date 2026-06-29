"""thinking 出站：按模型 dispatch profile 保留/剥离/回灌。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch
from llgraph.core.agent_turn import THINK_CONTINUE_NUDGE, think_continue_nudge_pending
from llgraph.core.gateway_kimi_patch import inject_reasoning_into_formatted_messages
from llgraph.core.model_thinking import set_runtime_thinking
from llgraph.core.model_thinking_profile import dispatch_preserves_thinking_on_outbound
from llgraph.context.message_dispatch_profile import resolve_dispatch_profile


def _thinking_only_messages() -> list:
    return [
        HumanMessage(content="goal"),
        AIMessage(content=[{"type": "thinking", "thinking": "我在规划 plan JSON..."}]),
    ]


def _thinking_plus_text_messages() -> list:
    return [
        HumanMessage(content="hi"),
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "内部推理"},
                {"type": "text", "text": "hello"},
            ],
        ),
    ]


@pytest.fixture(autouse=True)
def _reset_runtime_thinking() -> None:
    set_runtime_thinking(None)
    yield
    set_runtime_thinking(None)


@pytest.mark.parametrize(
    ("model_id", "thinking_on", "preserve", "patch_reasoning", "strip"),
    [
        ("kimi-k2.6", True, True, True, False),
        ("kimi-k2.6", False, True, True, False),
        ("deepseek-v4-flash", True, True, True, False),
        ("deepseek-v4-flash", False, False, False, False),
        ("claude-sonnet-4-6", True, True, False, False),
        ("claude-sonnet-4-6", False, False, False, True),
        ("gpt-5.4", False, False, False, True),
        ("deepseek-chat", True, True, True, False),
        ("glm-5", True, True, True, False),
    ],
)
def test_dispatch_profile_thinking_matrix(
    model_id: str,
    thinking_on: bool,
    preserve: bool,
    patch_reasoning: bool,
    strip: bool,
) -> None:
    profile = resolve_dispatch_profile(None, model_id, thinking_enabled=thinking_on)
    assert profile.patch_tool_ai_reasoning is patch_reasoning
    assert profile.strip_assistant_thinking_blocks is strip
    assert dispatch_preserves_thinking_on_outbound(profile) is preserve


@pytest.mark.parametrize(
    ("model_id", "thinking_on", "expect_reasoning", "expect_thinking_text"),
    [
        ("kimi-k2.6", True, True, True),
        ("deepseek-v4-flash", True, True, True),
        ("deepseek-v4-flash", False, False, True),
        ("claude-sonnet-4-6", True, False, True),
        ("claude-sonnet-4-6", False, False, True),
        ("gpt-5.4", False, False, True),
        ("glm-5", True, True, True),
    ],
)
def test_thinking_only_persisted_by_model(
    model_id: str,
    thinking_on: bool,
    expect_reasoning: bool,
    expect_thinking_text: bool,
) -> None:
    if thinking_on:
        set_runtime_thinking(True)
    prepared = prepare_messages_for_llm_dispatch(
        _thinking_only_messages(),
        agent_system_content="sys",
        workspace=None,
        model_id=model_id,
    )
    ai = prepared[-1]
    assert isinstance(ai, AIMessage)
    meta = (ai.additional_kwargs or {}).get("llgraph") or {}
    has_meta = "我在规划" in str(meta.get("thinking_text") or "")
    assert has_meta is expect_thinking_text
    has_reasoning = bool((ai.additional_kwargs or {}).get("reasoning_content"))
    assert has_reasoning is expect_reasoning


def test_claude_thinking_on_rehydrates_native_block() -> None:
    set_runtime_thinking(True)
    prepared = prepare_messages_for_llm_dispatch(
        _thinking_only_messages(),
        agent_system_content="sys",
        workspace=None,
        model_id="claude-sonnet-4-6",
    )
    ai = prepared[-1]
    content = ai.content
    assert isinstance(content, list)
    assert any(
        isinstance(b, dict) and b.get("type") == "thinking"
        for b in content
    )


@pytest.mark.parametrize(
    ("model_id", "thinking_on", "expect_reasoning"),
    [
        ("kimi-k2.6", True, True),
        ("deepseek-v4-flash", True, True),
        ("claude-sonnet-4-6", True, False),
        ("gpt-5.4", False, False),
    ],
)
def test_thinking_plus_text_reasoning_by_model(
    model_id: str,
    thinking_on: bool,
    expect_reasoning: bool,
) -> None:
    if thinking_on:
        set_runtime_thinking(True)
    prepared = prepare_messages_for_llm_dispatch(
        _thinking_plus_text_messages(),
        agent_system_content="sys",
        workspace=None,
        model_id=model_id,
    )
    ai = prepared[-1]
    if isinstance(ai.content, str):
        assert ai.content == "hello"
    else:
        texts = [
            str(b.get("text", ""))
            for b in ai.content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert "hello" in "".join(texts)
    has_reasoning = bool((ai.additional_kwargs or {}).get("reasoning_content"))
    assert has_reasoning is expect_reasoning


def test_kimi_inject_thinking_only_assistant() -> None:
    ai = AIMessage(
        content=" ",
        additional_kwargs={
            "llgraph": {"thinking_text": "内部推理链"},
            "reasoning_content": "内部推理链",
        },
    )
    source = [HumanMessage(content="hi"), ai]
    formatted = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]
    injected = inject_reasoning_into_formatted_messages(source, formatted)
    assert injected == 1
    assert formatted[1].get("reasoning_content") == "内部推理链"
    content = formatted[1].get("content")
    assert isinstance(content, list)
    assert any(b.get("type") == "thinking" for b in content if isinstance(b, dict))


def test_should_inject_thinking_only_reasoning_kimi_only() -> None:
    from llgraph.core.gateway_kimi_patch import should_inject_thinking_only_reasoning

    assert should_inject_thinking_only_reasoning(None, "kimi-k2.6") is True
    assert should_inject_thinking_only_reasoning(None, "deepseek-v4-flash") is True
    assert should_inject_thinking_only_reasoning(None, "claude-sonnet-4-6") is False


def test_think_nudge_not_duplicated() -> None:
    ai = AIMessage(content=[{"type": "thinking", "thinking": "x"}])
    messages = [HumanMessage(content="hi"), ai, HumanMessage(content=THINK_CONTINUE_NUDGE)]
    assert think_continue_nudge_pending(messages)
