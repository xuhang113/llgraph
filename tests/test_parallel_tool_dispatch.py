"""并行 tool 出站：Claude 式 1 assistant 多 tool_use + 1 user 多 tool_result。"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.message_dispatch_profile import resolve_dispatch_profile
from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch
from llgraph.core.gateway_kimi_patch import (
    is_kimi_thinking_model,
    missing_reasoning_on_formatted_tool_assistants,
    patch_gateway_kimi_reasoning_payload,
)
from llgraph.core.llm import create_gateway_llm
from llgraph.core.model_thinking import set_runtime_thinking
from llgraph.core.model_thinking_profile import (
    apply_thinking_dispatch_overrides,
    resolve_model_thinking_spec,
)
from llgraph.context.message_dispatch_profile import MessageDispatchProfile

WORKSPACE = Path(__file__).resolve().parents[1] / "examples" / "user-llgraph"

CATALOG_MODELS = [
    "kimi-k2.6",
    "kimi-k2.5",
    "claude-sonnet-4-6",
    "gpt-5.4",
    "deepseek-v4-flash",
    "glm-5",
]


def _parallel_tool_round() -> list:
    human = HumanMessage(content="并行搜索三处")
    ai = AIMessage(
        content="【规划】并行 grep",
        tool_calls=[
            {
                "name": "grep_files",
                "args": {"pattern": "foo"},
                "id": "call_p1",
                "type": "tool_call",
            },
            {
                "name": "grep_files",
                "args": {"pattern": "bar"},
                "id": "call_p2",
                "type": "tool_call",
            },
            {
                "name": "grep_files",
                "args": {"pattern": "baz"},
                "id": "call_p3",
                "type": "tool_call",
            },
        ],
        additional_kwargs={"llgraph": {"thinking_text": "并行 tool 思考"}},
    )
    return [
        human,
        ai,
        ToolMessage(content="r1", tool_call_id="call_p1", name="grep_files"),
        ToolMessage(content="r2", tool_call_id="call_p2", name="grep_files"),
        ToolMessage(content="r3", tool_call_id="call_p3", name="grep_files"),
    ]


def _count_blocks(block: dict, block_type: str) -> int:
    content = block.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for item in content
        if isinstance(item, dict) and item.get("type") == block_type
    )


def _parallel_tool_shapes(formatted: list[dict]) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for idx, block in enumerate(formatted):
        if block.get("role") != "assistant":
            continue
        tool_use_n = _count_blocks(block, "tool_use")
        if tool_use_n < 1:
            continue
        if idx + 1 >= len(formatted):
            continue
        nxt = formatted[idx + 1]
        if nxt.get("role") != "user":
            continue
        tool_result_n = _count_blocks(nxt, "tool_result")
        shapes.append((tool_use_n, tool_result_n))
    return shapes


def _http_payload_for_model(model_id: str, messages: list) -> list[dict]:
    patch_gateway_kimi_reasoning_payload()
    prepared = prepare_messages_for_llm_dispatch(
        messages,
        agent_system_content="sys",
        workspace=WORKSPACE,
        model_id=model_id,
    )
    llm = create_gateway_llm(WORKSPACE)
    payload = llm._get_request_payload(prepared)
    formatted = payload.get("messages")
    assert isinstance(formatted, list)
    return formatted


@pytest.fixture(autouse=True)
def _reset_runtime_thinking() -> None:
    set_runtime_thinking(None)
    yield
    set_runtime_thinking(None)


@pytest.mark.parametrize("model_id", CATALOG_MODELS)
def test_parallel_tools_bundled_in_http_payload(model_id: str) -> None:
    formatted = _http_payload_for_model(model_id, _parallel_tool_round())
    shapes = _parallel_tool_shapes(formatted)
    assert (3, 3) in shapes, f"{model_id} HTTP shapes={shapes}, expected bundled (3,3)"
    serial_triple = shapes.count((1, 1)) >= 3
    assert not serial_triple, f"{model_id} still serial (1,1)x3: {shapes}"


@pytest.mark.parametrize("model_id", ["kimi-k2.6", "deepseek-v4-flash", "glm-5"])
def test_thinking_override_does_not_expand_parallel(model_id: str) -> None:
    set_runtime_thinking(True)
    profile = resolve_dispatch_profile(WORKSPACE, model_id, thinking_enabled=True)
    assert profile.expand_parallel_tool_rounds is False


def test_apply_thinking_dispatch_override_keeps_expand_false() -> None:
    base = MessageDispatchProfile(
        expand_parallel_tool_rounds=False,
        patch_tool_ai_reasoning=False,
        strip_assistant_thinking_blocks=True,
        label="base",
    )
    for model_id in ("kimi-k2.6", "deepseek-v4-flash", "glm-5"):
        spec = resolve_model_thinking_spec(model_id)
        out = apply_thinking_dispatch_overrides(base, spec)
        assert out.expand_parallel_tool_rounds is False, model_id


@pytest.mark.parametrize("model_id", ["kimi-k2.6", "deepseek-v4-flash", "glm-5"])
def test_kimi_style_models_reasoning_on_bundled_tool_assistant(model_id: str) -> None:
    formatted = _http_payload_for_model(model_id, _parallel_tool_round())
    missing = missing_reasoning_on_formatted_tool_assistants(
        formatted,
        require_thinking_block=is_kimi_thinking_model(model_id),
    )
    assert missing == [], f"{model_id} missing reasoning at {missing}"


@pytest.mark.parametrize("model_id", CATALOG_MODELS)
def test_lc_layer_keeps_single_ai_for_parallel_tools(model_id: str) -> None:
    from llgraph.context.chat_history_repair import (
        ai_message_has_tool_calls,
        rebuild_provider_safe_messages,
    )

    profile = resolve_dispatch_profile(WORKSPACE, model_id)
    dispatched, _ = rebuild_provider_safe_messages(_parallel_tool_round(), profile)
    tool_ais = [
        msg
        for msg in dispatched
        if isinstance(msg, AIMessage) and ai_message_has_tool_calls(msg)
    ]
    assert len(tool_ais) == 1
    assert len(tool_ais[0].tool_calls or []) == 3
