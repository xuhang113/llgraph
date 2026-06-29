"""各模型 thinking 能力、请求 payload 与出站 dispatch 策略。"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llgraph.context.message_dispatch_profile import MessageDispatchProfile


class ThinkingOutboundMode(str, Enum):
    """thinking 出站形态。"""

    NONE = "none"
    REASONING_CONTENT = "reasoning_content"
    NATIVE_BLOCKS = "native_blocks"


@dataclass(frozen=True)
class ModelThinkingSpec:
    """单模型 thinking 规格。"""

    supports_thinking: bool
    default_payload: dict[str, Any] | None
    outbound_mode: ThinkingOutboundMode
    expand_parallel_tools: bool | None = None


def resolve_model_thinking_spec(model_id: str | None) -> ModelThinkingSpec:
    """
    按模型 id 解析 thinking 能力与默认 payload。

    @param model_id 模型 id
    @return thinking 规格
    """
    if not model_id or not str(model_id).strip():
        return ModelThinkingSpec(False, None, ThinkingOutboundMode.NONE)
    mid = str(model_id).strip().lower()

    if re.search(r"kimi-k2", mid):
        return ModelThinkingSpec(
            True,
            {"type": "enabled", "keep": "all"},
            ThinkingOutboundMode.REASONING_CONTENT,
            expand_parallel_tools=False,
        )
    if re.search(r"deepseek-v4", mid):
        return ModelThinkingSpec(
            True,
            {"type": "enabled"},
            ThinkingOutboundMode.REASONING_CONTENT,
            expand_parallel_tools=False,
        )
    if re.search(r"glm-5", mid):
        return ModelThinkingSpec(
            True,
            {"type": "enabled"},
            ThinkingOutboundMode.REASONING_CONTENT,
            expand_parallel_tools=False,
        )
    if re.search(r"^(glm|minimax|deepseek)", mid):
        return ModelThinkingSpec(
            True,
            {"type": "enabled"},
            ThinkingOutboundMode.REASONING_CONTENT,
            expand_parallel_tools=False,
        )
    if re.search(r"claude-(opus|sonnet|haiku)-4", mid):
        return ModelThinkingSpec(
            True,
            {"type": "adaptive"},
            ThinkingOutboundMode.NATIVE_BLOCKS,
            expand_parallel_tools=False,
        )
    if re.search(r"kimi", mid):
        return ModelThinkingSpec(
            True,
            {"type": "enabled", "keep": "all"},
            ThinkingOutboundMode.REASONING_CONTENT,
            expand_parallel_tools=False,
        )
    return ModelThinkingSpec(False, None, ThinkingOutboundMode.NONE)


def apply_thinking_dispatch_overrides(
    profile: MessageDispatchProfile,
    spec: ModelThinkingSpec,
) -> MessageDispatchProfile:
    """
    thinking 开启时覆盖出站 profile（不 strip、按协议回灌）。

    @param profile 基础 profile
    @param spec 模型 thinking 规格
    @return 覆盖后的 profile
    """
    from dataclasses import replace
    if not spec.supports_thinking or spec.outbound_mode == ThinkingOutboundMode.NONE:
        return profile

    expand = (
        spec.expand_parallel_tools
        if spec.expand_parallel_tools is not None
        else profile.expand_parallel_tool_rounds
    )

    if spec.outbound_mode == ThinkingOutboundMode.REASONING_CONTENT:
        return replace(
            profile,
            expand_parallel_tool_rounds=expand,
            patch_tool_ai_reasoning=True,
            strip_assistant_thinking_blocks=False,
            label=f"{profile.label}+thinking",
        )

    return replace(
        profile,
        expand_parallel_tool_rounds=expand,
        patch_tool_ai_reasoning=False,
        strip_assistant_thinking_blocks=False,
        label=f"{profile.label}+thinking-native",
    )


def dispatch_preserves_thinking_on_outbound(profile: MessageDispatchProfile) -> bool:
    """
    出站是否保留/回灌 thinking（含 native blocks 与 reasoning_content）。

    @param profile 出站 profile
    @return 是否保留
    """
    if profile.strip_assistant_thinking_blocks:
        return False
    return profile.patch_tool_ai_reasoning or "+thinking-native" in profile.label


def model_uses_reasoning_content_injection(model_id: str | None) -> bool:
    """
    HTTP 层是否需 content.thinking + reasoning_content 注入。

    @param model_id 模型 id
    @return 是否注入
    """
    spec = resolve_model_thinking_spec(model_id)
    return spec.outbound_mode == ThinkingOutboundMode.REASONING_CONTENT


def dispatch_rehydrates_native_thinking_blocks(profile: MessageDispatchProfile) -> bool:
    """
    出站是否将 llgraph.thinking_text 还原为 content 内 thinking 块（Claude 等）。

    @param profile 出站 profile
    @return 是否 rehydrate
    """
    return (
        dispatch_preserves_thinking_on_outbound(profile)
        and not profile.patch_tool_ai_reasoning
    )


def model_uses_native_thinking_blocks(model_id: str | None) -> bool:
    """
    模型规格是否使用 content 内 native thinking 块。

    @param model_id 模型 id
    @return 是否 native blocks
    """
    spec = resolve_model_thinking_spec(model_id)
    return spec.outbound_mode == ThinkingOutboundMode.NATIVE_BLOCKS
