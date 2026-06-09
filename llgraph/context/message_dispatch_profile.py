"""按模型解析发往网关前的消息修链策略（落盘与出站分离）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config
from llgraph.core.gateway_models import ModelCatalogEntry, load_model_catalog


@dataclass(frozen=True)
class MessageDispatchProfile:
    """
    消息出站/清理策略。

    落盘与会话内存应使用 canonical_persist_profile()，避免按某一模型写死形态。
    """

    expand_parallel_tool_rounds: bool = False
    patch_tool_ai_reasoning: bool = False
    strip_assistant_thinking_blocks: bool = False
    label: str = "default"

    def summary(self) -> str:
        """
        简短描述，用于日志。

        @return 策略摘要
        """
        parts: list[str] = []
        if self.expand_parallel_tool_rounds:
            parts.append("展开并行工具")
        if self.patch_tool_ai_reasoning:
            parts.append("补 reasoning")
        if self.strip_assistant_thinking_blocks:
            parts.append("剥离 thinking 块")
        if not parts:
            parts.append("宽松 tool 链")
        return f"{self.label}: {'、'.join(parts)}"


def canonical_persist_profile() -> MessageDispatchProfile:
    """
    落盘/加载/内存状态用的 canonical 策略：只做无害清理，不按模型展开或补 reasoning。

    @return 持久化 profile
    """
    return MessageDispatchProfile(
        expand_parallel_tool_rounds=False,
        patch_tool_ai_reasoning=False,
        strip_assistant_thinking_blocks=False,
        label="persist",
    )


def _parse_dispatch_bool(raw: object, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return default


def _dispatch_from_mapping(
    raw: dict[str, object],
    *,
    label: str,
    fallback_expand: bool,
    fallback_reasoning: bool,
) -> MessageDispatchProfile:
    expand_raw = raw.get("expand_parallel_tools")
    if expand_raw is None:
        expand_raw = raw.get("expand_parallel_tool_rounds")
    reasoning_raw = raw.get("patch_reasoning_content")
    if reasoning_raw is None:
        reasoning_raw = raw.get("patch_tool_ai_reasoning")
    thinking_raw = raw.get("strip_assistant_thinking_blocks")
    if thinking_raw is None:
        thinking_raw = raw.get("strip_thinking_blocks")
    return MessageDispatchProfile(
        expand_parallel_tool_rounds=_parse_dispatch_bool(
            expand_raw,
            fallback_expand,
        ),
        patch_tool_ai_reasoning=_parse_dispatch_bool(
            reasoning_raw,
            fallback_reasoning,
        ),
        strip_assistant_thinking_blocks=_parse_dispatch_bool(
            thinking_raw,
            False,
        ),
        label=label,
    )


def _catalog_entry_profile(entry: ModelCatalogEntry) -> MessageDispatchProfile | None:
    dispatch = getattr(entry, "dispatch", None)
    if not isinstance(dispatch, dict):
        return None
    return _dispatch_from_mapping(
        dispatch,
        label=entry.model_id,
        fallback_expand=False,
        fallback_reasoning=False,
    )


def _global_dispatch_defaults(workspace: Path | None) -> dict[str, object] | None:
    if workspace is None:
        return None
    cfg = load_agent_config(workspace)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    raw = llm_cfg.get("message_dispatch_defaults")
    if isinstance(raw, dict):
        return raw
    raw = llm_cfg.get("message_dispatch")
    if isinstance(raw, dict):
        return raw
    return None


def _heuristic_dispatch_profile(model_id: str) -> MessageDispatchProfile:
    """
    按模型 id 推断出站策略（未在 agent.json 显式配置时）。

    @param model_id 模型 id
    @return 推断 profile
    """
    mid = model_id.strip().lower()
    if re.search(r"kimi-k2", mid, re.I):
        return MessageDispatchProfile(
            expand_parallel_tool_rounds=True,
            patch_tool_ai_reasoning=True,
            strip_assistant_thinking_blocks=False,
            label=f"{model_id}(heuristic:kimi-k2)",
        )
    if re.search(r"^(gpt|claude|o1|o3|o4)", mid, re.I):
        return MessageDispatchProfile(
            expand_parallel_tool_rounds=False,
            patch_tool_ai_reasoning=False,
            strip_assistant_thinking_blocks=True,
            label=f"{model_id}(heuristic:anthropic-openai)",
        )
    if re.search(r"^(glm|minimax|deepseek)", mid, re.I):
        return MessageDispatchProfile(
            expand_parallel_tool_rounds=True,
            patch_tool_ai_reasoning=False,
            strip_assistant_thinking_blocks=True,
            label=f"{model_id}(heuristic:strict-tool)",
        )
    if re.search(r"kimi", mid, re.I):
        return MessageDispatchProfile(
            expand_parallel_tool_rounds=True,
            patch_tool_ai_reasoning=False,
            strip_assistant_thinking_blocks=False,
            label=f"{model_id}(heuristic:kimi)",
        )
    return MessageDispatchProfile(
        expand_parallel_tool_rounds=False,
        patch_tool_ai_reasoning=False,
        strip_assistant_thinking_blocks=False,
        label=f"{model_id}(heuristic:default)",
    )


def resolve_dispatch_profile(
    workspace: Path | None,
    model_id: str | None = None,
) -> MessageDispatchProfile:
    """
    解析当前模型的出站修链策略。

    优先级：catalog 项 dispatch > llm.message_dispatch_defaults > 模型名启发式。

    @param workspace 工作区根
    @param model_id 模型 id；None 时用 resolve_effective_model
    @return MessageDispatchProfile
    """
    from llgraph.core.llm_settings import resolve_effective_model

    effective = (model_id or "").strip() or resolve_effective_model(workspace)
    catalog, _ = load_model_catalog(workspace)
    for entry in catalog:
        if entry.model_id == effective:
            from_catalog = _catalog_entry_profile(entry)
            if from_catalog is not None:
                return from_catalog
            break

    defaults = _global_dispatch_defaults(workspace)
    if defaults is not None:
        return _dispatch_from_mapping(
            defaults,
            label=f"{effective}(agent-defaults)",
            fallback_expand=False,
            fallback_reasoning=False,
        )

    return _heuristic_dispatch_profile(effective)
