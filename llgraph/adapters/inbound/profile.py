"""入站 AIMessage 归一化策略（与 MessageDispatchProfile 对称）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config
from llgraph.core.gateway_models import ModelCatalogEntry, load_model_catalog


@dataclass(frozen=True)
class InboundAdapterProfile:
    """
    模型响应归一化策略。

    @param parse_kimi_native_tool_tokens 是否解析 content 内 Kimi K2 原生 tool token
    @param repair_streaming_tool_calls 是否修复 invalid_tool_calls 流式碎片
    @param label 日志标签
    """

    parse_kimi_native_tool_tokens: bool = False
    repair_streaming_tool_calls: bool = True
    label: str = "default"

    def summary(self) -> str:
        parts: list[str] = []
        if self.parse_kimi_native_tool_tokens:
            parts.append("Kimi token→tool_calls")
        if self.repair_streaming_tool_calls:
            parts.append("流式 invalid 修复")
        if not parts:
            parts.append("透传")
        return f"{self.label}: {'、'.join(parts)}"


def _parse_inbound_bool(raw: object, default: bool) -> bool:
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


def _heuristic_inbound_profile(model_id: str) -> InboundAdapterProfile:
    """
    按模型 id 推断入站策略（未在 agent.json 显式配置时）。

    @param model_id 模型 id
    @return InboundAdapterProfile
    """
    mid = model_id.strip().lower()
    if re.search(r"kimi-k2", mid, re.I) or re.search(r"k2\.[56]", mid):
        return InboundAdapterProfile(
            parse_kimi_native_tool_tokens=True,
            repair_streaming_tool_calls=True,
            label=f"{model_id}(heuristic:kimi-k2)",
        )
    if re.search(r"kimi", mid, re.I):
        return InboundAdapterProfile(
            parse_kimi_native_tool_tokens=True,
            repair_streaming_tool_calls=True,
            label=f"{model_id}(heuristic:kimi)",
        )
    return InboundAdapterProfile(
        parse_kimi_native_tool_tokens=False,
        repair_streaming_tool_calls=True,
        label=f"{model_id}(heuristic:default)",
    )


def _catalog_entry_inbound_profile(entry: ModelCatalogEntry) -> InboundAdapterProfile | None:
    dispatch = entry.dispatch
    if not isinstance(dispatch, dict):
        return None
    raw = dispatch.get("inbound")
    if not isinstance(raw, dict):
        return None
    kimi_raw = raw.get("parse_kimi_native_tool_tokens")
    if kimi_raw is None:
        kimi_raw = raw.get("parse_kimi_native")
    stream_raw = raw.get("repair_streaming_tool_calls")
    return InboundAdapterProfile(
        parse_kimi_native_tool_tokens=_parse_inbound_bool(kimi_raw, False),
        repair_streaming_tool_calls=_parse_inbound_bool(stream_raw, True),
        label=f"{entry.model_id}(catalog)",
    )


def _defaults_from_agent_config(workspace: Path | None) -> dict[str, object] | None:
    if workspace is None:
        return None
    cfg = load_agent_config(workspace)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    raw = llm_cfg.get("inbound_defaults")
    if isinstance(raw, dict):
        return raw
    return None


def resolve_inbound_profile(
    workspace: Path | None,
    model_id: str | None = None,
) -> InboundAdapterProfile:
    """
    解析当前模型的入站归一化策略。

    优先级：catalog.dispatch.inbound > llm.inbound_defaults > 模型名启发式。

    @param workspace 工作区根
    @param model_id 模型 id；None 时用 resolve_effective_model
    @return InboundAdapterProfile
    """
    from llgraph.core.llm_settings import resolve_effective_model

    effective = (model_id or "").strip() or resolve_effective_model(workspace)
    catalog, _ = load_model_catalog(workspace)
    for entry in catalog:
        if entry.model_id == effective:
            from_catalog = _catalog_entry_inbound_profile(entry)
            if from_catalog is not None:
                return from_catalog
            break

    defaults = _defaults_from_agent_config(workspace)
    if isinstance(defaults, dict):
        kimi_raw = defaults.get("parse_kimi_native_tool_tokens")
        if kimi_raw is None:
            kimi_raw = defaults.get("parse_kimi_native")
        stream_raw = defaults.get("repair_streaming_tool_calls")
        if kimi_raw is not None or stream_raw is not None:
            heuristic = _heuristic_inbound_profile(effective)
            return InboundAdapterProfile(
                parse_kimi_native_tool_tokens=_parse_inbound_bool(
                    kimi_raw,
                    heuristic.parse_kimi_native_tool_tokens,
                ),
                repair_streaming_tool_calls=_parse_inbound_bool(
                    stream_raw,
                    heuristic.repair_streaming_tool_calls,
                ),
                label=f"{effective}(defaults)",
            )

    return _heuristic_inbound_profile(effective)
