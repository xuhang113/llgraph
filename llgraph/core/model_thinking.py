"""按 agent.json 与模型族解析网关 thinking 请求参数。"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from pathlib import Path

from llgraph.core.gateway_models import load_model_catalog


def _heuristic_thinking_payload(model_id: str) -> dict[str, Any] | None:
    """
    未显式配置时的模型族默认 thinking。

    仅 Kimi k 系列默认开启；其它模型默认关闭（省 token、减少 tail 膨胀）。

    @param model_id 模型 id
    @return thinking 请求体；None 表示不发送
    """
    mid = model_id.strip().lower()
    if "kimi" in mid or re.search(r"k2\.[56]", mid):
        return {"type": "enabled", "keep": "all"}
    return None


def _coerce_thinking_dict(raw: dict[str, Any]) -> dict[str, Any] | None:
    """
    解析 thinking 配置对象。

    @param raw agent.json 中的 thinking 段
    @return 网关 payload.thinking；None 表示关闭
    """
    if raw.get("enabled") is False:
        return None
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "enabled":
            continue
        out[key] = value
    if not out and raw.get("enabled") is True:
        return {}
    return out if out else {}


def _parse_thinking_raw(raw: object) -> dict[str, Any] | None | object:
    """
    解析单条 thinking 配置。

    @param raw false/对象/省略
    @return dict=启用；None=关闭；_USE_DEFAULT=走默认/heuristic
    """
    if raw is None:
        return _USE_DEFAULT
    if raw is False:
        return None
    if raw is True:
        return {}
    if isinstance(raw, dict):
        return _coerce_thinking_dict(raw)
    return _USE_DEFAULT


_USE_DEFAULT = object()

_runtime_thinking: bool | None = None


def set_runtime_thinking(enabled: bool | None) -> None:
    """
    会话内覆盖 thinking 开关（Web / 终端运行时）。

    @param enabled True=强制开；False=强制关；None=恢复 agent.json/启发式
    """
    global _runtime_thinking
    _runtime_thinking = enabled


def get_runtime_thinking() -> bool | None:
    """当前 thinking 运行时覆盖。"""
    return _runtime_thinking


def model_supports_thinking(workspace: Path | None, model_id: str | None) -> bool:
    """
    模型是否支持（可配置）thinking。

    @param workspace 工作区根
    @param model_id 模型 id
    @return 是否展示 thinking 开关
    """
    if not model_id or not str(model_id).strip():
        return False
    effective = str(model_id).strip()
    catalog, _ = load_model_catalog(workspace)
    for entry in catalog:
        if entry.model_id == effective:
            parsed = _parse_thinking_raw(getattr(entry, "thinking", None))
            if parsed is None:
                return False
            if parsed is not _USE_DEFAULT:
                return True
            break
    return _heuristic_thinking_payload(effective) is not None


def is_thinking_enabled(workspace: Path | None, model_id: str | None) -> bool:
    """
    当前模型 thinking 是否实际启用（含运行时覆盖）。

    @param workspace 工作区根
    @param model_id 模型 id
    @return 是否发往网关 thinking 参数
    """
    return resolve_model_thinking_payload(workspace, model_id) is not None


def _resolve_config_thinking_payload(
    workspace: Path | None,
    model_id: str | None,
) -> dict[str, Any] | None:
    """
    按 agent.json 与模型族解析 thinking（不含运行时覆盖）。

    @param workspace 工作区根
    @param model_id 模型 id
    @return thinking 字典；None 表示不发送
    """
    if not model_id or not str(model_id).strip():
        return None
    effective = str(model_id).strip()

    catalog, _ = load_model_catalog(workspace)
    entry_thinking: object = _USE_DEFAULT
    for entry in catalog:
        if entry.model_id == effective:
            entry_thinking = _parse_thinking_raw(getattr(entry, "thinking", None))
            break

    if entry_thinking is None:
        return None
    if entry_thinking is not _USE_DEFAULT and isinstance(entry_thinking, dict):
        base = deepcopy(entry_thinking)
        heuristic = _heuristic_thinking_payload(effective)
        if not base:
            return heuristic
        if heuristic is None:
            return base
        merged = {**heuristic, **base}
        return merged

    defaults_raw: object = None
    if workspace is not None:
        from llgraph.config.edit_settings import load_agent_config

        cfg = load_agent_config(workspace)
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        defaults_raw = llm_cfg.get("thinking_defaults")

    parsed_defaults = _parse_thinking_raw(defaults_raw)
    if parsed_defaults is None:
        return None
    if isinstance(parsed_defaults, dict):
        heuristic = _heuristic_thinking_payload(effective)
        if not parsed_defaults:
            return heuristic
        if heuristic is None:
            return parsed_defaults if parsed_defaults else None
        return {**heuristic, **parsed_defaults}

    return _heuristic_thinking_payload(effective)


def resolve_model_thinking_payload(
    workspace: Path | None,
    model_id: str | None,
) -> dict[str, Any] | None:
    """
    解析当前模型发往网关的 thinking 参数。

    优先级：运行时覆盖 > catalog 项 thinking > llm.thinking_defaults > 模型族 heuristic。

    @param workspace 工作区根
    @param model_id 模型 id
    @return thinking 字典；None 表示不发送
    """
    if _runtime_thinking is False:
        return None
    base = _resolve_config_thinking_payload(workspace, model_id)
    if _runtime_thinking is True:
        if base is not None:
            return base
        if not model_id or not str(model_id).strip():
            return None
        heuristic = _heuristic_thinking_payload(str(model_id).strip())
        return heuristic if heuristic is not None else {"type": "enabled"}
    return base


def merge_payload_thinking(
    existing: object,
    desired: dict[str, Any],
    *,
    model_id: str | None,
) -> dict[str, Any]:
    """
    合并已有 payload.thinking 与期望配置（Kimi 保留 keep=all）。

    @param existing 当前 payload.thinking
    @param desired resolve_model_thinking_payload 结果
    @param model_id 模型 id
    @return 合并后的 thinking
    """
    if not isinstance(existing, dict):
        return dict(desired)
    merged = {**desired, **existing}
    mid = (model_id or "").lower()
    if ("kimi" in mid or "k2.5" in mid or "k2.6" in mid) and desired.get("keep"):
        merged["keep"] = desired["keep"]
    return merged
