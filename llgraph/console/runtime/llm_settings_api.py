"""Web LLM 设置 API 载荷。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.core.gateway_models import load_model_catalog, list_available_models
from llgraph.core.llm_settings import get_runtime_model, resolve_effective_model, set_runtime_model
from llgraph.core.model_thinking import (
    get_runtime_thinking,
    is_thinking_enabled,
    model_supports_thinking,
    set_runtime_thinking,
)


def build_llm_settings_payload(workspace: Path) -> dict[str, Any]:
    """
    当前 LLM 配置与可选模型列表。

    @param workspace 工作区根
    @return Web JSON
    """
    current = resolve_effective_model(workspace)
    catalog, display = load_model_catalog(workspace)
    model_ids, models_source = list_available_models(workspace)
    catalog_by_id = {e.model_id: e for e in catalog}

    models: list[dict[str, Any]] = []
    for mid in model_ids:
        entry = catalog_by_id.get(mid)
        models.append(
            {
                "id": mid,
                "hint": entry.hint if entry else "",
                "rate": entry.rate if entry else None,
                "supports_thinking": model_supports_thinking(workspace, mid),
                "current": mid == current,
            }
        )

    supports = model_supports_thinking(workspace, current)
    runtime_model = get_runtime_model()
    runtime_thinking = get_runtime_thinking()

    return {
        "model": current,
        "model_runtime_override": runtime_model,
        "models": models,
        "models_source": models_source,
        "provider_label": display.provider_label,
        "thinking": {
            "supported": supports,
            "enabled": is_thinking_enabled(workspace, current) if supports else False,
            "runtime_override": runtime_thinking,
        },
    }


def apply_llm_settings(
    workspace: Path,
    *,
    model: str | None = None,
    thinking_enabled: bool | None = None,
    reset_model: bool = False,
    reset_thinking: bool = False,
) -> dict[str, Any]:
    """
    应用 Web 侧模型 / thinking 设置。

    @param workspace 工作区根
    @param model 切换模型 id
    @param thinking_enabled 运行时 thinking 开关
    @param reset_model 恢复默认模型
    @param reset_thinking 清除 thinking 运行时覆盖
    @return 更新后的 settings payload
    """
    if reset_model:
        set_runtime_model(None)
    elif model is not None and model.strip():
        set_runtime_model(model.strip())

    if reset_thinking:
        set_runtime_thinking(None)
    elif thinking_enabled is not None:
        set_runtime_thinking(thinking_enabled)

    return build_llm_settings_payload(workspace)
