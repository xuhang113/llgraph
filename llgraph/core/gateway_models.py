"""AI 网关可用模型列表（OpenAI 兼容 GET /v1/models + agent.json 目录）。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from llgraph.config.config import get_llgraph_settings
from llgraph.config.edit_settings import load_agent_config

_CACHE_TTL_SEC = 300.0
_cache_models: list[str] | None = None
_cache_at: float = 0.0
_cache_key: str = ""

DEFAULT_PROVIDER_LABEL = "AI 网关"


@dataclass(frozen=True)
class ModelCatalogSettings:
    """模型目录展示配置（agent.json → llm）。"""

    provider_label: str
    models_doc: str
    rate_label: str


@dataclass(frozen=True)
class ModelCatalogEntry:
    """支持模型目录项。"""

    model_id: str
    rate: float | None = None
    hint: str = ""
    context_window: int | None = None
    dispatch: dict[str, object] | None = None


def _load_llm_display_settings(workspace: Path | None) -> ModelCatalogSettings:
    """
    读取模型列表展示配置。

    @param workspace 工作区根
    @return ModelCatalogSettings
    """
    if workspace is None:
        return ModelCatalogSettings(
            provider_label=DEFAULT_PROVIDER_LABEL,
            models_doc="",
            rate_label="倍率",
        )
    cfg = load_agent_config(workspace)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}

    label_raw = llm_cfg.get("provider_label")
    provider_label = (
        label_raw.strip()
        if isinstance(label_raw, str) and label_raw.strip()
        else DEFAULT_PROVIDER_LABEL
    )

    doc_raw = llm_cfg.get("models_doc")
    models_doc = doc_raw.strip() if isinstance(doc_raw, str) else ""

    rate_raw = llm_cfg.get("rate_label")
    rate_label = (
        rate_raw.strip()
        if isinstance(rate_raw, str) and rate_raw.strip()
        else "倍率"
    )

    return ModelCatalogSettings(
        provider_label=provider_label,
        models_doc=models_doc,
        rate_label=rate_label,
    )


def _parse_catalog_item(item: object) -> ModelCatalogEntry | None:
    """
    解析 agent.json 中单条模型配置。

    @param item 字符串或 {id, rate, hint}
    @return ModelCatalogEntry 或 None
    """
    if isinstance(item, str) and item.strip():
        return ModelCatalogEntry(model_id=item.strip())
    if isinstance(item, dict):
        mid = item.get("id") or item.get("model") or item.get("name")
        if not isinstance(mid, str) or not mid.strip():
            return None
        rate_raw = item.get("rate")
        rate: float | None = None
        if rate_raw is not None:
            try:
                rate = float(rate_raw)
            except (TypeError, ValueError):
                rate = None
        hint = item.get("hint")
        hint_s = hint.strip() if isinstance(hint, str) else ""
        ctx_raw = item.get("context_window")
        if ctx_raw is None:
            ctx_raw = item.get("context_tokens")
        ctx_window = None
        if ctx_raw is not None:
            from llgraph.core.model_context_window import parse_context_window_value

            ctx_window = parse_context_window_value(ctx_raw)
        dispatch_raw = item.get("dispatch")
        dispatch: dict[str, object] | None = None
        if isinstance(dispatch_raw, dict):
            dispatch = dict(dispatch_raw)
        return ModelCatalogEntry(
            model_id=mid.strip(),
            rate=rate,
            hint=hint_s,
            context_window=ctx_window,
            dispatch=dispatch,
        )
    return None


def load_model_catalog(
    workspace: Path | None,
) -> tuple[list[ModelCatalogEntry], ModelCatalogSettings]:
    """
    从 agent.json llm 段加载支持模型目录与展示配置。

    @param workspace 工作区根
    @return (目录项列表, 展示配置)
    """
    settings = _load_llm_display_settings(workspace)
    if workspace is None:
        return [], settings
    cfg = load_agent_config(workspace)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    raw = llm_cfg.get("models")
    if not isinstance(raw, list):
        return [], settings
    out: list[ModelCatalogEntry] = []
    seen: set[str] = set()
    for item in raw:
        entry = _parse_catalog_item(item)
        if entry is None or entry.model_id in seen:
            continue
        seen.add(entry.model_id)
        out.append(entry)
    return out, settings


def catalog_model_ids(workspace: Path | None) -> list[str]:
    """
    目录中的模型 id 列表（保持配置顺序）。

    @param workspace 工作区根
    @return 模型 id
    """
    catalog, _ = load_model_catalog(workspace)
    return [e.model_id for e in catalog]


def fetch_gateway_models(*, force_refresh: bool = False) -> list[str]:
    """
    从网关拉取模型列表（OpenAI 兼容 /v1/models）。

    @param force_refresh 忽略缓存
    @return 模型 id 列表（可能为空）
    """
    global _cache_models, _cache_at, _cache_key

    settings = get_llgraph_settings()
    base = settings["base_url"].rstrip("/")
    api_key = settings["api_key"]
    key = f"{base}|{api_key[:8]}"
    now = time.monotonic()

    if (
        not force_refresh
        and _cache_models is not None
        and _cache_key == key
        and now - _cache_at < _CACHE_TTL_SEC
    ):
        return list(_cache_models)

    url = f"{base}/v1/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    models: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data", payload if isinstance(payload, list) else [])
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("name")
                    if isinstance(mid, str) and mid.strip():
                        models.append(mid.strip())
                elif isinstance(item, str) and item.strip():
                    models.append(item.strip())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, OSError):
        models = []

    seen: set[str] = set()
    unique: list[str] = []
    for mid in models:
        if mid not in seen:
            seen.add(mid)
            unique.append(mid)

    _cache_models = unique
    _cache_at = now
    _cache_key = key
    return list(unique)


def list_available_models(
    workspace: Path | None,
    *,
    force_refresh: bool = False,
) -> tuple[list[str], str]:
    """
    返回 /model 列举用的模型 id 列表。

    若 agent.json 配置了 llm.models（模型目录），**以目录顺序为准**；
    否则回退网关 /v1/models。

    @param workspace 工作区根
    @param force_refresh 强制刷新网关缓存
    @return (模型列表, 来源说明)
    """
    from llgraph.core.llm_settings import resolve_effective_model

    catalog, _ = load_model_catalog(workspace)
    current = resolve_effective_model(workspace)

    if catalog:
        ids = [e.model_id for e in catalog]
        if current and current not in ids:
            ids = [current] + ids
        return ids, "catalog"

    gateway = fetch_gateway_models(force_refresh=force_refresh)
    if gateway:
        if current and current not in gateway:
            return [current] + gateway, "gateway"
        return gateway, "gateway"

    if current:
        return [current], "env"
    return [], "empty"


def format_models_list(
    workspace: Path,
    *,
    current: str,
    force_refresh: bool = False,
) -> str:
    """
    格式化 /model 列表输出。

    @param workspace 工作区根
    @param current 当前模型
    @param force_refresh 是否刷新网关
    @return 多行文本
    """
    catalog, settings = load_model_catalog(workspace)
    gateway = fetch_gateway_models(force_refresh=force_refresh)
    gateway_set = set(gateway)

    lines = [
        f"支持模型（{settings.provider_label}）",
        f"当前: {current}",
    ]
    if settings.models_doc:
        lines.append(f"说明文档: {settings.models_doc}")
    lines.append("")

    if catalog:
        has_rate = any(e.rate is not None for e in catalog)
        if has_rate:
            lines.append(
                f"模型                          {settings.rate_label}  网关  备注"
            )
        else:
            lines.append("模型                                    网关  备注")
        lines.append("-" * 72)
        for entry in catalog:
            mark = " ← 当前" if entry.model_id == current else ""
            rate_s = f"{entry.rate:g}" if entry.rate is not None else "-"
            on_gw = "✓" if entry.model_id in gateway_set else "?"
            hint_s = f"  {entry.hint}" if entry.hint else ""
            if has_rate:
                lines.append(
                    f"  {entry.model_id:<28} {rate_s:>8}{'':8}{on_gw:>4}{mark}{hint_s}"
                )
            else:
                lines.append(f"  {entry.model_id:<28} {'':16}{on_gw:>4}{mark}{hint_s}")
        if current and current not in {e.model_id for e in catalog}:
            lines.append(f"  {current:<28}        ?   ← 当前（不在官方目录，仍可试用）")
        lines.extend(
            [
                "",
                "「?」表示本次未在网关 /v1/models 中见到，可能未开通或需 /model refresh",
                "切换: /model <模型名>  例: /model kimi-k2.6",
                "恢复默认: /model reset",
            ]
        )
        return "\n".join(lines)

    # 无目录配置：回退网关列表
    models, source = list_available_models(workspace, force_refresh=force_refresh)
    lines.append(f"来源: {source}（未配置 llm.models，/model refresh 拉网关）")
    lines.append("")
    if not models:
        lines.append(
            "（未获取到列表；请在 .llgraph/agent.json 配置 llm.models，"
            "或检查 LLGRAPH_API_*）"
        )
        return "\n".join(lines)

    for mid in models:
        mark = " ← 当前" if mid == current else ""
        lines.append(f"  {mid}{mark}")
    lines.extend(
        [
            "",
            "切换: /model <模型名>",
            "可在 ~/.llgraph/agent.json 配置 llm.provider_label / llm.models_doc / llm.models",
        ]
    )
    return "\n".join(lines)


def is_catalog_model(workspace: Path, model_id: str) -> bool:
    """
    模型是否在官方支持目录中。

    @param workspace 工作区根
    @param model_id 模型 id
    @return 无目录配置时恒为 True
    """
    catalog, _ = load_model_catalog(workspace)
    if not catalog:
        return True
    return model_id in {e.model_id for e in catalog}
