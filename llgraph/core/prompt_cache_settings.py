"""Prompt Cache 配置（.llgraph/agent.json → context.prompt_cache）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config


@dataclass(frozen=True)
class PromptCacheSettings:
    """Anthropic 风格 prompt caching 参数。"""

    enabled: bool
    ttl: str
    tag_tools: bool
    tag_conversation_tail: bool
    min_messages: int
    model_allowlist: frozenset[str] | None


def resolve_prompt_cache_settings(workspace: Path | None) -> PromptCacheSettings:
    """
    解析 prompt_cache 配置。

    @param workspace 工作区根
    @return PromptCacheSettings
    """
    if workspace is None:
        return PromptCacheSettings(
            enabled=False,
            ttl="5m",
            tag_tools=True,
            tag_conversation_tail=True,
            min_messages=0,
            model_allowlist=None,
        )

    cfg = load_agent_config(workspace)
    ctx = cfg.get("context") if isinstance(cfg.get("context"), dict) else {}
    raw = ctx.get("prompt_cache")
    if not isinstance(raw, dict):
        raw = {}

    enabled = raw.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in ("0", "false", "no", "off")

    ttl = str(raw.get("ttl", "5m")).strip().lower()
    if ttl not in ("5m", "1h"):
        ttl = "5m"

    tag_tools = raw.get("tag_tools", True)
    if isinstance(tag_tools, str):
        tag_tools = tag_tools.strip().lower() not in ("0", "false", "no")

    tag_tail = raw.get("tag_conversation_tail", True)
    if isinstance(tag_tail, str):
        tag_tail = tag_tail.strip().lower() not in ("0", "false", "no")

    min_msgs = raw.get("min_messages", 0)
    try:
        min_msgs = max(0, int(min_msgs))
    except (TypeError, ValueError):
        min_msgs = 0

    allowlist: frozenset[str] | None = None
    models_raw = raw.get("models")
    if isinstance(models_raw, list) and models_raw:
        allowlist = frozenset(str(m).strip().lower() for m in models_raw if str(m).strip())

    return PromptCacheSettings(
        enabled=bool(enabled),
        ttl=ttl,
        tag_tools=bool(tag_tools),
        tag_conversation_tail=bool(tag_tail),
        min_messages=min_msgs,
        model_allowlist=allowlist,
    )


def prompt_cache_enabled_for_model(
    workspace: Path | None,
    model_id: str | None,
) -> bool:
    """
    当前模型是否启用 prompt cache。

    @param workspace 工作区根
    @param model_id 模型 id
    @return 是否启用
    """
    settings = resolve_prompt_cache_settings(workspace)
    if not settings.enabled:
        return False
    if not model_id or not str(model_id).strip():
        return True
    if settings.model_allowlist is None:
        return True
    mid = str(model_id).strip().lower()
    if mid in settings.model_allowlist:
        return True
    for prefix in settings.model_allowlist:
        if mid.startswith(prefix) or prefix in mid:
            return True
    return False
