"""会话编辑行为配置（agent.json 内 edits 段）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llgraph.core.agent_config import (
    AGENT_CONFIG_FILENAME,
    LLGRAPH_DIR,
    load_agent_config,
)

# 已废弃：会话落盘统一在 ~/.llgraph/context/<工作区>/sessions/<thread_id>/
DEFAULT_SESSIONS_DIR = ".llgraph/sessions"

# 兼容旧 import 路径
__all__ = [
    "AGENT_CONFIG_FILENAME",
    "LLGRAPH_DIR",
    "DEFAULT_SESSIONS_DIR",
    "EditSettings",
    "load_agent_config",
    "resolve_edit_settings",
]


def _parse_bool(value: object, default: bool) -> bool:
    """
    解析布尔配置。

    @param value 配置值
    @param default 默认值
    @return 布尔结果
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    return bool(value)


@dataclass(frozen=True)
class EditSettings:
    """编辑与会话变更记录参数。"""

    prefer_search_replace: bool
    require_unique_match: bool
    persist_edits: bool
    sessions_dir: str  # 仅绝对路径可覆盖；.llgraph/sessions 视为用户目录
    snapshot_on_first_edit: bool
    write_chunk_max_chars: int
    write_failures_before_hint: int
    confirm_writes: str


def resolve_edit_settings(workspace: Path) -> EditSettings:
    """
    解析 edits 配置。

    @param workspace 工作区根
    @return EditSettings
    """
    cfg = load_agent_config(workspace)
    edits = cfg.get("edits") if isinstance(cfg.get("edits"), dict) else {}

    sessions_dir = edits.get("sessions_dir", DEFAULT_SESSIONS_DIR)
    if not isinstance(sessions_dir, str) or not sessions_dir.strip():
        sessions_dir = DEFAULT_SESSIONS_DIR

    writes = cfg.get("writes") if isinstance(cfg.get("writes"), dict) else {}
    if not writes and isinstance(edits.get("writes"), dict):
        writes = edits["writes"]

    chunk_max = writes.get("chunk_max_chars", 8000)
    try:
        chunk_max = max(1000, int(chunk_max))
    except (TypeError, ValueError):
        chunk_max = 8000

    fail_hint = writes.get("failures_before_hint", 2)
    try:
        fail_hint = max(1, int(fail_hint))
    except (TypeError, ValueError):
        fail_hint = 2

    confirm_raw = edits.get("confirm_writes", "interactive")
    if isinstance(confirm_raw, str):
        confirm_writes = confirm_raw.strip().lower()
    else:
        confirm_writes = "interactive"
    if confirm_writes not in ("interactive", "always", "never"):
        confirm_writes = "interactive"

    return EditSettings(
        prefer_search_replace=_parse_bool(edits.get("prefer_search_replace"), True),
        require_unique_match=_parse_bool(edits.get("require_unique_match"), True),
        persist_edits=_parse_bool(edits.get("persist_edits"), True),
        sessions_dir=sessions_dir.strip(),
        snapshot_on_first_edit=_parse_bool(edits.get("snapshot_on_first_edit"), True),
        write_chunk_max_chars=chunk_max,
        write_failures_before_hint=fail_hint,
        confirm_writes=confirm_writes,
    )
