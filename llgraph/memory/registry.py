"""已触发长期记忆的工作区注册表（供定时整理扫描）。"""

from __future__ import annotations

import threading
from pathlib import Path

_lock = threading.Lock()
_known: dict[str, Path] = {}


def register_memory_workspace(workspace: Path) -> None:
    """登记工作区路径。"""
    ws = workspace.expanduser().resolve()
    key = str(ws)
    with _lock:
        _known[key] = ws


def iter_registered_workspaces() -> list[Path]:
    """返回已登记工作区列表。"""
    with _lock:
        return list(_known.values())
