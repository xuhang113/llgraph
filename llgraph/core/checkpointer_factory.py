"""LangGraph 会话内存：进程内 MemorySaver；跨重启由 messages.jsonl 恢复。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from llgraph.session.user_storage import cleanup_obsolete_session_storage

_CHECKPOINTER_LOCK = threading.Lock()
_MEMORY_SAVERS: dict[str, MemorySaver] = {}


def create_checkpointer(
    workspace: Path,
    *,
    with_memory: bool,
    thread_key: str | None = None,
) -> MemorySaver | None:
    """
    创建 checkpointer（仅内存；跨重启靠 session_file_store 恢复 jsonl）。

    每个 thread_key 独立 MemorySaver，并行 Worker 互不干扰。

    @param workspace 工作区根
    @param with_memory 是否启用记忆
    @param thread_key checkpoint 隔离键（如 plan-xxx:worker:w1）
    @return MemorySaver 或 None
    """
    if not with_memory:
        return None

    cleanup_obsolete_session_storage(workspace.expanduser().resolve())

    key = (thread_key or "").strip() or "__default__"
    with _CHECKPOINTER_LOCK:
        saver = _MEMORY_SAVERS.get(key)
        if saver is None:
            saver = MemorySaver()
            _MEMORY_SAVERS[key] = saver
        return saver


def checkpointer_kind(workspace: Path, *, with_memory: bool) -> str:
    """
    当前会话持久化方式说明。

    @param workspace 工作区根
    @param with_memory 是否启用记忆
    @return jsonl | none
    """
    if not with_memory:
        return "none"
    return "jsonl"
