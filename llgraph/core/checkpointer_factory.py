"""LangGraph 会话内存：进程内 MemorySaver；跨重启由 messages.jsonl 恢复。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from llgraph.session.user_storage import cleanup_obsolete_session_storage

_CHECKPOINTER_LOCK = threading.Lock()
_MEMORY_SAVER: MemorySaver | None = None


def create_checkpointer(workspace: Path, *, with_memory: bool) -> MemorySaver | None:
    """
    创建 checkpointer（仅内存；跨重启靠 session_file_store 恢复 jsonl）。

    @param workspace 工作区根
    @param with_memory 是否启用记忆
    @return MemorySaver 或 None
    """
    if not with_memory:
        return None

    cleanup_obsolete_session_storage(workspace.expanduser().resolve())

    global _MEMORY_SAVER
    with _CHECKPOINTER_LOCK:
        if _MEMORY_SAVER is None:
            _MEMORY_SAVER = MemorySaver()
        return _MEMORY_SAVER


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
