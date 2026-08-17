"""记忆整理锁。"""

from __future__ import annotations

import os
from pathlib import Path

from llgraph.memory.paths import LOCK_FILENAME, memory_root


class MemoryConsolidateLock:
    """基于 flock 的整理互斥锁。"""

    def __init__(self, user_id: str, workspace_key: str) -> None:
        root = memory_root(user_id, workspace_key)
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / LOCK_FILENAME
        self._handle = None

    def try_acquire(self) -> bool:
        """尝试获取锁。"""
        if self._handle is not None:
            return True
        try:
            handle = self._path.open("a+", encoding="utf-8")
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            self._handle = handle
            return True
        except (OSError, BlockingIOError):
            return False

    def release(self) -> None:
        """释放锁。"""
        if self._handle is None:
            return
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
        except OSError:
            pass
        self._handle = None
