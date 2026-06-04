"""索引进程互斥锁（全量 index 与 watch 增量互斥）。"""

from __future__ import annotations

import os
from pathlib import Path

from llgraph.code_index.paths import ensure_index_dirs, index_root


class IndexLock:
    """
    基于 flock 的索引锁；非阻塞 try_acquire。
    """

    def __init__(self, workspace: Path) -> None:
        ensure_index_dirs(workspace)
        self._path = index_root(workspace) / ".index.lock"
        self._handle = None

    def try_acquire(self) -> bool:
        """
        尝试获取锁。

        @return 是否成功
        """
        if self._handle is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
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

    def __enter__(self) -> IndexLock:
        """上下文管理器进入。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """上下文管理器退出时释放。"""
        self.release()
