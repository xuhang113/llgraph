"""会话锁：避免 Web 与 CLI 同时操作同 thread。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class SessionLockInfo:
    """锁信息。"""

    thread_id: str
    owner: str
    since: float


class SessionLockRegistry:
    """进程内 session 锁。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionLockInfo] = {}

    def acquire(self, thread_id: str, owner: str = "web") -> tuple[bool, SessionLockInfo | None]:
        """
        尝试获取锁。

        @param thread_id 会话 ID
        @param owner 持有者标识
        @return (是否成功, 当前持有者)
        """
        with self._lock:
            existing = self._sessions.get(thread_id)
            if existing is not None and existing.owner != owner:
                return False, existing
            info = SessionLockInfo(thread_id=thread_id, owner=owner, since=time.time())
            self._sessions[thread_id] = info
            return True, info

    def release(self, thread_id: str, owner: str = "web") -> None:
        """
        释放锁。

        @param thread_id 会话 ID
        @param owner 持有者
        """
        with self._lock:
            existing = self._sessions.get(thread_id)
            if existing is not None and existing.owner == owner:
                del self._sessions[thread_id]

    def get(self, thread_id: str) -> SessionLockInfo | None:
        """
        查询锁状态。

        @param thread_id 会话 ID
        @return 锁信息
        """
        with self._lock:
            return self._sessions.get(thread_id)


LOCKS = SessionLockRegistry()
