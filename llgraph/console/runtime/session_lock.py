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


def release_stale_web_lock(thread_id: str) -> bool:
    """
    后台 job 已结束但锁未释放时清理（避免停止 Plan 后仍无法删除）。

    @param thread_id 会话 ID
    @return 是否释放了 stale 锁
    """
    from llgraph.plan.execution_coordinator import is_running

    if is_running(thread_id):
        return False
    info = LOCKS.get(thread_id)
    if info is None or info.owner != "web":
        return False
    LOCKS.release(thread_id, owner="web")
    return True


def delete_lock_block_reason(thread_id: str) -> str | None:
    """
    删除前检查 session 锁；job 未跑时可清理 stale web 锁。

    @param thread_id 会话 ID
    @return 阻塞原因；None 表示可删
    """
    from llgraph.plan.execution_coordinator import is_running

    if is_running(thread_id):
        return "Plan 仍在执行中，请先取消并等待当前 Work 结束后再删除"
    info = LOCKS.get(thread_id)
    if info is None:
        return None
    if info.owner == "web" and release_stale_web_lock(thread_id):
        return None
    return "会话正在使用中，请稍后再试"
