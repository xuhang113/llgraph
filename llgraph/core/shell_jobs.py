"""本进程内 shell 后台任务与会话 cwd（对齐 Cursor terminals / Claude Code Bash）。"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

from llgraph.sandbox.exec import LiveShellProcess

_MAX_KEPT_JOBS = 32
_COMPLETED_TTL_SEC = 900.0
_REAP_INTERVAL_SEC = 2.0


def command_fingerprint(command: str, cwd_rel: str) -> str:
    """相同命令 + cwd 视为同一后台任务。"""
    return f"{(cwd_rel or '.').strip()}|{command.strip()}"


@dataclass
class ShellJob:
    """一条可 await 的 shell 任务。"""

    job_id: str
    thread_id: str
    command: str
    cwd_rel: str
    fingerprint: str
    live: LiveShellProcess
    created_at: float


class ShellJobRegistry:
    """线程安全的 job 表 + 每会话 cwd。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ShellJob] = {}
        self._cwd: dict[str, str] = {}
        self._reaper_started = False

    def _ensure_reaper(self) -> None:
        if self._reaper_started:
            return
        self._reaper_started = True
        thread = threading.Thread(
            target=self._reaper_loop,
            name="llgraph-shell-reaper",
            daemon=True,
        )
        thread.start()

    def _reaper_loop(self) -> None:
        while True:
            time.sleep(_REAP_INTERVAL_SEC)
            try:
                self.reap()
            except Exception:
                pass

    def get_cwd(self, thread_id: str) -> str:
        """本会话上次工作目录（相对工作区）。"""
        with self._lock:
            return self._cwd.get(thread_id, ".")

    def set_cwd(self, thread_id: str, rel: str) -> None:
        """记住本会话工作目录。"""
        value = (rel or ".").strip() or "."
        with self._lock:
            self._cwd[thread_id] = value

    def running_count(self, thread_id: str | None = None) -> int:
        """正在运行的任务数。"""
        with self._lock:
            jobs = list(self._jobs.values())
        n = 0
        for job in jobs:
            if thread_id is not None and job.thread_id != thread_id:
                continue
            if job.live.returncode() is None and job.live.error is None:
                n += 1
            elif job.live.returncode() is None:
                # 已记录 error 但仍未 wait 完
                if job.live.proc.poll() is None:
                    n += 1
        return n

    def find_running(self, thread_id: str, fingerprint: str) -> ShellJob | None:
        """查找本会话中相同命令且仍在运行的任务。"""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.thread_id != thread_id:
                continue
            if job.fingerprint != fingerprint:
                continue
            if job.live.returncode() is None:
                return job
        return None

    def get(self, job_id: str) -> ShellJob | None:
        """按 id 取任务。"""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, thread_id: str | None = None) -> list[ShellJob]:
        """列出任务（新的在后）。"""
        with self._lock:
            jobs = list(self._jobs.values())
        if thread_id is not None:
            jobs = [job for job in jobs if job.thread_id == thread_id]
        return jobs

    def register(self, job: ShellJob) -> ShellJob:
        """登记任务。"""
        self._ensure_reaper()
        with self._lock:
            self._jobs[job.job_id] = job
            self._drop_oldest_unlocked()
        return job

    def _drop_oldest_unlocked(self) -> None:
        if len(self._jobs) <= _MAX_KEPT_JOBS:
            return
        completed = [
            job
            for job in self._jobs.values()
            if job.live.returncode() is not None
        ]
        completed.sort(key=lambda item: item.created_at)
        while len(self._jobs) > _MAX_KEPT_JOBS and completed:
            old = completed.pop(0)
            self._jobs.pop(old.job_id, None)

    def reap(self) -> None:
        """杀掉超过硬超时的进程，丢掉过期已完成任务。"""
        now_mono = time.monotonic()
        now_wall = time.perf_counter()
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            live = job.live
            if live.returncode() is None and live.hard_deadline and now_mono >= live.hard_deadline:
                live.kill("timeout")
        with self._lock:
            stale: list[str] = []
            for job_id, job in self._jobs.items():
                if job.live.returncode() is None:
                    continue
                if now_wall - job.created_at > _COMPLETED_TTL_SEC:
                    stale.append(job_id)
            for job_id in stale:
                self._jobs.pop(job_id, None)

    def kill_all(self, thread_id: str | None = None) -> int:
        """
        杀掉运行中的任务。

        @param thread_id 仅该会话；None 表示全部
        @return 发出终止的任务数
        """
        with self._lock:
            jobs = list(self._jobs.values())
        n = 0
        for job in jobs:
            if thread_id is not None and job.thread_id != thread_id:
                continue
            if job.live.returncode() is None:
                job.live.kill("cancelled")
                n += 1
        if thread_id is None:
            with self._lock:
                self._jobs.clear()
                self._cwd.clear()
        return n


_REGISTRY = ShellJobRegistry()


def get_shell_registry() -> ShellJobRegistry:
    """进程内全局 registry。"""
    return _REGISTRY


def new_job_id() -> str:
    """短 job id，如 sh-a1b2c3。"""
    return "sh-" + secrets.token_hex(3)


def reset_shell_runtime_for_tests() -> None:
    """测试用：杀掉全部任务并清空 cwd。"""
    _REGISTRY.kill_all()
    with _REGISTRY._lock:
        _REGISTRY._jobs.clear()
        _REGISTRY._cwd.clear()
