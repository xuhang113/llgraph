"""Plan 图后台执行与中断协调（终端 detach / Web 可复用）。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

_registry_lock = threading.Lock()
_jobs: dict[str, "PlanBackgroundJob"] = {}
_cancelled_tasks: dict[str, set[str]] = {}


@dataclass
class PlanBackgroundJob:
    """单个 Plan thread 的后台执行任务。"""

    thread_id: str
    cancel_requested: bool = False
    running: bool = False
    error: str | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)


def is_running(thread_id: str) -> bool:
    """
    Plan 图是否仍在后台执行。

    @param thread_id plan-* thread
    @return 是否运行中
    """
    with _registry_lock:
        job = _jobs.get(thread_id)
        return bool(job and job.running)


def is_cancel_requested(thread_id: str) -> bool:
    """
    是否已请求停止（当前 task 完成后生效）。

    @param thread_id plan-* thread
    @return 是否请求取消
    """
    with _registry_lock:
        job = _jobs.get(thread_id)
        return bool(job and job.cancel_requested)


def request_cancel_task(thread_id: str, task_id: str) -> None:
    """
    请求停止单个 Work task（当前 batch 结束后生效；in-flight 轮询检查）。

    @param thread_id plan-* thread
    @param task_id 如 w1
    """
    key = (task_id or "").strip()
    if not key:
        return
    with _registry_lock:
        _cancelled_tasks.setdefault(thread_id, set()).add(key)


def is_task_cancel_requested(thread_id: str, task_id: str) -> bool:
    """@param thread_id plan thread @param task_id work id @return 是否请求取消该 task"""
    key = (task_id or "").strip()
    if not key:
        return False
    with _registry_lock:
        return key in _cancelled_tasks.get(thread_id, set())


def clear_cancelled_tasks(thread_id: str, *, task_ids: list[str] | None = None) -> None:
    """
    清除 task 取消标记。

    @param thread_id plan thread
    @param task_ids 指定 id；None 表示清空该 plan 下全部
    """
    with _registry_lock:
        if task_ids is None:
            _cancelled_tasks.pop(thread_id, None)
            return
        bucket = _cancelled_tasks.get(thread_id)
        if not bucket:
            return
        for tid in task_ids:
            bucket.discard(str(tid).strip())
        if not bucket:
            _cancelled_tasks.pop(thread_id, None)


def request_cancel_all_tasks(thread_id: str, task_ids: list[str]) -> None:
    """
    请求停止 Plan 下多个 Work task。

    @param thread_id plan-* thread
    @param task_ids 如 t1,t2
    """
    keys = [str(tid).strip() for tid in task_ids if str(tid).strip()]
    if not keys:
        return
    with _registry_lock:
        bucket = _cancelled_tasks.setdefault(thread_id, set())
        bucket.update(keys)


def request_cancel(thread_id: str) -> bool:
    """
    请求停止 Plan 执行。

    @param thread_id plan-* thread
    @return 是否有活跃任务
    """
    with _registry_lock:
        job = _jobs.get(thread_id)
        if job is None or not job.running:
            return False
        job.cancel_requested = True
        return True


def start_background(
    thread_id: str,
    runner: Callable[[], None],
    *,
    on_complete: Callable[[], None] | None = None,
) -> bool:
    """
    在后台线程执行 Plan 图 invoke。

    @param thread_id plan-* thread
    @param runner 同步执行函数（内部应调用 run_until_interrupt）
    @param on_complete 完成后回调（终端刷新 state）
    @return False 表示已有任务在跑
    """
    with _registry_lock:
        existing = _jobs.get(thread_id)
        if existing is not None and existing.running:
            return False
        job = PlanBackgroundJob(thread_id=thread_id, running=True)

        def _wrapper() -> None:
            try:
                runner()
            except Exception as exc:
                job.error = str(exc)
            finally:
                job.running = False
                try:
                    from llgraph.console.runtime.session_lock import release_stale_web_lock

                    release_stale_web_lock(thread_id)
                except Exception:
                    pass
                if on_complete is not None:
                    try:
                        on_complete()
                    except Exception:
                        pass

        job._thread = threading.Thread(
            target=_wrapper,
            name=f"llgraph-plan-{thread_id}",
            daemon=False,
        )
        _jobs[thread_id] = job
        job._thread.start()
        return True


def wait_until_done(thread_id: str, *, timeout: float | None = None) -> bool:
    """
    阻塞等待后台任务结束。

    @param thread_id plan-* thread
    @param timeout 秒；None 表示一直等
    @return 是否在超时前结束
    """
    with _registry_lock:
        job = _jobs.get(thread_id)
        thread = job._thread if job else None
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


def job_snapshot(thread_id: str) -> dict[str, str | bool | None]:
    """
    后台任务快照（status / Web 轮询）。

    @param thread_id plan-* thread
    @return 状态 dict
    """
    with _registry_lock:
        job = _jobs.get(thread_id)
        if job is None:
            return {"running": False, "cancel_requested": False, "error": None}
        return {
            "running": job.running,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
        }
