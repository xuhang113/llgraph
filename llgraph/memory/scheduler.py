"""长期记忆定时整理调度器。"""

from __future__ import annotations

import atexit
import threading
import time
from pathlib import Path

from llgraph.memory.consolidate import consolidate_workspace_memory
from llgraph.memory.registry import iter_registered_workspaces, register_memory_workspace
from llgraph.memory.settings import resolve_memory_settings
from llgraph.memory.trace_emit import emit_memory_consolidate_trace_step

_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()
_started = False
_lock = threading.Lock()


def _scheduler_loop() -> None:
    first_delay_sec: float | None = None
    while not _stop_event.is_set():
        workspaces = iter_registered_workspaces()
        if workspaces:
            ws0 = workspaces[0]
            settings = resolve_memory_settings(ws0)
            if first_delay_sec is None:
                first_delay_sec = max(0.0, settings.consolidate_on_startup_delay_minutes * 60.0)
            if first_delay_sec is not None and first_delay_sec > 0:
                if _stop_event.wait(first_delay_sec):
                    break
                first_delay_sec = 0.0
            interval = max(3600.0, settings.scheduled_consolidate_hours * 3600.0)
            for ws in workspaces:
                if _stop_event.is_set():
                    break
                settings = resolve_memory_settings(ws)
                if not settings.enabled:
                    continue
                report = consolidate_workspace_memory(ws)
                emit_memory_consolidate_trace_step(
                    merged=report.merged,
                    replaced=report.replaced,
                    pruned_ttl=report.pruned_ttl,
                    pruned_cap=report.pruned_cap,
                    elapsed_sec=report.elapsed_sec,
                )
            if _stop_event.wait(interval):
                break
        else:
            if _stop_event.wait(60.0):
                break


def start_memory_consolidate_scheduler() -> None:
    """进程启动时注册后台整理线程。"""
    global _scheduler_thread, _started
    with _lock:
        if _started:
            return
        _started = True
        _stop_event.clear()
        _scheduler_thread = threading.Thread(
            target=_scheduler_loop,
            name="llgraph-memory-consolidate",
            daemon=True,
        )
        _scheduler_thread.start()


def stop_memory_consolidate_scheduler() -> None:
    """停止调度器。"""
    global _started
    _stop_event.set()
    thread = _scheduler_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=30.0)
    with _lock:
        _started = False


def attach_memory_scheduler_shutdown() -> None:
    """atexit 停止调度器。"""
    atexit.register(stop_memory_consolidate_scheduler)


def schedule_light_consolidate(workspace: Path) -> None:
    """轮次结束后异步轻量整理。"""
    register_memory_workspace(workspace)

    def _run() -> None:
        consolidate_workspace_memory(workspace, light=True)

    threading.Thread(target=_run, name="memory-light-consolidate", daemon=True).start()


def touch_memory_workspace(workspace: Path) -> None:
    """记忆读写前登记工作区。"""
    register_memory_workspace(workspace)
