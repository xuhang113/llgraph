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


def _memory_store_has_rows(workspace: Path) -> bool:
    from llgraph.memory.paths import ACTIVE_KINDS, workspace_identity
    from llgraph.memory.store import list_memory_rows

    user_id, workspace_key, _ = workspace_identity(workspace)
    return bool(list_memory_rows(user_id, workspace_key, status="active", kinds=ACTIVE_KINDS))


def schedule_memory_embedder_prewarm(workspace: Path) -> threading.Thread | None:
    """
    会话建好后后台加载本地 embedding 模型，把冷启动挪出首轮 TTFT。

    仅在「已有记忆可召回」且用的是本地模型时预热：空库首轮根本不会 embed，
    预热只会白占内存。

    @param workspace 工作区根
    @return 预热线程；未触发时 None
    """
    try:
        if not resolve_memory_settings(workspace).enabled:
            return None
        from llgraph.code_index.embedding_config import resolve_embedding_profile
        from llgraph.code_index.local_embedder import local_embedder_is_loaded

        profile = resolve_embedding_profile(workspace)
        if profile.provider != "local" or local_embedder_is_loaded(profile):
            return None
        if not _memory_store_has_rows(workspace):
            return None
    except Exception:
        return None

    def _run() -> None:
        try:
            from llgraph.code_index.local_embedder import prewarm_local_embedder

            prewarm_local_embedder(profile)
        except Exception:
            pass

    thread = threading.Thread(target=_run, name="memory-embedder-prewarm", daemon=True)
    thread.start()
    return thread
