"""僵死 Web Agent 占用自动释放。"""

from __future__ import annotations

import time
from pathlib import Path

from llgraph.console.runtime import agent_service


def test_force_release_agent_chat_clears_registration() -> None:
    tid = "cli-stale-test"
    with agent_service._ACTIVE_AGENT_CHATS_LOCK:
        agent_service._ACTIVE_AGENT_CHATS.add(tid)
    assert agent_service.is_agent_chat_running(tid)
    assert agent_service.force_release_agent_chat(tid)
    assert not agent_service.is_agent_chat_running(tid)


def test_reconcile_stale_agent_chat_by_trace_mtime(tmp_path: Path, monkeypatch) -> None:
    tid = "cli-stale-trace"
    ws = tmp_path
    from llgraph.session.web_trace_store import live_web_trace_path

    path = live_web_trace_path(ws, tid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"steps": []}', encoding="utf-8")
    old = time.time() - 400
    import os

    os.utime(path, (old, old))

    with agent_service._ACTIVE_AGENT_CHATS_LOCK:
        agent_service._ACTIVE_AGENT_CHATS.add(tid)
        # 无存活 worker：登记残留，应按 idle 释放
        agent_service._ACTIVE_AGENT_CHAT_THREADS.pop(tid, None)

    assert agent_service.reconcile_stale_agent_chat(ws, tid, max_trace_idle_sec=180)
    assert not agent_service.is_agent_chat_running(tid)


def test_reconcile_skips_when_worker_thread_alive(tmp_path: Path) -> None:
    """长工具阻塞时 live_web_trace 可能长时间无更新，存活 worker 不得被误杀。"""
    import threading

    tid = "cli-spawn-long"
    ws = tmp_path
    from llgraph.session.web_trace_store import live_web_trace_path

    path = live_web_trace_path(ws, tid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"steps": []}', encoding="utf-8")
    old = time.time() - 400
    import os

    os.utime(path, (old, old))

    stop = threading.Event()

    def _block() -> None:
        stop.wait(30)

    worker = threading.Thread(target=_block, daemon=True)
    worker.start()
    try:
        with agent_service._ACTIVE_AGENT_CHATS_LOCK:
            agent_service._ACTIVE_AGENT_CHATS.add(tid)
            agent_service._ACTIVE_AGENT_CHAT_THREADS[tid] = worker
        assert not agent_service.reconcile_stale_agent_chat(ws, tid, max_trace_idle_sec=1)
        assert agent_service.is_agent_chat_running(tid)
    finally:
        stop.set()
        worker.join(timeout=2)
        agent_service.force_release_agent_chat(tid)
