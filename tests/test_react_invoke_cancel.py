"""可中断 LLM invoke（Web Stop）。"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from llgraph.context.runtime_context import set_active_thread_id
from llgraph.core.react_invoke import invoke_agent_runnable_cancellable
from llgraph.console.runtime.agent_service import (
    clear_agent_cancel,
    request_agent_cancel,
)
from llgraph.session.session_run_log import UserCancelledError


def test_invoke_cancellable_raises_immediately_when_already_cancelled() -> None:
    tid = "cli-inv-cancel-1"
    clear_agent_cancel(tid)
    from llgraph.console.runtime import agent_service

    with agent_service._ACTIVE_AGENT_CHATS_LOCK:
        agent_service._ACTIVE_AGENT_CHATS.add(tid)
    set_active_thread_id(tid)
    try:
        request_agent_cancel(tid)
        runnable = MagicMock()
        try:
            invoke_agent_runnable_cancellable(runnable, {}, {})
            raise AssertionError("expected UserCancelledError")
        except UserCancelledError:
            pass
        runnable.stream.assert_not_called()
    finally:
        set_active_thread_id(None)
        clear_agent_cancel(tid)
        with agent_service._ACTIVE_AGENT_CHATS_LOCK:
            agent_service._ACTIVE_AGENT_CHATS.discard(tid)


def test_invoke_cancellable_aborts_during_slow_stream() -> None:
    tid = "cli-inv-cancel-2"
    clear_agent_cancel(tid)
    from llgraph.console.runtime import agent_service

    with agent_service._ACTIVE_AGENT_CHATS_LOCK:
        agent_service._ACTIVE_AGENT_CHATS.add(tid)
    set_active_thread_id(tid)

    def slow_stream(_state, _config):
        yield AIMessageChunk(content="x")
        time.sleep(2.0)
        yield AIMessageChunk(content="y")

    runnable = MagicMock()
    runnable.stream.side_effect = slow_stream

    try:
        import threading

        def cancel_soon() -> None:
            time.sleep(0.15)
            request_agent_cancel(tid)

        threading.Thread(target=cancel_soon, daemon=True).start()
        t0 = time.perf_counter()
        try:
            invoke_agent_runnable_cancellable(runnable, {}, {})
            raise AssertionError("expected UserCancelledError")
        except UserCancelledError:
            pass
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.5, elapsed
    finally:
        set_active_thread_id(None)
        clear_agent_cancel(tid)
        with agent_service._ACTIVE_AGENT_CHATS_LOCK:
            agent_service._ACTIVE_AGENT_CHATS.discard(tid)


if __name__ == "__main__":
    test_invoke_cancellable_raises_immediately_when_already_cancelled()
    test_invoke_cancellable_aborts_during_slow_stream()
    print("ok")
