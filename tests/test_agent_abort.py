"""Web Agent Stop / 取消。"""

from __future__ import annotations

from llgraph.console.runtime.agent_service import (
    abort_agent_chat,
    clear_agent_cancel,
    is_agent_cancel_requested,
    is_agent_chat_running,
    request_agent_cancel,
    try_register_agent_chat,
)


def test_request_agent_cancel_when_not_running() -> None:
    clear_agent_cancel("cli-nope")
    assert abort_agent_chat("cli-nope") == {
        "ok": False,
        "message": "当前无进行中的 Agent 对话",
    }


def test_cancel_flag_lifecycle() -> None:
    tid = "cli-cancel1"
    clear_agent_cancel(tid)
    from llgraph.console.runtime import agent_service

    with agent_service._ACTIVE_AGENT_CHATS_LOCK:
        agent_service._ACTIVE_AGENT_CHATS.add(tid)
    try:
        assert request_agent_cancel(tid) is True
        assert is_agent_cancel_requested(tid) is True
        assert abort_agent_chat(tid)["ok"] is True
        assert is_agent_cancel_requested(tid) is True
    finally:
        with agent_service._ACTIVE_AGENT_CHATS_LOCK:
            agent_service._ACTIVE_AGENT_CHATS.discard(tid)
        clear_agent_cancel(tid)


def test_try_register_agent_chat_rejects_duplicate() -> None:
    tid = "cli-dup-reg"
    from llgraph.console.runtime import agent_service

    with agent_service._ACTIVE_AGENT_CHATS_LOCK:
        agent_service._ACTIVE_AGENT_CHATS.discard(tid)
    try:
        assert try_register_agent_chat(tid) is True
        assert is_agent_chat_running(tid) is True
        assert try_register_agent_chat(tid) is False
    finally:
        with agent_service._ACTIVE_AGENT_CHATS_LOCK:
            agent_service._ACTIVE_AGENT_CHATS.discard(tid)
