"""轮次开始用户消息早落盘测试。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from llgraph.session.session_file_store import (
    append_pending_user_turn,
    load_session_messages,
    save_session_messages,
)


def test_append_pending_user_turn_persists_before_turn_end(tmp_path: Path) -> None:
    thread_id = "cli-test01"
    save_session_messages(
        tmp_path,
        thread_id,
        [HumanMessage(content="你好"), AIMessage(content="你好！")],
    )
    path = append_pending_user_turn(tmp_path, thread_id, "继续帮我改代码")
    assert path is not None
    loaded = load_session_messages(tmp_path, thread_id)
    assert len(loaded) == 3
    assert isinstance(loaded[-1], HumanMessage)
    assert loaded[-1].content == "继续帮我改代码"


def test_append_pending_user_turn_idempotent_mid_turn(tmp_path: Path) -> None:
    thread_id = "cli-test02"
    append_pending_user_turn(tmp_path, thread_id, "进行中")
    path = append_pending_user_turn(tmp_path, thread_id, "进行中")
    assert path is not None
    loaded = load_session_messages(tmp_path, thread_id)
    assert len(loaded) == 1


def test_append_pending_user_turn_allows_repeat_after_assistant(tmp_path: Path) -> None:
    thread_id = "cli-test03"
    append_pending_user_turn(tmp_path, thread_id, "相同")
    save_session_messages(
        tmp_path,
        thread_id,
        [HumanMessage(content="相同"), AIMessage(content="ok")],
    )
    append_pending_user_turn(tmp_path, thread_id, "相同")
    loaded = load_session_messages(tmp_path, thread_id)
    assert len(loaded) == 3
    assert loaded[-1].content == "相同"
