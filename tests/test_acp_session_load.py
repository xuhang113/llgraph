"""ACP ``session/load`` 续聊：编辑器重启后拿旧 sessionId 接回来。

两层分开测：
- 回放映射（messages.jsonl 的三类消息 → session/update）用纯函数测；
- 协议这一层用真 pipe，测试侧扮编辑器，历史加载器可注入。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llgraph.editor.acp.jsonrpc import INVALID_PARAMS, INVALID_REQUEST
from llgraph.editor.acp.replay import (
    load_session_updates,
    replay_updates,
    session_is_resumable,
)
from llgraph.editor.acp.turn import AcpTurnRequest, AcpTurnResult
from llgraph.session.session_file_store import save_session_messages

from tests.test_acp_server import _Harness, workspace  # noqa: F401 - fixture 复用


def _texts(updates: list[dict[str, Any]], kind: str) -> list[str]:
    return [u["content"]["text"] for u in updates if u["sessionUpdate"] == kind]


# ---- 回放映射 ----


def test_replay_maps_user_assistant_and_tool() -> None:
    updates = replay_updates(
        [
            SystemMessage(content="你是 llgraph"),
            HumanMessage(content="读一下 hello.txt"),
            AIMessage(
                content="我先看看文件",
                tool_calls=[
                    {"id": "t1", "name": "read_file", "args": {"path": "hello.txt"}}
                ],
            ),
            ToolMessage(content="hello from llgraph", tool_call_id="t1", name="read_file"),
            AIMessage(content="里面只有一行问候。"),
        ]
    )

    kinds = [u["sessionUpdate"] for u in updates]
    assert kinds == [
        "user_message_chunk",
        "agent_message_chunk",
        "tool_call",
        "agent_message_chunk",
    ]
    assert _texts(updates, "user_message_chunk") == ["读一下 hello.txt"]

    tool_call = updates[2]
    assert tool_call["kind"] == "read"
    assert "read_file" in tool_call["title"]
    assert "hello.txt" in tool_call["title"]
    assert tool_call["status"] == "completed"
    assert "hello from llgraph" in tool_call["content"][0]["content"]["text"]


def test_replay_skips_system_and_injected_context() -> None:
    """manifest / anchor 是给模型看的，回放出来用户会以为自己发过这些话。"""
    from llgraph.context.conversation_anchor import CONVERSATION_ANCHOR_TAG
    from llgraph.session.session_manifest import SESSION_MANIFEST_TAG

    updates = replay_updates(
        [
            SystemMessage(content="系统提示"),
            HumanMessage(content=f"{SESSION_MANIFEST_TAG}\n工作区清单"),
            HumanMessage(content=f"{CONVERSATION_ANCHOR_TAG}\n上轮要点"),
            HumanMessage(content="真正的问题"),
        ]
    )
    assert [u["sessionUpdate"] for u in updates] == ["user_message_chunk"]
    assert _texts(updates, "user_message_chunk") == ["真正的问题"]


def test_replay_tool_call_ids_are_unique_and_prefixed() -> None:
    """回放 id 不能撞上实时那轮的 call_N，否则编辑器会把两次调用画成一次。"""
    updates = replay_updates(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "read_file", "args": {"path": "a.txt"}},
                    {"id": "t2", "name": "read_file", "args": {"path": "b.txt"}},
                ],
            ),
            ToolMessage(content="A", tool_call_id="t1", name="read_file"),
            ToolMessage(content="B", tool_call_id="t2", name="read_file"),
        ]
    )
    ids = [u["toolCallId"] for u in updates]
    assert ids == ["load_1", "load_2"]
    assert all(not i.startswith("call_") for i in ids)
    # 工具结果已经并进 tool_call，不该再单独出现一条
    assert len(updates) == 2


def test_replay_marks_failed_tool_and_truncates_output() -> None:
    updates = replay_updates(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "t1", "name": "grep", "args": {"pattern": "x"}}],
            ),
            ToolMessage(
                content="\n".join(f"line{i}" for i in range(100)),
                tool_call_id="t1",
                name="grep",
                status="error",
            ),
        ],
        max_tool_output_lines=5,
    )
    assert updates[0]["status"] == "failed"
    text = updates[0]["content"][0]["content"]["text"]
    assert text.count("\n") == 5
    assert "还有 95 行" in text


def test_replay_orphan_tool_result_still_shows_up() -> None:
    """条数上限会把 AIMessage 切掉，剩下的工具结果不能凭空消失。"""
    updates = replay_updates(
        [
            HumanMessage(content="第一问"),
            AIMessage(
                content="",
                tool_calls=[{"id": "t1", "name": "read_file", "args": {"path": "a.txt"}}],
            ),
            ToolMessage(content="A", tool_call_id="t1", name="read_file"),
            AIMessage(content="答复"),
        ],
        max_messages=2,
    )
    kinds = [u["sessionUpdate"] for u in updates]
    assert kinds == ["agent_thought_chunk", "tool_call", "agent_message_chunk"]
    assert "2 条历史未回放" in updates[0]["content"]["text"]
    assert "read_file" in updates[1]["title"]


def test_replay_empty_history_is_empty() -> None:
    assert replay_updates([]) == []


# ---- 读盘 ----


def test_load_session_updates_reads_messages_jsonl(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    save_session_messages(
        workspace_dir,
        "cli-load01",
        [HumanMessage(content="上次问的问题"), AIMessage(content="上次的答复")],
    )

    assert session_is_resumable(workspace_dir, "cli-load01") is True
    updates = load_session_updates(workspace_dir, "cli-load01")
    assert _texts(updates, "user_message_chunk") == ["上次问的问题"]
    assert "上次的答复" in "".join(_texts(updates, "agent_message_chunk"))


def test_session_is_resumable_needs_this_workspace(tmp_path: Path) -> None:
    """别的工作区的会话不能被接回来：历史按工作区分目录存。"""
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()
    save_session_messages(ws_a, "cli-load02", [HumanMessage(content="hi")])

    assert session_is_resumable(ws_a, "cli-load02") is True
    assert session_is_resumable(ws_b, "cli-load02") is False


def test_session_with_meta_only_is_resumable(tmp_path: Path) -> None:
    """开了会话还没说话就重启，也该接得回去（回放为空）。"""
    from llgraph.console.runtime.agent_service import create_agent_session

    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    session_id = create_agent_session(workspace_dir)

    assert session_is_resumable(workspace_dir, session_id) is True
    assert load_session_updates(workspace_dir, session_id) == []


# ---- 协议层 ----


def test_initialize_declares_load_session(workspace: Path) -> None:
    h = _Harness()
    try:
        h.send("initialize", {"protocolVersion": 1}, request_id=1)
        assert h.response(1)["result"]["agentCapabilities"]["loadSession"] is True
    finally:
        h.close()


def test_session_load_replays_history_before_responding(workspace: Path) -> None:
    seen: list[tuple[Path, str]] = []

    def loader(ws: Path, thread_id: str) -> list[dict[str, Any]]:
        seen.append((ws, thread_id))
        return [
            {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "旧问题"}},
            {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "旧答复"}},
        ]

    h = _Harness(history_loader=loader)
    try:
        h.send("initialize", {"protocolVersion": 1}, request_id=1)
        h.response(1)
        h.send("session/new", {"cwd": str(workspace)}, request_id=2)
        session_id = h.response(2)["result"]["sessionId"]

        h.send("session/load", {"sessionId": session_id, "cwd": str(workspace)}, request_id=3)
        assert h.response(3)["result"] == {}
        assert seen == [(workspace.resolve(), session_id)]

        kinds = [u["sessionUpdate"] for u in h.updates()]
        assert kinds == ["user_message_chunk", "agent_message_chunk"]
        # 回放必须在回包之前送达，否则编辑器会先以为加载完了
        order = [m.get("method") or f"response:{m.get('id')}" for m in h.out.messages]
        assert order.index("session/update") < order.index("response:3")
    finally:
        h.close()


def test_session_load_rejects_unknown_session(workspace: Path) -> None:
    h = _Harness()
    try:
        h.send("session/load", {"sessionId": "cli-nope", "cwd": str(workspace)}, request_id=1)
        assert h.response(1)["error"]["code"] == INVALID_PARAMS
    finally:
        h.close()


def test_session_load_rejects_missing_session_id(workspace: Path) -> None:
    h = _Harness()
    try:
        h.send("session/load", {"cwd": str(workspace)}, request_id=1)
        assert h.response(1)["error"]["code"] == INVALID_PARAMS
    finally:
        h.close()


def test_loaded_session_can_be_prompted(workspace: Path) -> None:
    """接回来之后要能接着聊——这才是「续聊」，不是只把历史画一遍。"""
    seen: dict[str, Any] = {}

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        seen["thread_id"] = req.thread_id
        seen["workspace"] = req.workspace
        return AcpTurnResult(text="接着聊")

    h = _Harness(turn_runner=runner, history_loader=lambda ws, tid: [])
    try:
        h.send("session/new", {"cwd": str(workspace)}, request_id=1)
        session_id = h.response(1)["result"]["sessionId"]

        # 另起一台「重启后的编辑器」：新连接上只有 sessionId
        h2 = _Harness(turn_runner=runner, history_loader=lambda ws, tid: [])
        try:
            h2.send("session/load", {"sessionId": session_id, "cwd": str(workspace)}, request_id=1)
            assert "error" not in h2.response(1)
            h2.send(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "继续"}]},
                request_id=2,
            )
            assert h2.response(2)["result"]["stopReason"] == "end_turn"
            assert seen["thread_id"] == session_id
            assert seen["workspace"] == workspace.resolve()
        finally:
            h2.close()
    finally:
        h.close()


def test_prompt_during_load_is_rejected(workspace: Path) -> None:
    release = threading.Event()

    def loader(ws: Path, thread_id: str) -> list[dict[str, Any]]:
        release.wait(timeout=5)
        return []

    h = _Harness(
        history_loader=loader,
        turn_runner=lambda req, **kw: AcpTurnResult(text="不该跑到这里"),
    )
    try:
        h.send("session/new", {"cwd": str(workspace)}, request_id=1)
        session_id = h.response(1)["result"]["sessionId"]
        h.send("session/load", {"sessionId": session_id, "cwd": str(workspace)}, request_id=2)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "急着问"}]},
            request_id=3,
        )
        assert h.response(3)["error"]["code"] == INVALID_REQUEST
        release.set()
        assert h.response(2)["result"] == {}
    finally:
        h.close()


def test_load_failure_returns_error_and_frees_session(workspace: Path) -> None:
    def loader(ws: Path, thread_id: str) -> list[dict[str, Any]]:
        raise RuntimeError("历史文件读不动")

    h = _Harness(history_loader=loader)
    try:
        h.send("session/new", {"cwd": str(workspace)}, request_id=1)
        session_id = h.response(1)["result"]["sessionId"]
        h.send("session/load", {"sessionId": session_id, "cwd": str(workspace)}, request_id=2)
        assert "历史文件读不动" in h.response(2)["error"]["message"]
        # 加载失败不能把会话一直占着
        assert h.server._sessions[session_id].busy is False
    finally:
        h.close()


def test_load_falls_back_to_default_workspace(workspace: Path) -> None:
    """编辑器没给 cwd 时用 ``llgraph acp -C`` 指定的工作区（与 session/new 一致）。"""
    h = _Harness(default_workspace=workspace, history_loader=lambda ws, tid: [])
    try:
        h.send("session/new", {}, request_id=1)
        session_id = h.response(1)["result"]["sessionId"]
        h.send("session/load", {"sessionId": session_id}, request_id=2)
        assert h.response(2)["result"] == {}
    finally:
        h.close()


@pytest.mark.parametrize("cwd", ["relative/dir", ""])
def test_load_rejects_bad_cwd(cwd: str) -> None:
    h = _Harness()
    try:
        h.send("session/load", {"sessionId": "cli-x", "cwd": cwd}, request_id=1)
        assert h.response(1)["error"]["code"] == INVALID_PARAMS
    finally:
        h.close()
