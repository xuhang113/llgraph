"""ACP 端到端：编辑器提一个问题 → 真 ReAct 跑一轮工具 → 过程与正文按 ACP 推回去。

不打真模型：本机起一个按 Anthropic 协议（网关那条默认路径）应答的 stub server，
第一轮让它要求调 read_file，第二轮给正文。测的是「真 trace 事件 → session/update」，
手搓步骤的单测覆盖不到这一段。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from llgraph.config.config import ENV_API_BASE_URL, ENV_API_KEY, ENV_MODEL
from llgraph.core.llm_settings import set_runtime_model
from llgraph.editor.acp.turn import AcpTurnRequest, run_acp_turn
from llgraph.permissions.approval import ApprovalDecision, ApprovalRequest

_FAKE_KEY = "test-key-not-a-secret"
_FILE_BODY = "hello from llgraph\n"
_REPLY = "hello.txt 里只有一行问候。"


def _completed_tool_call(updates: list[dict[str, Any]]) -> dict[str, Any]:
    """@param updates 收到的 session/update @return 带输出的那条完成态工具更新"""
    return next(
        u
        for u in updates
        if u["sessionUpdate"] in ("tool_call", "tool_call_update")
        and u.get("status") == "completed"
    )


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _tool_use_stream(tool_name: str = "read_file", tool_input: Any = None) -> bytes:
    """第一轮：模型要求调一个工具。"""
    args = json.dumps(tool_input if tool_input is not None else {"path": "hello.txt"})
    return b"".join(
        [
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "stub",
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": tool_name,
                        "input": {},
                    },
                },
            ),
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                },
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 8},
                },
            ),
            _sse("message_stop", {"type": "message_stop"}),
        ]
    )


def _text_stream() -> bytes:
    """第二轮：拿到工具结果后给正文（分两段发，顺便验证流式落成多个 chunk）。"""
    events = [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "model": "stub",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                },
            },
        ),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for piece in (_REPLY[:6], _REPLY[6:]):
        events.append(
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": piece},
                },
            )
        )
    events.extend(
        [
            _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 12},
                },
            ),
            _sse("message_stop", {"type": "message_stop"}),
        ]
    )
    return b"".join(events)


def _has_tool_result(body: dict[str, Any]) -> bool:
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        ):
            return True
    return False


class _StubState:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        # 第一轮让模型要求调哪个工具（写入那条用例换成 search_replace）
        self.tool_name = "read_file"
        self.tool_input: dict[str, Any] = {"path": "hello.txt"}

    def tool_results(self) -> list[str]:
        """@return 回灌给模型的工具结果文本（验收「模型看到了拒绝」用）"""
        out: list[str] = []
        for body in self.requests:
            for message in body.get("messages", []):
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        raw = block.get("content")
                        if isinstance(raw, str):
                            out.append(raw)
                        elif isinstance(raw, list):
                            out.extend(
                                str(part.get("text") or "")
                                for part in raw
                                if isinstance(part, dict)
                            )
        return out


def _make_handler(state: _StubState):
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            with state.lock:
                state.requests.append(body)

            if self.path.endswith("/v1/embeddings"):
                data = json.dumps(
                    {"data": [{"embedding": [0.0] * 8, "index": 0}], "model": "stub"}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            wants_reply = _has_tool_result(body)
            if body.get("stream"):
                payload = (
                    _text_stream()
                    if wants_reply
                    else _tool_use_stream(state.tool_name, state.tool_input)
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            content = (
                [{"type": "text", "text": _REPLY}]
                if wants_reply
                else [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": state.tool_name,
                        "input": state.tool_input,
                    }
                ]
            )
            data = json.dumps(
                {
                    "id": "msg_stub",
                    "type": "message",
                    "role": "assistant",
                    "model": "stub",
                    "content": content,
                    "stop_reason": "end_turn" if wants_reply else "tool_use",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler


@pytest.fixture
def stub_gateway(monkeypatch: pytest.MonkeyPatch) -> _StubState:
    state = _StubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    monkeypatch.setenv(ENV_API_BASE_URL, f"http://{host}:{port}")
    monkeypatch.setenv(ENV_API_KEY, _FAKE_KEY)
    monkeypatch.setenv(ENV_MODEL, "claude-opus-4-6")
    set_runtime_model(None)
    try:
        yield state
    finally:
        set_runtime_model(None)
        server.shutdown()
        server.server_close()


@pytest.fixture
def clean_runtime() -> None:
    from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER

    yield
    RUNTIME_MANAGER.shutdown_all()
    # 会话保活池是进程级的：留着不清，别的用例一调 get_or_build 就会触发淘汰，
    # 把 release_checkpointer 的调用记到人家的 mock 上
    from llgraph.core.agent_session_pool import (
        agent_session_pool_stats,
        invalidate_agent_session_thread,
    )

    for entry in agent_session_pool_stats()["threads"]:
        invalidate_agent_session_thread(Path(entry["workspace"]), entry["thread_id"])


def test_acp_turn_streams_tool_call_and_reply(
    tmp_path: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hello.txt").write_text(_FILE_BODY, encoding="utf-8")

    updates: list[dict[str, Any]] = []
    result = run_acp_turn(
        AcpTurnRequest(
            workspace=workspace,
            thread_id="cli-acpe2e1",
            message="读一下 hello.txt 并告诉我内容",
        ),
        send_update=updates.append,
        cancel_check=lambda: False,
    )

    assert result.stop_reason == "end_turn"
    assert _REPLY in result.text

    kinds = [u["sessionUpdate"] for u in updates]
    assert "tool_call" in kinds, kinds
    assert "agent_message_chunk" in kinds, kinds

    announced = next(u for u in updates if u["sessionUpdate"] == "tool_call")
    assert announced["kind"] == "read"
    assert "read_file" in announced["title"]
    tool_call = _completed_tool_call(updates)
    assert "hello from llgraph" in tool_call["content"][0]["content"]["text"]

    streamed = "".join(
        u["content"]["text"] for u in updates if u["sessionUpdate"] == "agent_message_chunk"
    )
    assert _REPLY in streamed

    # 正文 chunk 必须排在工具调用之后：编辑器按到达顺序渲染
    assert kinds.index("tool_call") < len(kinds) - 1


def test_acp_turn_reports_tool_call_pending_then_in_progress_then_completed(
    tmp_path: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    """长工具跑起来之前编辑器里也得有东西可看：三段状态按序到达，且是同一个 toolCallId。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hello.txt").write_text(_FILE_BODY, encoding="utf-8")

    updates: list[dict[str, Any]] = []
    run_acp_turn(
        AcpTurnRequest(
            workspace=workspace,
            thread_id="cli-acptc1",
            message="读一下 hello.txt",
            tool_call_prefix="t1_",
        ),
        send_update=updates.append,
        cancel_check=lambda: False,
    )

    tool_updates = [
        u for u in updates if u["sessionUpdate"] in ("tool_call", "tool_call_update")
    ]
    assert [u["status"] for u in tool_updates] == ["pending", "in_progress", "completed"]
    # 三段必须落在同一行上，否则编辑器里会画出三条各自独立的工具调用
    assert len({u["toolCallId"] for u in tool_updates}) == 1
    assert tool_updates[0]["toolCallId"].startswith("t1_")
    # 新建那条是 tool_call，后两段是对它的更新
    assert [u["sessionUpdate"] for u in tool_updates] == [
        "tool_call",
        "tool_call_update",
        "tool_call_update",
    ]
    # pending 那条就带好标题与 kind：编辑器不用等跑完才知道这是在读哪个文件
    assert tool_updates[0]["title"] == "执行 read_file(hello.txt)"
    assert tool_updates[0]["kind"] == "read"
    # 状态推进排在正文之前
    first_text = next(
        i for i, u in enumerate(updates) if u["sessionUpdate"] == "agent_message_chunk"
    )
    assert updates.index(tool_updates[-1]) < first_text


def test_acp_turn_is_read_only_by_default(
    tmp_path: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    """默认只读：不带 --write 时工具集里不该出现写工具。"""
    from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER
    from llgraph.core.session_bootstrap import AgentRuntimeBundle

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hello.txt").write_text(_FILE_BODY, encoding="utf-8")

    run_acp_turn(
        AcpTurnRequest(
            workspace=workspace,
            thread_id="cli-acpe2e2",
            message="读一下 hello.txt",
        ),
        send_update=lambda _u: None,
        cancel_check=lambda: False,
    )
    assert RUNTIME_MANAGER.get(workspace).allow_write is False
    assert AgentRuntimeBundle is not None

    tool_names: set[str] = set()
    for body in stub_gateway.requests:
        for spec in body.get("tools") or []:
            name = spec.get("name")
            if name:
                tool_names.add(name)
    assert "read_file" in tool_names
    assert "write_file" not in tool_names
    assert "search_replace" not in tool_names


def test_loaded_session_sends_old_history_to_the_model(
    tmp_path: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    """续聊的实质：编辑器重启后接回来的那一轮，模型必须还看得见之前说过的话。"""
    from langchain_core.messages import AIMessage, HumanMessage

    from llgraph.editor.acp.replay import load_session_updates
    from llgraph.session.session_file_store import save_session_messages

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hello.txt").write_text(_FILE_BODY, encoding="utf-8")
    thread_id = "cli-acpload1"
    save_session_messages(
        workspace,
        thread_id,
        [
            HumanMessage(content="记住暗号是 菠萝蜜"),
            AIMessage(content="好的，记住了。"),
        ],
    )

    # 编辑器侧先拿回历史
    updates = load_session_updates(workspace, thread_id)
    assert any(
        u["sessionUpdate"] == "user_message_chunk" and "菠萝蜜" in u["content"]["text"]
        for u in updates
    ), updates

    run_acp_turn(
        AcpTurnRequest(
            workspace=workspace,
            thread_id=thread_id,
            message="暗号是什么？",
        ),
        send_update=lambda _u: None,
        cancel_check=lambda: False,
    )

    sent = json.dumps(stub_gateway.requests, ensure_ascii=False)
    assert "菠萝蜜" in sent


def _edit_turn(
    workspace: Path,
    thread_id: str,
    decision: ApprovalDecision,
) -> tuple[list[ApprovalRequest], Any]:
    """跑一轮「模型要改代码」，闸门按给定决定应答。"""
    asked: list[ApprovalRequest] = []

    def ask(req: ApprovalRequest) -> ApprovalDecision:
        asked.append(req)
        return decision

    result = run_acp_turn(
        AcpTurnRequest(
            workspace=workspace,
            thread_id=thread_id,
            message="把 return 1 改成 return 2",
            allow_write=True,
            permission_ask=ask,
        ),
        send_update=lambda _u: None,
        cancel_check=lambda: False,
    )
    return asked, result


@pytest.fixture
def edit_workspace(tmp_path: Path, stub_gateway: _StubState) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    stub_gateway.tool_name = "search_replace"
    stub_gateway.tool_input = {
        "path": "app.py",
        "old_string": "return 1",
        "new_string": "return 2",
    }
    return workspace


def test_acp_turn_writes_after_permission_granted(
    edit_workspace: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    """允许之后这一刀真落地——「编辑器里能改代码」就是这一条。"""
    asked, result = _edit_turn(
        edit_workspace, "cli-acpw1", ApprovalDecision(allowed=True)
    )

    assert result.stop_reason == "end_turn"
    assert len(asked) == 1
    assert asked[0].tool == "search_replace"
    assert asked[0].path == "app.py"
    assert "return 1" in (asked[0].old_text or "")
    assert "return 2" in (asked[0].new_text or "")
    assert (edit_workspace / "app.py").read_text(encoding="utf-8") == (
        "def run():\n    return 2\n"
    )


class _FakeEditorFiles:
    """假编辑器的未保存缓冲区（真桥的协议细节在 test_acp_fs_bridge.py 里测）。"""

    def __init__(self, buffers: dict[Path, str]) -> None:
        self.buffers = dict(buffers)
        self.writes: list[tuple[Path, str]] = []

    def read_text(self, path: Path) -> str | None:
        return self.buffers.get(Path(path))

    def write_text(self, path: Path, text: str) -> bool:
        self.writes.append((Path(path), text))
        self.buffers[Path(path)] = text
        return True


def test_acp_turn_reads_the_unsaved_buffer(
    tmp_path: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    """编辑器里改了还没保存：模型该看到缓冲区那份，不是磁盘那份。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "hello.txt"
    target.write_text(_FILE_BODY, encoding="utf-8")
    editor = _FakeEditorFiles({target: "用户刚打的这行还没保存\n"})

    updates: list[dict[str, Any]] = []
    run_acp_turn(
        AcpTurnRequest(
            workspace=workspace,
            thread_id="cli-acpfs1",
            message="读一下 hello.txt",
            editor_files=editor,
        ),
        send_update=updates.append,
        cancel_check=lambda: False,
    )

    tool_call = _completed_tool_call(updates)
    shown = tool_call["content"][0]["content"]["text"]
    assert "还没保存" in shown
    assert "hello from llgraph" not in shown
    # 模型拿到的工具结果同样是缓冲区那份
    sent = json.dumps(stub_gateway.requests, ensure_ascii=False)
    assert "还没保存" in sent


def test_acp_turn_edit_of_a_dirty_file_goes_back_to_the_editor(
    edit_workspace: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    """有未保存改动的文件：这一刀交给编辑器写，别让用户一按保存就盖掉。"""
    target = edit_workspace / "app.py"
    buffered = "def run():\n    return 1  # 还没保存\n"
    editor = _FakeEditorFiles({target: buffered})

    result = run_acp_turn(
        AcpTurnRequest(
            workspace=edit_workspace,
            thread_id="cli-acpfs2",
            message="把 return 1 改成 return 2",
            allow_write=True,
            editor_files=editor,
        ),
        send_update=lambda _u: None,
        cancel_check=lambda: False,
    )

    assert result.stop_reason == "end_turn"
    assert len(editor.writes) == 1
    written = editor.writes[0][1]
    assert written == "def run():\n    return 2  # 还没保存\n"
    assert target.read_text(encoding="utf-8") == "def run():\n    return 1\n"


def test_acp_turn_rejection_keeps_file_and_tells_the_model(
    edit_workspace: Path,
    stub_gateway: _StubState,
    clean_runtime: None,
) -> None:
    asked, result = _edit_turn(
        edit_workspace,
        "cli-acpw2",
        ApprovalDecision(allowed=False, reason="用户在编辑器里选择了拒绝"),
    )

    assert len(asked) == 1
    assert result.stop_reason == "end_turn"
    assert (edit_workspace / "app.py").read_text(encoding="utf-8") == (
        "def run():\n    return 1\n"
    )
    # 模型必须知道这次没改成：否则下一轮会以为改过了
    assert any("拒绝" in text for text in stub_gateway.tool_results())
