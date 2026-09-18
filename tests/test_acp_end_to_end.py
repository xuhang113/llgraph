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

_FAKE_KEY = "test-key-not-a-secret"
_FILE_BODY = "hello from llgraph\n"
_REPLY = "hello.txt 里只有一行问候。"


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _tool_use_stream() -> bytes:
    """第一轮：模型要求读文件。"""
    args = json.dumps({"path": "hello.txt"})
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
                        "name": "read_file",
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
                payload = _text_stream() if wants_reply else _tool_use_stream()
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
                        "name": "read_file",
                        "input": {"path": "hello.txt"},
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

    tool_call = next(u for u in updates if u["sessionUpdate"] == "tool_call")
    assert tool_call["kind"] == "read"
    assert "read_file" in tool_call["title"]
    assert "hello from llgraph" in tool_call["content"][0]["content"]["text"]

    streamed = "".join(
        u["content"]["text"] for u in updates if u["sessionUpdate"] == "agent_message_chunk"
    )
    assert _REPLY in streamed

    # 正文 chunk 必须排在工具调用之后：编辑器按到达顺序渲染
    assert kinds.index("tool_call") < len(kinds) - 1


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
