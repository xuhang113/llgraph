"""ACP（Agent Client Protocol）服务端：握手、开会话、一轮提问、停止。

不打真模型：turn_runner 用桩件注入，测的是协议这一层——
编辑器发什么、我们回什么、过程事件按什么顺序推出去。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from llgraph.editor.acp.jsonrpc import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcConnection,
)
from llgraph.editor.acp.server import AcpServer
from llgraph.editor.acp.sink import AcpTraceSink
from llgraph.editor.acp.turn import AcpTurnRequest, AcpTurnResult
from llgraph.editor.acp.updates import (
    ACP_PROTOCOL_VERSION,
    acp_tool_kind,
    prompt_text,
    tool_call_from_step,
    tool_name_from_title,
)


class _LineWriter:
    """收集服务端写出的 ndjson（工作线程与读循环都会写，要加锁）。"""

    def __init__(self) -> None:
        self._buf = ""
        self.messages: list[dict[str, Any]] = []
        self._cond = threading.Condition()

    def write(self, text: str) -> int:
        with self._cond:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self.messages.append(json.loads(line))
            self._cond.notify_all()
        return len(text)

    def flush(self) -> None:
        pass

    def wait_for(
        self,
        predicate: Any,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """@param predicate 消息判定 @param timeout 超时秒 @return 命中的消息"""
        with self._cond:
            found = self._cond.wait_for(
                lambda: any(predicate(m) for m in self.messages),
                timeout=timeout,
            )
            if not found:
                raise AssertionError(f"超时未等到目标消息，已收到: {self.messages}")
            return next(m for m in self.messages if predicate(m))


class _Harness:
    """把 AcpServer 跑在真 pipe 上，测试侧扮演编辑器。"""

    def __init__(self, **server_kwargs: Any) -> None:
        read_fd, write_fd = os.pipe()
        self._stdin = os.fdopen(read_fd, "r", encoding="utf-8")
        self._client = os.fdopen(write_fd, "w", encoding="utf-8")
        self.out = _LineWriter()
        self.server = AcpServer(
            JsonRpcConnection(self._stdin, self.out),
            **server_kwargs,
        )
        self._thread = threading.Thread(target=self.server.serve, daemon=True)
        self._thread.start()

    def send(self, method: str, params: Any = None, request_id: Any = None) -> None:
        """@param method 方法 @param params 参数 @param request_id 请求 id（None 为通知）"""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if request_id is not None:
            payload["id"] = request_id
        if params is not None:
            payload["params"] = params
        self.send_raw(json.dumps(payload))

    def send_raw(self, line: str) -> None:
        """@param line 原始一行（测畸形输入用）"""
        self._client.write(line + "\n")
        self._client.flush()

    def response(self, request_id: Any, timeout: float = 5.0) -> dict[str, Any]:
        """@param request_id 请求 id @param timeout 超时秒 @return 回包"""
        return self.out.wait_for(lambda m: m.get("id") == request_id, timeout=timeout)

    def updates(self) -> list[dict[str, Any]]:
        """@return 已收到的 session/update 的 update 字段"""
        return [
            m["params"]["update"]
            for m in list(self.out.messages)
            if m.get("method") == "session/update"
        ]

    def close(self) -> None:
        """关掉客户端写端，读循环随之退出。"""
        self._client.close()
        self._thread.join(timeout=5)
        self._stdin.close()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    return tmp_path


def _new_session(h: _Harness, workspace: Path) -> str:
    h.send("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, request_id=1)
    h.response(1)
    h.send("session/new", {"cwd": str(workspace), "mcpServers": []}, request_id=2)
    return h.response(2)["result"]["sessionId"]


def test_initialize_returns_protocol_version_and_capabilities(workspace: Path) -> None:
    h = _Harness()
    try:
        h.send("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, request_id=1)
        result = h.response(1)["result"]
        assert result["protocolVersion"] == ACP_PROTOCOL_VERSION
        assert result["agentCapabilities"]["promptCapabilities"]["embeddedContext"] is True
        assert result["authMethods"] == []
    finally:
        h.close()


def test_initialize_negotiates_down_to_client_version() -> None:
    """客户端只会更老的版本时，回它那个版本，而不是把我们的顶上去。"""
    h = _Harness()
    try:
        h.send("initialize", {"protocolVersion": 0}, request_id=1)
        assert h.response(1)["result"]["protocolVersion"] == 0
    finally:
        h.close()


def test_session_new_uses_cwd_and_returns_session_id(workspace: Path) -> None:
    h = _Harness()
    try:
        session_id = _new_session(h, workspace)
        assert session_id.startswith("cli-")
        assert h.server._sessions[session_id].workspace == workspace.resolve()
    finally:
        h.close()


def test_session_new_rejects_relative_cwd() -> None:
    h = _Harness()
    try:
        h.send("session/new", {"cwd": "relative/dir"}, request_id=1)
        assert h.response(1)["error"]["code"] == INVALID_PARAMS
    finally:
        h.close()


def test_session_new_falls_back_to_default_workspace(workspace: Path) -> None:
    """编辑器没给 cwd 时用 ``llgraph acp -C`` 指定的工作区。"""
    h = _Harness(default_workspace=workspace)
    try:
        h.send("session/new", {}, request_id=1)
        assert h.response(1)["result"]["sessionId"].startswith("cli-")
    finally:
        h.close()


def test_prompt_streams_updates_then_responds_end_turn(workspace: Path) -> None:
    seen: dict[str, Any] = {}

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        seen["message"] = req.message
        seen["workspace"] = req.workspace
        seen["allow_write"] = req.allow_write
        send_update({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "想想"}})
        send_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_1",
                "title": "执行 read_file(hello.txt)",
                "kind": "read",
                "status": "completed",
            }
        )
        send_update({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "读到了"}})
        return AcpTurnResult(text="读到了", stop_reason="end_turn")

    h = _Harness(turn_runner=runner)
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "看看 hello.txt"}]},
            request_id=3,
        )
        response = h.response(3)
        assert response["result"]["stopReason"] == "end_turn"
        assert seen["message"] == "看看 hello.txt"
        assert seen["workspace"] == workspace.resolve()
        assert seen["allow_write"] is False

        kinds = [u["sessionUpdate"] for u in h.updates()]
        assert kinds == ["agent_thought_chunk", "tool_call", "agent_message_chunk"]
        # 过程事件必须在回包之前送达，否则编辑器会先收尾再补内容
        order = [m.get("method") or f"response:{m.get('id')}" for m in h.out.messages]
        assert order.index("session/update") < order.index("response:3")
    finally:
        h.close()


def test_prompt_passes_write_mode_through(workspace: Path) -> None:
    seen: dict[str, Any] = {}

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        seen["allow_write"] = req.allow_write
        return AcpTurnResult(text="ok")

    h = _Harness(allow_write=True, turn_runner=runner)
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "改一下"}]},
            request_id=3,
        )
        h.response(3)
        assert seen["allow_write"] is True
    finally:
        h.close()


def test_cancel_during_prompt_stops_turn(workspace: Path) -> None:
    """跑一轮的同时还要能收 session/cancel：读循环不能被 prompt 占住。"""
    started = threading.Event()

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        started.set()
        for _ in range(500):
            if cancel_check():
                return AcpTurnResult(text="", stop_reason="cancelled")
            threading.Event().wait(0.01)
        return AcpTurnResult(text="没被停下", stop_reason="end_turn")

    h = _Harness(turn_runner=runner)
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "跑个长活"}]},
            request_id=3,
        )
        assert started.wait(timeout=5)
        h.send("session/cancel", {"sessionId": session_id})
        assert h.response(3)["result"]["stopReason"] == "cancelled"
    finally:
        h.close()


def test_second_prompt_while_running_is_rejected(workspace: Path) -> None:
    release = threading.Event()

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        release.wait(timeout=5)
        return AcpTurnResult(text="ok")

    h = _Harness(turn_runner=runner)
    try:
        session_id = _new_session(h, workspace)
        prompt = {"sessionId": session_id, "prompt": [{"type": "text", "text": "第一轮"}]}
        h.send("session/prompt", prompt, request_id=3)
        h.send("session/prompt", {**prompt, "prompt": [{"type": "text", "text": "第二轮"}]}, request_id=4)
        assert "error" in h.response(4)
        release.set()
        assert h.response(3)["result"]["stopReason"] == "end_turn"
    finally:
        h.close()


def test_prompt_failure_returns_error_not_silence(workspace: Path) -> None:
    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        raise RuntimeError("模型入口没配")

    h = _Harness(turn_runner=runner)
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "你好"}]},
            request_id=3,
        )
        assert "模型入口没配" in h.response(3)["error"]["message"]
        # 出错后会话要能继续用，不能一直卡在 busy
        assert h.server._sessions[session_id].busy is False
    finally:
        h.close()


def test_unknown_session_and_method_and_bad_json(workspace: Path) -> None:
    h = _Harness()
    try:
        h.send("session/prompt", {"sessionId": "cli-nope", "prompt": []}, request_id=1)
        assert h.response(1)["error"]["code"] == INVALID_PARAMS

        h.send("session/whatever", {}, request_id=2)
        assert h.response(2)["error"]["code"] == METHOD_NOT_FOUND

        h.send_raw("{ 这不是 JSON")
        assert h.out.wait_for(lambda m: m.get("error", {}).get("code") == PARSE_ERROR)

        # 畸形消息之后连接仍可用
        h.send("initialize", {"protocolVersion": 1}, request_id=3)
        assert h.response(3)["result"]["protocolVersion"] == ACP_PROTOCOL_VERSION
    finally:
        h.close()


def test_notification_never_gets_a_response(workspace: Path) -> None:
    h = _Harness()
    try:
        h.send("session/cancel", {"sessionId": "cli-nope"})
        h.send("initialize", {"protocolVersion": 1}, request_id=9)
        h.response(9)
        assert all("id" not in m or m["id"] == 9 for m in h.out.messages)
    finally:
        h.close()


# ---- prompt / trace 映射 ----


def test_prompt_text_flattens_content_blocks() -> None:
    text = prompt_text(
        [
            {"type": "text", "text": "看看这个"},
            {"type": "resource_link", "uri": "file:///tmp/a.py", "name": "a.py"},
            {
                "type": "resource",
                "resource": {"uri": "file:///tmp/b.py", "text": "print(1)"},
            },
            {"type": "image", "data": "..."},
        ]
    )
    assert "看看这个" in text
    assert "a.py" in text
    assert "print(1)" in text


def test_tool_title_and_kind_mapping() -> None:
    assert tool_name_from_title("执行 read_file(hello.txt)") == "read_file"
    assert tool_name_from_title("执行 grep") == "grep"
    assert acp_tool_kind("read_file") == "read"
    assert acp_tool_kind("search_replace") == "edit"
    assert acp_tool_kind("run_terminal_cmd") == "execute"
    # MCP 工具名不在表里，按动词位判（Server 前缀后的那个词也算动词位）
    assert acp_tool_kind("github_list_issues") == "read"
    assert acp_tool_kind("createOrUpdateFile") == "edit"
    assert acp_tool_kind("slack_post_message") == "edit"
    assert acp_tool_kind("odd_vendor_thing") == "other"


def test_tool_call_from_step_only_for_tool_steps() -> None:
    step = {
        "kind": "tool",
        "title": "执行 read_file(hello.txt)",
        "body_lines": ["line1", "line2"],
    }
    update = tool_call_from_step(step, tool_call_id="call_1")
    assert update is not None
    assert update["sessionUpdate"] == "tool_call"
    assert update["kind"] == "read"
    assert update["status"] == "completed"
    assert "line1" in update["content"][0]["content"]["text"]

    assert tool_call_from_step({"kind": "reply", "title": "回复"}, tool_call_id="x") is None


def test_tool_call_content_is_truncated() -> None:
    step = {
        "kind": "tool",
        "title": "执行 grep(x)",
        "body_lines": [f"line{i}" for i in range(200)],
    }
    update = tool_call_from_step(step, tool_call_id="call_1", max_content_lines=10)
    text = update["content"][0]["content"]["text"]
    assert text.count("\n") == 10
    assert "还有 190 行" in text


def test_sink_streams_text_and_thinking_delta() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.line("[trace] 某一行")
    sink.stream("你")
    sink.stream("好")
    sink.stream_end()
    # trace 的 thinking 每次给全文，只能把新增部分发出去
    sink.thinking_update("想了一半")
    sink.thinking_update("想了一半又想了后半")

    kinds = [u["sessionUpdate"] for u in updates]
    assert kinds == [
        "agent_message_chunk",
        "agent_message_chunk",
        "agent_thought_chunk",
        "agent_thought_chunk",
    ]
    assert [u["content"]["text"] for u in updates] == ["你", "好", "想了一半", "又想了后半"]
    assert sink.streamed_chars == 2
    assert sink.log_lines == ["[trace] 某一行"]


def test_sink_tool_steps_get_unique_ids() -> None:
    from llgraph.display.trace_display import TraceStepRecord

    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)
    for i in (1, 2):
        sink.step_added(
            TraceStepRecord(
                step_id=i,
                kind="tool",
                title=f"执行 read_file(f{i}.txt)",
                elapsed=0.1,
                summary="1 行输出",
                body_lines=["ok"],
            )
        )
    sink.step_added(
        TraceStepRecord(
            step_id=3,
            kind="reply",
            title="回复",
            elapsed=0.1,
            summary="",
        )
    )
    ids = [u["toolCallId"] for u in updates]
    assert ids == ["call_1", "call_2"]
