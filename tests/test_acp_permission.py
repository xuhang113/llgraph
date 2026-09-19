"""写入授权：工具层闸门 + ACP 的 ``session/request_permission`` 弹窗。

分三层测：
1. `permissions/approval.py` 的闸门语义（没闸门一律放行，CLI / Web 行为不变）
2. 写工具 / shell 工具真的会在落盘前问闸门，被拒时一个字节都不落
3. ACP 侧：授权请求的载荷、四个选项的处理、等待期间被取消、编辑器断线
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from llgraph.config.shell_settings import ShellSettings
from llgraph.context.runtime_context import set_active_thread_id
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.shell_jobs import reset_shell_runtime_for_tests
from llgraph.core.shell_tools import create_shell_tools
from llgraph.core.workspace import WorkspaceContext
from llgraph.editor.acp.jsonrpc import RequestCancelled, RequestFailed
from llgraph.editor.acp.permission import AcpPermissionGate
from llgraph.editor.acp.turn import AcpTurnRequest, AcpTurnResult
from llgraph.permissions.approval import (
    APPROVAL_KIND_EDIT,
    APPROVAL_KIND_EXECUTE,
    ApprovalDecision,
    ApprovalRequest,
    current_approval_gate,
    request_approval,
    use_approval_gate,
)

from tests.test_acp_server import _Harness, _new_session

_ORIGINAL = "def run():\n    return 1\n"


def _fs_tool(root: Path, name: str):
    ctx = WorkspaceContext(root, allow_write=True)
    return next(t for t in create_filesystem_tools(ctx) if t.name == name)


def _shell_tool(root: Path):
    ctx = WorkspaceContext(root, allow_write=True)
    settings = ShellSettings(
        enabled=True,
        timeout_sec=5.0,
        background_timeout_sec=20.0,
        max_output_chars=4000,
        max_jobs=4,
        terminal_log_dir=".llgraph/context/terminals",
        log_commands=False,
    )
    tools = create_shell_tools(ctx, allow_write=True, settings=settings)
    return next(t for t in tools if t.name == "run_shell_command")


def _allow(_req: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(allowed=True)


def _reject(_req: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(allowed=False, reason="测试拒绝")


@pytest.fixture(autouse=True)
def _reset_shell_runtime():
    reset_shell_runtime_for_tests()
    set_active_thread_id("test-acp-permission")
    yield
    reset_shell_runtime_for_tests()
    set_active_thread_id(None)


# ---- 闸门语义 ----


def test_no_gate_means_allow() -> None:
    """CLI / Web 不登记闸门：写入不能因为本轮改造变成「要授权」。"""
    assert current_approval_gate() is None
    assert request_approval(ApprovalRequest(tool="write_file", path="a.py")).allowed


def test_gate_is_scoped_to_the_with_block() -> None:
    with use_approval_gate(_reject):
        assert request_approval(ApprovalRequest(tool="write_file")).allowed is False
    assert request_approval(ApprovalRequest(tool="write_file")).allowed


def test_gate_exception_counts_as_deny() -> None:
    """授权链路自己炸了也不能默认放行。"""

    def boom(_req: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError("编辑器没了")

    with use_approval_gate(boom):
        decision = request_approval(ApprovalRequest(tool="write_file"))
    assert decision.allowed is False
    assert "编辑器没了" in decision.reason


def test_gate_garbage_reply_counts_as_deny() -> None:
    with use_approval_gate(lambda _req: "当然可以"):  # type: ignore[arg-type,return-value]
        assert request_approval(ApprovalRequest(tool="write_file")).allowed is False


# ---- 写工具 ----


def test_rejected_write_file_leaves_nothing_behind(tmp_path: Path) -> None:
    tool = _fs_tool(tmp_path, "write_file")
    with use_approval_gate(_reject):
        out = tool.invoke({"path": "new.py", "content": "print(1)\n"})
    assert "拒绝" in out
    assert "测试拒绝" in out
    assert not (tmp_path / "new.py").exists()


def test_rejected_search_replace_keeps_old_text(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL, encoding="utf-8")
    tool = _fs_tool(tmp_path, "search_replace")
    with use_approval_gate(_reject):
        out = tool.invoke(
            {"path": "app.py", "old_string": "return 1", "new_string": "return 2"}
        )
    assert "拒绝" in out
    assert target.read_text(encoding="utf-8") == _ORIGINAL


def test_rejected_append_file_keeps_old_text(tmp_path: Path) -> None:
    target = tmp_path / "log.md"
    target.write_text("# 标题\n", encoding="utf-8")
    tool = _fs_tool(tmp_path, "append_file")
    with use_approval_gate(_reject):
        tool.invoke({"path": "log.md", "content": "追加\n"})
    assert target.read_text(encoding="utf-8") == "# 标题\n"


def test_approved_write_still_lands(tmp_path: Path) -> None:
    tool = _fs_tool(tmp_path, "write_file")
    with use_approval_gate(_allow):
        out = tool.invoke({"path": "new.py", "content": "print(1)\n"})
    assert "已写入" in out
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print(1)\n"


def test_gate_gets_path_and_both_sides_of_the_diff(tmp_path: Path) -> None:
    """编辑器要渲染改动预览，闸门必须拿到改前 / 改后全文。"""
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL, encoding="utf-8")
    seen: list[ApprovalRequest] = []

    def record(req: ApprovalRequest) -> ApprovalDecision:
        seen.append(req)
        return ApprovalDecision(allowed=True)

    tool = _fs_tool(tmp_path, "search_replace")
    with use_approval_gate(record):
        tool.invoke(
            {"path": "app.py", "old_string": "return 1", "new_string": "return 2"}
        )

    assert len(seen) == 1
    req = seen[0]
    assert req.tool == "search_replace"
    assert req.kind == APPROVAL_KIND_EDIT
    assert req.path == "app.py"
    assert req.old_text == _ORIGINAL
    assert "return 2" in (req.new_text or "")
    assert req.title() == "执行 search_replace(app.py)"


def test_write_failures_before_hint_not_polluted_by_rejection(tmp_path: Path) -> None:
    """按了拒绝不是「写法不对」，不该触发下一轮的分块重试提示。"""
    from llgraph.context.context_session import ContextSession
    from llgraph.core.write_failure_tracker import WriteFailureTracker

    from langchain_core.messages import ToolMessage

    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tracker = WriteFailureTracker(ContextSession(), failures_before_hint=1)
    tool = next(
        t
        for t in create_filesystem_tools(ctx, write_failure_tracker=tracker)
        if t.name == "write_file"
    )
    with use_approval_gate(_reject):
        out = tool.invoke({"path": "a.py", "content": "x = 1\n"})
    # 计数器实际是扫 ToolMessage 文本的，所以拒绝语也不能撞上它的失败特征
    tracker.inspect_tool_messages([ToolMessage(content=out, name="write_file", tool_call_id="1")])
    assert tracker.consume_hint_for_context() == ""


# ---- shell ----


def test_rejected_shell_command_does_not_run(tmp_path: Path) -> None:
    tool = _shell_tool(tmp_path)
    script = f"{sys.executable} -c 'open(\"touched.txt\", \"w\").write(\"x\")'"
    with use_approval_gate(_reject):
        out = tool.invoke({"command": script})
    assert "拒绝" in out
    assert not (tmp_path / "touched.txt").exists()


def test_shell_gate_sees_command_and_cwd(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    seen: list[ApprovalRequest] = []

    def record(req: ApprovalRequest) -> ApprovalDecision:
        seen.append(req)
        return ApprovalDecision(allowed=False)

    tool = _shell_tool(tmp_path)
    with use_approval_gate(record):
        tool.invoke({"command": "cd pkg && echo hi"})

    assert len(seen) == 1
    assert seen[0].kind == APPROVAL_KIND_EXECUTE
    assert seen[0].command == "echo hi"
    assert seen[0].cwd == "pkg"


def test_read_only_shell_is_not_asked(tmp_path: Path) -> None:
    """只读模式能过策略闸门的命令本身不改工作区，不该再弹框。"""
    ctx = WorkspaceContext(tmp_path, allow_write=False)
    tool = next(
        t
        for t in create_shell_tools(ctx, allow_write=False)
        if t.name == "run_shell_command"
    )
    asked: list[ApprovalRequest] = []

    def record(req: ApprovalRequest) -> ApprovalDecision:
        asked.append(req)
        return ApprovalDecision(allowed=False)

    with use_approval_gate(record):
        out = tool.invoke({"command": "echo hi"})
    assert asked == []
    assert "hi" in out


# ---- ACP 授权弹窗 ----


class _FakeConn:
    """只实现 ``request``：按脚本回 result 或抛异常。"""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def request(self, method: str, params: dict[str, Any], **_kwargs: Any) -> Any:
        self.sent.append((method, params))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _selected(option_id: str) -> dict[str, Any]:
    return {"outcome": {"outcome": "selected", "optionId": option_id}}


def _gate(replies: list[Any], workspace: Path) -> tuple[AcpPermissionGate, _FakeConn]:
    conn = _FakeConn(replies)
    gate = AcpPermissionGate(
        conn,  # type: ignore[arg-type]
        "cli-perm",
        workspace=workspace,
        timeout_sec=1.0,
    )
    return gate, conn


def test_permission_payload_carries_diff_and_options(tmp_path: Path) -> None:
    gate, conn = _gate([_selected("allow_once")], tmp_path)
    decision = gate.ask(
        ApprovalRequest(
            tool="search_replace",
            path="app.py",
            old_text="a\n",
            new_text="b\n",
        )
    )
    assert decision.allowed is True

    method, params = conn.sent[0]
    assert method == "session/request_permission"
    assert params["sessionId"] == "cli-perm"
    assert [o["kind"] for o in params["options"]] == [
        "allow_once",
        "allow_always",
        "reject_once",
        "reject_always",
    ]
    call = params["toolCall"]
    assert call["kind"] == "edit"
    assert call["status"] == "pending"
    assert call["locations"][0]["path"] == str(tmp_path / "app.py")
    diff = call["content"][0]
    assert diff["type"] == "diff"
    assert diff["oldText"] == "a\n"
    assert diff["newText"] == "b\n"


def test_permission_payload_for_shell_is_the_command(tmp_path: Path) -> None:
    gate, conn = _gate([_selected("reject_once")], tmp_path)
    decision = gate.ask(
        ApprovalRequest(
            tool="run_shell_command",
            kind=APPROVAL_KIND_EXECUTE,
            command="pytest -q",
        )
    )
    assert decision.allowed is False
    call = conn.sent[0][1]["toolCall"]
    assert call["kind"] == "execute"
    assert "locations" not in call
    assert call["content"][0]["content"]["text"] == "pytest -q"


def test_allow_always_stops_asking_for_that_kind(tmp_path: Path) -> None:
    gate, conn = _gate([_selected("allow_always")], tmp_path)
    edit = ApprovalRequest(tool="write_file", path="a.py", new_text="x")
    assert gate.ask(edit).allowed is True
    assert gate.ask(edit).allowed is True
    assert len(conn.sent) == 1

    # 记住的是「这一类」，shell 仍要单独问
    gate._replies = [_selected("reject_once")]  # type: ignore[attr-defined]
    conn._replies = [_selected("reject_once")]
    shell = ApprovalRequest(
        tool="run_shell_command", kind=APPROVAL_KIND_EXECUTE, command="rm -rf build"
    )
    assert gate.ask(shell).allowed is False
    assert len(conn.sent) == 2


def test_reject_always_keeps_rejecting_without_asking(tmp_path: Path) -> None:
    gate, conn = _gate([_selected("reject_always")], tmp_path)
    edit = ApprovalRequest(tool="write_file", path="a.py", new_text="x")
    assert gate.ask(edit).allowed is False
    second = gate.ask(edit)
    assert second.allowed is False
    assert "都拒绝" in second.reason
    assert len(conn.sent) == 1


def test_cancelled_outcome_and_cancelled_wait(tmp_path: Path) -> None:
    gate, _conn = _gate([{"outcome": {"outcome": "cancelled"}}], tmp_path)
    decision = gate.ask(ApprovalRequest(tool="write_file", path="a.py"))
    assert decision.cancelled is True
    assert decision.allowed is False

    gate2, _c2 = _gate([RequestCancelled("停止")], tmp_path)
    assert gate2.ask(ApprovalRequest(tool="write_file", path="a.py")).cancelled is True


def test_editor_failure_or_garbage_outcome_denies(tmp_path: Path) -> None:
    gate, _conn = _gate([RequestFailed("超时")], tmp_path)
    decision = gate.ask(ApprovalRequest(tool="write_file", path="a.py"))
    assert decision.allowed is False
    assert decision.cancelled is False
    assert "超时" in decision.reason

    gate2, _c2 = _gate([{"outcome": {"outcome": "选了个奇怪的"}}], tmp_path)
    assert gate2.ask(ApprovalRequest(tool="write_file", path="a.py")).allowed is False


def test_gate_short_circuits_when_turn_already_cancelled(tmp_path: Path) -> None:
    conn = _FakeConn([])
    gate = AcpPermissionGate(
        conn,  # type: ignore[arg-type]
        "cli-perm",
        workspace=tmp_path,
        cancel_check=lambda: True,
    )
    assert gate.ask(ApprovalRequest(tool="write_file", path="a.py")).cancelled is True
    assert conn.sent == []


# ---- 协议层：编辑器与 Agent 之间真走一趟 ----


def _ask_runner(asked: list[dict[str, Any]], answers: list[ApprovalDecision]):
    """turn_runner 桩件：跑一轮里问一次授权，把决定写进 result.text。"""

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        assert req.permission_ask is not None
        decision = req.permission_ask(
            ApprovalRequest(
                tool="write_file",
                path="app.py",
                old_text="a\n",
                new_text="b\n",
            )
        )
        answers.append(decision)
        asked.append({"allowed": decision.allowed, "cancelled": decision.cancelled})
        return AcpTurnResult(
            text="已改" if decision.allowed else "没改",
            stop_reason="cancelled" if decision.cancelled else "end_turn",
        )

    return runner


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(_ORIGINAL, encoding="utf-8")
    return tmp_path


def test_editor_gets_permission_request_and_answer_reaches_the_tool(
    workspace: Path,
) -> None:
    asked: list[dict[str, Any]] = []
    h = _Harness(
        allow_write=True,
        ask_permission=True,
        turn_runner=_ask_runner(asked, []),
    )
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "改一下"}]},
            request_id=3,
        )
        ask = h.out.wait_for(
            lambda m: m.get("method") == "session/request_permission"
        )
        assert ask["params"]["sessionId"] == session_id
        assert ask["params"]["toolCall"]["content"][0]["type"] == "diff"

        h.send_raw(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": ask["id"],
                    "result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}},
                }
            )
        )
        assert h.response(3)["result"]["stopReason"] == "end_turn"
        assert asked == [{"allowed": True, "cancelled": False}]
    finally:
        h.close()


def test_editor_rejection_stops_the_write_but_not_the_turn(workspace: Path) -> None:
    asked: list[dict[str, Any]] = []
    h = _Harness(allow_write=True, ask_permission=True, turn_runner=_ask_runner(asked, []))
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "改一下"}]},
            request_id=3,
        )
        ask = h.out.wait_for(lambda m: m.get("method") == "session/request_permission")
        h.send_raw(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": ask["id"],
                    "result": {"outcome": {"outcome": "selected", "optionId": "reject_once"}},
                }
            )
        )
        # 拒绝一次写入不等于停止对话：模型还要能解释它想干什么
        assert h.response(3)["result"]["stopReason"] == "end_turn"
        assert asked == [{"allowed": False, "cancelled": False}]
    finally:
        h.close()


def test_cancel_while_waiting_for_permission_releases_the_turn(workspace: Path) -> None:
    """人不点按钮直接按停止：等待中的授权请求必须被放掉，否则这一轮永远挂着。"""
    asked: list[dict[str, Any]] = []
    h = _Harness(allow_write=True, ask_permission=True, turn_runner=_ask_runner(asked, []))
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "改一下"}]},
            request_id=3,
        )
        h.out.wait_for(lambda m: m.get("method") == "session/request_permission")
        h.send("session/cancel", {"sessionId": session_id})
        assert h.response(3)["result"]["stopReason"] == "cancelled"
        assert asked == [{"allowed": False, "cancelled": True}]
    finally:
        h.close()


def test_editor_disconnect_while_waiting_denies_instead_of_hanging(
    workspace: Path,
) -> None:
    started = threading.Event()
    decisions: list[ApprovalDecision] = []

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        started.set()
        decisions.append(
            req.permission_ask(ApprovalRequest(tool="write_file", path="app.py", new_text="b"))
        )
        return AcpTurnResult(text="ok")

    h = _Harness(allow_write=True, ask_permission=True, turn_runner=runner)
    session_id = _new_session(h, workspace)
    h.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "改一下"}]},
        request_id=3,
    )
    assert started.wait(timeout=5)
    h.out.wait_for(lambda m: m.get("method") == "session/request_permission")
    h.close()

    for _ in range(100):
        if decisions:
            break
        threading.Event().wait(0.05)
    assert decisions and decisions[0].allowed is False
    assert "编辑器" in decisions[0].reason


def test_write_without_ask_permission_never_asks(workspace: Path) -> None:
    """``llgraph acp --write``（免确认）不该给编辑器发授权请求。"""
    seen: list[Any] = []

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        seen.append(req.permission_ask)
        return AcpTurnResult(text="ok")

    h = _Harness(allow_write=True, ask_permission=False, turn_runner=runner)
    try:
        session_id = _new_session(h, workspace)
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "改一下"}]},
            request_id=3,
        )
        h.response(3)
        assert seen == [None]
        assert all(
            m.get("method") != "session/request_permission" for m in h.out.messages
        )
    finally:
        h.close()
