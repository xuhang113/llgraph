"""编辑器终端：命令在编辑器里跑，输出边跑边看。

分五层测：
1. `core/shell_terminal.py` 的来源语义（没登记来源时 CLI / Web 行为不变）
2. 同形进程（`EditorTerminalProcess`）：等退出、取最终输出、杀、放
3. `run_shell_command` 真的把命令交出去，沙箱开着时不交
4. ACP 侧的桥：载荷、回包解析、失败熔断
5. 载荷与 Sink：终端块马上发，收尾那条重发它而不是再附一份文本
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from llgraph.config.shell_settings import ShellSettings
from llgraph.context.runtime_context import set_active_thread_id
from llgraph.core.shell_jobs import reset_shell_runtime_for_tests
from llgraph.core.shell_terminal import (
    EditorTerminalProcess,
    TerminalExit,
    TerminalSnapshot,
    current_editor_terminal_source,
    spawn_editor_terminal,
    terminal_output_byte_limit,
    use_editor_terminal_source,
)
from llgraph.core.shell_tools import create_shell_tools
from llgraph.core.tool_progress import (
    notify_terminal_created,
    use_current_tool_call,
    use_tool_terminal_observer,
)
from llgraph.core.workspace import WorkspaceContext
from llgraph.editor.acp.jsonrpc import RequestCancelled, RequestFailed
from llgraph.editor.acp.sink import AcpTraceSink
from llgraph.editor.acp.terminal_bridge import (
    AcpTerminalBridge,
    client_terminal_capability,
    parse_exit_status,
)
from llgraph.editor.acp.turn import AcpTurnRequest, AcpTurnResult
from llgraph.editor.acp.updates import tool_call_from_step

from tests.test_acp_server import _Harness

_EDITOR_OUTPUT = "来自编辑器终端的输出\n"


class _Fake:
    """假编辑器终端：命令不真跑，输出按构造时给的脚本回。"""

    def __init__(
        self,
        *,
        text: str = _EDITOR_OUTPUT,
        exit_code: int | None = 0,
        truncated: bool = False,
        terminal_id: str = "term-1",
        running: bool = False,
        answer_output: bool = True,
    ) -> None:
        self.created: list[dict[str, Any]] = []
        self.killed: list[str] = []
        self.released: list[str] = []
        self.output_calls = 0
        self.text = text
        self.truncated = truncated
        self.terminal_id = terminal_id
        self.answer_output = answer_output
        self.status = TerminalExit(exit_code=exit_code)
        # running=True 的终端要等 finish() / kill() 才退出
        self.exited = threading.Event()
        if not running:
            self.exited.set()

    def finish(self) -> None:
        self.exited.set()

    def create(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str,
        output_byte_limit: int,
    ) -> str | None:
        self.created.append(
            {
                "command": command,
                "args": list(args),
                "cwd": cwd,
                "output_byte_limit": output_byte_limit,
            }
        )
        return self.terminal_id

    def output(self, terminal_id: str) -> TerminalSnapshot | None:
        self.output_calls += 1
        if not self.answer_output:
            return None
        return TerminalSnapshot(
            output=self.text,
            truncated=self.truncated,
            exit=self.status if self.exited.is_set() else None,
        )

    def wait_for_exit(self, terminal_id: str) -> TerminalExit | None:
        self.exited.wait(5.0)
        return self.status

    def kill(self, terminal_id: str) -> None:
        self.killed.append(terminal_id)
        self.exited.set()

    def release(self, terminal_id: str) -> None:
        self.released.append(terminal_id)


@pytest.fixture(autouse=True)
def _reset_shell_runtime():
    reset_shell_runtime_for_tests()
    set_active_thread_id("test-acp-terminal")
    yield
    reset_shell_runtime_for_tests()
    set_active_thread_id(None)


def _settings(**overrides: object) -> ShellSettings:
    data: dict[str, object] = {
        "enabled": True,
        "timeout_sec": 5.0,
        "background_timeout_sec": 20.0,
        "max_output_chars": 4000,
        "max_jobs": 4,
        "terminal_log_dir": ".llgraph/context/terminals",
        "log_commands": False,
    }
    data.update(overrides)
    return ShellSettings(**data)  # type: ignore[arg-type]


def _shell_tool(root: Path, *, sandbox_policy: Any = None, **overrides: object):
    ctx = WorkspaceContext(root, allow_write=False, sandbox_policy=sandbox_policy)
    tools = create_shell_tools(ctx, allow_write=False, settings=_settings(**overrides))
    return next(t for t in tools if t.name == "run_shell_command")


# ---- 来源语义 ----


def test_no_source_means_local_process(tmp_path: Path) -> None:
    """CLI / Web 不登记来源：命令照旧本地起进程。"""
    assert current_editor_terminal_source() is None
    assert (
        spawn_editor_terminal(command="echo hi", cwd=tmp_path, output_byte_limit=1024)
        is None
    )


def test_source_is_scoped_to_the_with_block(tmp_path: Path) -> None:
    fake = _Fake()
    with use_editor_terminal_source(fake):
        assert current_editor_terminal_source() is fake
    assert current_editor_terminal_source() is None


def test_editor_that_cannot_open_a_terminal_falls_back(tmp_path: Path) -> None:
    class _Refuses:
        def create(self, **_kwargs: Any) -> str | None:
            return None

    with use_editor_terminal_source(_Refuses()):
        assert (
            spawn_editor_terminal(
                command="echo hi", cwd=tmp_path, output_byte_limit=1024
            )
            is None
        )


def test_editor_blowing_up_falls_back(tmp_path: Path) -> None:
    """编辑器侧炸了只能少一条执行路径，不能让工具报错。"""

    class _Boom:
        def create(self, **_kwargs: Any) -> str | None:
            raise RuntimeError("编辑器没了")

    with use_editor_terminal_source(_Boom()):
        assert (
            spawn_editor_terminal(
                command="echo hi", cwd=tmp_path, output_byte_limit=1024
            )
            is None
        )


def test_spawn_reports_the_terminal_to_the_entry(tmp_path: Path) -> None:
    """建好就报：入口据此把终端挂到当前那条 tool_call 上。"""
    seen: list[tuple[str, str]] = []
    fake = _Fake()
    with use_editor_terminal_source(fake), use_tool_terminal_observer(
        lambda cid, tid: seen.append((cid, tid))
    ), use_current_tool_call("toolu_7"):
        live = spawn_editor_terminal(
            command="echo hi", cwd=tmp_path, output_byte_limit=1024
        )
    assert live is not None
    assert seen == [("toolu_7", "term-1")]


def test_terminal_notice_without_a_current_call_is_dropped() -> None:
    seen: list[tuple[str, str]] = []
    with use_tool_terminal_observer(lambda cid, tid: seen.append((cid, tid))):
        notify_terminal_created("term-9")
        with use_current_tool_call("toolu_1"):
            notify_terminal_created("")
    assert seen == []


def test_terminal_observer_exception_does_not_escape() -> None:
    def _boom(_cid: str, _tid: str) -> None:
        raise RuntimeError("界面炸了")

    with use_tool_terminal_observer(_boom), use_current_tool_call("toolu_1"):
        notify_terminal_created("term-1")


def test_output_byte_limit_is_wider_than_the_model_budget() -> None:
    """编辑器里要能往上翻日志，但也不能无上限（跑完要整段过 stdio）。"""
    assert terminal_output_byte_limit(4_000) == 131_072
    assert terminal_output_byte_limit(100_000) == 400_000
    assert terminal_output_byte_limit(500_000) == 1_048_576
    assert terminal_output_byte_limit("脏数据") == 131_072  # type: ignore[arg-type]


# ---- 同形进程 ----


def _process(fake: _Fake, tmp_path: Path, **kwargs: Any) -> EditorTerminalProcess:
    return EditorTerminalProcess(
        fake,
        fake.terminal_id,
        command="echo hi",
        cwd=tmp_path,
        **kwargs,
    )


def test_finished_command_reports_code_and_output(tmp_path: Path) -> None:
    fake = _Fake(exit_code=3)
    live = _process(fake, tmp_path)
    assert live.wait(2.0) is True
    assert live.returncode() == 3
    assert live.snapshot_stdio() == (_EDITOR_OUTPUT, "")
    assert live.sandboxed is False
    assert live.elapsed_sec() >= 0.0


def test_final_output_is_fetched_once_and_the_terminal_is_released(
    tmp_path: Path,
) -> None:
    """跑完取一次就留着：终端此时已经放掉，编辑器那边仍继续显示输出。"""
    fake = _Fake()
    live = _process(fake, tmp_path)
    live.wait(2.0)
    assert live.snapshot_stdio()[0] == _EDITOR_OUTPUT
    assert live.snapshot_stdio()[0] == _EDITOR_OUTPUT
    assert fake.output_calls == 1
    assert fake.released == ["term-1"]


def test_running_command_is_not_released_and_keeps_being_asked(
    tmp_path: Path,
) -> None:
    fake = _Fake(running=True)
    live = _process(fake, tmp_path)
    assert live.wait(0.1) is False
    assert live.returncode() is None
    assert live.snapshot_stdio()[0] == _EDITOR_OUTPUT
    assert live.snapshot_stdio()[0] == _EDITOR_OUTPUT
    assert fake.output_calls == 2
    assert fake.released == []
    fake.finish()
    assert live.wait(2.0) is True


def test_truncated_output_says_so(tmp_path: Path) -> None:
    fake = _Fake(truncated=True)
    live = _process(fake, tmp_path)
    live.wait(2.0)
    text, _err = live.snapshot_stdio()
    assert "只保留了最近的输出" in text
    assert _EDITOR_OUTPUT in text


def test_signal_only_exit_is_reported_as_minus_one(tmp_path: Path) -> None:
    fake = _Fake(exit_code=None)
    live = _process(fake, tmp_path)
    live.wait(2.0)
    assert live.returncode() == -1


def test_cancel_kills_the_editor_terminal(tmp_path: Path) -> None:
    fake = _Fake(running=True)
    live = _process(fake, tmp_path)
    assert live.wait(2.0, cancel_check=lambda: True) is True
    assert fake.killed == ["term-1"]
    assert live.error == "cancelled"
    assert live.returncode() is not None


def test_hard_deadline_kills_the_editor_terminal(tmp_path: Path) -> None:
    fake = _Fake(running=True)
    live = _process(fake, tmp_path, hard_timeout_sec=0.0)
    # hard_timeout 下限是 5 秒，直接把 deadline 挪到过去
    live.hard_deadline = time.monotonic() - 1.0
    assert live.wait(2.0) is True
    assert fake.killed == ["term-1"]
    assert live.error == "timeout"


def test_kill_is_idempotent(tmp_path: Path) -> None:
    fake = _Fake(running=True)
    live = _process(fake, tmp_path)
    live.kill("cancelled")
    live.kill("timeout")
    assert fake.killed == ["term-1"]
    assert live.error == "cancelled"


def test_a_lost_terminal_still_finishes_the_job(tmp_path: Path) -> None:
    """编辑器失联：这条 job 不能永远算在运行中，输出里要说清为什么是空的。"""

    class _Lost(_Fake):
        def wait_for_exit(self, terminal_id: str) -> TerminalExit | None:
            return None

    fake = _Lost(answer_output=False)
    live = _process(fake, tmp_path)
    assert live.wait(2.0) is True
    assert live.returncode() == -1
    text, _err = live.snapshot_stdio()
    assert "失联" in text


# ---- run_shell_command ----


def test_run_shell_command_hands_the_command_to_the_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有编辑器终端时不许再本地起进程：否则同一条命令跑两遍。"""

    def _no_local(*_args: Any, **_kwargs: Any):
        raise AssertionError("不该回落本地进程")

    monkeypatch.setattr("llgraph.core.shell_tools.spawn_sandboxed_shell", _no_local)
    fake = _Fake()
    tool = _shell_tool(tmp_path)
    with use_editor_terminal_source(fake):
        out = tool.invoke({"command": "echo hi"})

    # 输出只可能来自那个假终端：本地那条路一走就炸
    assert _EDITOR_OUTPUT.strip() in out
    assert len(fake.created) == 1
    created = fake.created[0]
    assert created["command"] == "/bin/sh"
    assert created["args"] == ["-c", "echo hi"]
    assert created["cwd"] == str(tmp_path)
    assert created["output_byte_limit"] == terminal_output_byte_limit(4000)
    # 结果格式化一行没改：头还是那个头，退出码照旧
    assert out.startswith("--- shell (nosandbox, cwd=.")
    assert "exit=0" in out


def test_working_directory_is_passed_as_an_absolute_path(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    fake = _Fake()
    tool = _shell_tool(tmp_path)
    with use_editor_terminal_source(fake):
        tool.invoke({"command": "ls", "working_directory": "src"})
    assert fake.created[0]["cwd"] == str(tmp_path / "src")


def test_failing_command_keeps_the_exit_footer(tmp_path: Path) -> None:
    fake = _Fake(exit_code=2)
    tool = _shell_tool(tmp_path)
    with use_editor_terminal_source(fake):
        out = tool.invoke({"command": "false"})
    assert "exit=2" in out
    assert "[exit 2]" in out


def test_background_job_goes_through_await_shell(tmp_path: Path) -> None:
    """后台 job 表与 await_shell 一行没改，编辑器终端照样能 await。"""
    fake = _Fake(running=True)
    ctx = WorkspaceContext(tmp_path, allow_write=False)
    tools = create_shell_tools(ctx, allow_write=False, settings=_settings())
    run_tool = next(t for t in tools if t.name == "run_shell_command")
    await_tool = next(t for t in tools if t.name == "await_shell")

    with use_editor_terminal_source(fake):
        started = run_tool.invoke({"command": "sleep 9", "block_until_ms": 0})
        assert "running" in started
        job_id = started.split("job=")[1].split(",")[0].split(")")[0]
        fake.finish()
        done = await_tool.invoke({"job_id": job_id, "block_until_ms": 2000})
    assert "exit=0" in done
    assert _EDITOR_OUTPUT.strip() in done


def test_sandboxed_workspace_keeps_running_locally(tmp_path: Path) -> None:
    """沙箱开着就别交给编辑器：那条路没有 seatbelt / bwrap 包装。"""

    class _SandboxOn:
        enabled = True
        backend = "不存在的后端"

    fake = _Fake()
    tool = _shell_tool(tmp_path, sandbox_policy=_SandboxOn())
    with use_editor_terminal_source(fake):
        out = tool.invoke({"command": "echo hi"})
    assert fake.created == []
    # 本地那条路被走到了（这个后端起不来，正好能看出来）
    assert "沙箱后端不可用" in out


def test_blocked_command_never_reaches_the_editor(tmp_path: Path) -> None:
    """能不能跑在交出去之前就判完了。"""
    fake = _Fake()
    tool = _shell_tool(tmp_path)
    with use_editor_terminal_source(fake):
        out = tool.invoke({"command": "rm -rf /"})
    assert out.startswith("错误:")
    assert fake.created == []


def test_cd_only_command_does_not_open_a_terminal(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    fake = _Fake()
    tool = _shell_tool(tmp_path)
    with use_editor_terminal_source(fake):
        out = tool.invoke({"command": "cd src"})
    assert "已切换工作目录" in out
    assert fake.created == []


# ---- ACP 桥 ----


class _FakeConn:
    """只实现 ``request``：按脚本回 result 或抛异常。"""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def request(self, method: str, params: dict[str, Any], **_kwargs: Any) -> Any:
        self.sent.append((method, params))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _bridge(replies: list[Any], **kwargs: Any) -> tuple[AcpTerminalBridge, _FakeConn]:
    conn = _FakeConn(replies)
    bridge = AcpTerminalBridge(
        conn,  # type: ignore[arg-type]
        "cli-term",
        timeout_sec=1.0,
        **kwargs,
    )
    return bridge, conn


def test_capability_parsing() -> None:
    assert client_terminal_capability(None) is False
    assert client_terminal_capability({}) is False
    assert client_terminal_capability({"terminal": False}) is False
    assert client_terminal_capability({"terminal": True}) is True


def test_create_payload() -> None:
    bridge, conn = _bridge([{"terminalId": "term_x"}])
    assert (
        bridge.create(
            command="/bin/sh", args=["-c", "pytest -q"], cwd="/ws", output_byte_limit=99
        )
        == "term_x"
    )
    method, params = conn.sent[0]
    assert method == "terminal/create"
    assert params == {
        "sessionId": "cli-term",
        "command": "/bin/sh",
        "args": ["-c", "pytest -q"],
        "cwd": "/ws",
        "outputByteLimit": 99,
    }


def test_output_payload_and_parsing() -> None:
    bridge, conn = _bridge(
        [
            {"output": "跑着呢\n", "truncated": False},
            {
                "output": "跑完了\n",
                "truncated": True,
                "exitStatus": {"exitCode": 0, "signal": None},
            },
        ]
    )
    running = bridge.output("term_x")
    assert running is not None
    assert running.output == "跑着呢\n"
    assert running.exit is None
    assert conn.sent[0] == (
        "terminal/output",
        {"sessionId": "cli-term", "terminalId": "term_x"},
    )

    done = bridge.output("term_x")
    assert done is not None
    assert done.truncated is True
    assert done.exit == TerminalExit(exit_code=0, signal=None)


def test_exit_status_is_read_from_both_shapes() -> None:
    """``wait_for_exit`` 放顶层，``output`` 放 exitStatus 下；认错一种就会误判还在跑。"""
    assert parse_exit_status({"exitCode": 1, "signal": None}) == TerminalExit(1, None)
    assert parse_exit_status({"exitStatus": {"exitCode": 0}}) == TerminalExit(0, None)
    assert parse_exit_status({"exitCode": None, "signal": "SIGKILL"}) == TerminalExit(
        None, "SIGKILL"
    )
    assert parse_exit_status({"output": "还在跑"}) is None
    assert parse_exit_status(None) is None
    assert parse_exit_status({"exitCode": True}) == TerminalExit(None, None)


def test_wait_for_exit_waits_without_a_timeout() -> None:
    """命令可能跑一小时，设了超时只会把它误判成失联。"""
    bridge, conn = _bridge([{"exitCode": 0, "signal": None}])
    assert bridge.wait_for_exit("term_x") == TerminalExit(0, None)
    assert conn.sent[0][0] == "terminal/wait_for_exit"


def test_kill_and_release_payloads() -> None:
    bridge, conn = _bridge([None, None])
    bridge.kill("term_x")
    bridge.release("term_x")
    assert [m for m, _p in conn.sent] == ["terminal/kill", "terminal/release"]


def test_kill_failure_is_swallowed() -> None:
    bridge, _conn = _bridge([RequestFailed("编辑器没了")])
    bridge.kill("term_x")


def test_garbage_create_reply_is_none() -> None:
    bridge, _conn = _bridge([{"id": "没按协议来"}])
    assert (
        bridge.create(command="/bin/sh", args=[], cwd="/ws", output_byte_limit=1) is None
    )


def test_repeated_failures_disable_the_bridge() -> None:
    """每次失败都要等一个超时，连着失败就别再问了。"""
    bridge, conn = _bridge([RequestFailed("超时")] * 3)
    for _ in range(3):
        assert (
            bridge.create(command="/bin/sh", args=[], cwd="/ws", output_byte_limit=1)
            is None
        )
    assert bridge.disabled is True
    assert len(conn.sent) == 3

    conn.replies = [{"terminalId": "term_x"}]
    assert (
        bridge.create(command="/bin/sh", args=[], cwd="/ws", output_byte_limit=1) is None
    )
    assert len(conn.sent) == 3


def test_cancelled_turn_does_not_open_a_terminal() -> None:
    bridge, conn = _bridge([], cancel_check=lambda: True)
    assert (
        bridge.create(command="/bin/sh", args=[], cwd="/ws", output_byte_limit=1) is None
    )
    assert conn.sent == []

    # 等待期间被取消：不算编辑器的错，不计进熔断
    bridge2, _c2 = _bridge([RequestCancelled("停止")])
    assert (
        bridge2.create(command="/bin/sh", args=[], cwd="/ws", output_byte_limit=1)
        is None
    )
    assert bridge2.disabled is False


def test_output_is_fetched_even_after_cancel() -> None:
    """用户点了停止，模型仍要看到命令跑出了什么。"""
    bridge, conn = _bridge([{"output": "停之前的输出\n"}], cancel_check=lambda: True)
    snapshot = bridge.output("term_x")
    assert snapshot is not None
    assert snapshot.output == "停之前的输出\n"
    assert conn.sent[0][0] == "terminal/output"


# ---- 载荷与 Sink ----


def _tool_step(**overrides: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "kind": "tool",
        "title": "执行 run_shell_command(pytest -q)",
        "body_lines": ["--- shell (nosandbox, exit=0) ---", "1 passed"],
        "tool_call_id": "toolu_1",
    }
    step.update(overrides)
    return step


def _record(**overrides: Any) -> Any:
    from llgraph.display.trace_display import TraceStepRecord

    base: dict[str, Any] = {
        "step_id": 1,
        "elapsed": 0.1,
        "summary": "2 行输出",
        **_tool_step(),
    }
    base.update(overrides)
    return TraceStepRecord(**base)


def test_finished_step_carries_the_terminal_instead_of_a_text_copy() -> None:
    payload = tool_call_from_step(
        _tool_step(), tool_call_id="t1_call_1", as_update=True, terminals=["term_x"]
    )
    assert payload is not None
    assert payload["content"] == [{"type": "terminal", "terminalId": "term_x"}]
    assert payload["status"] == "completed"


def test_finished_step_without_a_terminal_still_carries_text() -> None:
    payload = tool_call_from_step(
        _tool_step(), tool_call_id="t1_call_1", as_update=True
    )
    assert payload is not None
    assert payload["content"][0]["type"] == "content"


def test_failed_step_keeps_the_terminal_too() -> None:
    payload = tool_call_from_step(
        _tool_step(tool_failed=True),
        tool_call_id="t1_call_1",
        as_update=True,
        terminals=["term_x"],
    )
    assert payload is not None
    assert payload["status"] == "failed"
    assert payload["content"] == [{"type": "terminal", "terminalId": "term_x"}]


def _sink() -> tuple[AcpTraceSink, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    return AcpTraceSink(sent.append, id_prefix="t1_"), sent


def test_sink_sends_the_terminal_right_away() -> None:
    """实时输出就靠这一条：攒到收尾再发等于没做。"""
    sink, sent = _sink()
    sink.tool_calls_planned(
        [{"id": "toolu_1", "name": "run_shell_command", "title": "执行 run_shell_command"}]
    )
    sink.tool_started("toolu_1", "run_shell_command")
    sink.tool_terminal("toolu_1", "term_x")
    assert sent[-1] == {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "t1_call_1",
        "content": [{"type": "terminal", "terminalId": "term_x"}],
    }

    sink.step_added(_record())
    assert sent[-1]["status"] == "completed"
    assert sent[-1]["content"] == [{"type": "terminal", "terminalId": "term_x"}]


def test_sink_ignores_a_terminal_for_an_unknown_call() -> None:
    """没报过 pending 的调用没人收尾，挂上去的终端会一直转圈。"""
    sink, sent = _sink()
    sink.tool_terminal("toolu_unknown", "term_x")
    assert sent == []


def test_sink_drops_duplicate_terminal_notices() -> None:
    sink, sent = _sink()
    sink.tool_calls_planned([{"id": "toolu_1", "name": "run_shell_command", "title": "x"}])
    sink.tool_terminal("toolu_1", "term_x")
    sink.tool_terminal("toolu_1", "term_x")
    assert len([u for u in sent if "content" in u]) == 1


def test_sink_keeps_terminals_per_call() -> None:
    """并行两条命令各归各位，别把隔壁的终端挂到自己那一行。"""
    sink, sent = _sink()
    sink.tool_calls_planned(
        [
            {"id": "toolu_1", "name": "run_shell_command", "title": "a"},
            {"id": "toolu_2", "name": "run_shell_command", "title": "b"},
        ]
    )
    sink.tool_terminal("toolu_1", "term_a")
    sink.tool_terminal("toolu_2", "term_b")
    sink.step_added(_record(tool_call_id="toolu_2"))
    finished = sent[-1]
    assert finished["toolCallId"] == "t1_call_2"
    assert finished["content"] == [{"type": "terminal", "terminalId": "term_b"}]

    sink.step_added(_record(step_id=2, tool_call_id="toolu_1"))
    assert sent[-1]["content"] == [{"type": "terminal", "terminalId": "term_a"}]


# ---- 协议层：编辑器与 Agent 之间真走一趟 ----


def _create_runner(seen: list[Any]):
    """turn_runner 桩件：跑一轮里开一个终端，把终端 id 写进 result.text。"""

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        seen.append(req.editor_terminal)
        if req.editor_terminal is None:
            return AcpTurnResult(text="没有编辑器终端")
        tid = req.editor_terminal.create(
            command="/bin/sh", args=["-c", "echo hi"], cwd="/ws", output_byte_limit=1024
        )
        return AcpTurnResult(text=str(tid))

    return runner


def test_editor_gets_a_terminal_create_request(tmp_path: Path) -> None:
    seen: list[Any] = []
    h = _Harness(turn_runner=_create_runner(seen))
    try:
        h.send(
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {"terminal": True}},
            request_id=1,
        )
        h.response(1)
        h.send("session/new", {"cwd": str(tmp_path), "mcpServers": []}, request_id=2)
        session_id = h.response(2)["result"]["sessionId"]
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "跑个命令"}]},
            request_id=3,
        )
        ask = h.out.wait_for(lambda m: m.get("method") == "terminal/create")
        assert ask["params"]["sessionId"] == session_id
        assert ask["params"]["args"] == ["-c", "echo hi"]

        h.send_raw(
            json.dumps(
                {"jsonrpc": "2.0", "id": ask["id"], "result": {"terminalId": "term_x"}}
            )
        )
        assert h.response(3)["result"]["stopReason"] == "end_turn"
        assert seen and seen[0] is not None
    finally:
        h.close()


def test_client_without_terminal_capability_gets_no_reverse_request(
    tmp_path: Path,
) -> None:
    """编辑器没声明 terminal 能力：命令照旧本地跑，一条反向请求都不该发。"""
    seen: list[Any] = []
    h = _Harness(turn_runner=_create_runner(seen))
    try:
        h.send("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, request_id=1)
        h.response(1)
        h.send("session/new", {"cwd": str(tmp_path), "mcpServers": []}, request_id=2)
        session_id = h.response(2)["result"]["sessionId"]
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "跑个命令"}]},
            request_id=3,
        )
        h.response(3)
        assert seen == [None]
        assert all(
            not str(m.get("method") or "").startswith("terminal/")
            for m in h.out.messages
        )
    finally:
        h.close()
