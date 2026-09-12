"""MCP 子进程挂掉后的恢复回归：重连、快速失败、坏工具封停。

断的是语义边界，不是文案：

- 工具自己报错不许触发重连（连接是好的，重连白烧）
- 传输层断了要重连，且**写类工具与超时不许自动重放**（服务端可能已生效）
- 死循环必须终止：重连预算有限，用完之后立刻快速失败
- 一个坏工具只封它自己，不封整个 Server

刻意不依赖可选依赖 `mcp`：CI 只装 `[web]`。运行时的连接步骤用假 session 顶掉。
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from llgraph.config.mcp_config import McpServerConfig, resolve_mcp_settings
from llgraph.core.mcp_health import (
    REASON_DISCONNECTED,
    REASON_TIMEOUT,
    classify_mcp_failure,
    replay_allowed,
)
from llgraph.core.mcp_tools import _McpServerRuntime


class _PydanticLike:
    """模拟 pydantic 模型：读不存在的字段抛 AttributeError，而不是给 None。"""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)

    def __getattr__(self, item: str) -> object:
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r}"
        )


def _text_block(text: str) -> _PydanticLike:
    return _PydanticLike(type="text", text=text)


# ------------------------------------------------------------ 传输层故障分类


def test_connection_closed_is_transport_failure() -> None:
    assert classify_mcp_failure(RuntimeError("Connection closed")) is not None


def test_transport_failure_found_through_cause_chain() -> None:
    inner = RuntimeError("Connection closed")
    outer = ValueError("call_tool failed")
    outer.__cause__ = inner
    assert classify_mcp_failure(outer) is not None


def test_transport_failure_found_inside_exception_group() -> None:
    group = ExceptionGroup(  # noqa: F821 — 3.11+ 内建
        "unhandled errors in a TaskGroup", [RuntimeError("Connection closed")]
    )
    assert classify_mcp_failure(group) is not None


def test_transport_failure_recognised_by_exception_class_name() -> None:
    class ClosedResourceError(Exception):
        pass

    assert classify_mcp_failure(ClosedResourceError("")) is not None


def test_tool_level_error_is_not_transport_failure() -> None:
    """工具自己报错不能触发重连：连接是好的，重连只会白烧一次。"""
    assert classify_mcp_failure(ValueError("syntax error near 'SELEC'")) is None
    assert classify_mcp_failure(ValueError("table not found: orders")) is None


def test_replay_only_for_read_tools_and_known_disconnect() -> None:
    assert replay_allowed(REASON_DISCONNECTED, is_write_tool=False) is True
    # 写类工具：服务端可能已经生效，重放等于悄悄改两遍
    assert replay_allowed(REASON_DISCONNECTED, is_write_tool=True) is False
    # 超时：服务端是否执行完全未知
    assert replay_allowed(REASON_TIMEOUT, is_write_tool=False) is False


# ------------------------------------------------------------------ 运行时恢复


class _FakeSession:
    """假 MCP 会话：被指定的工具会「打挂进程」，之后所有调用都报连接已断。"""

    def __init__(self, script: _Script) -> None:
        self._script = script
        self.dead = False

    async def call_tool(self, name: str, arguments: dict) -> object:
        self._script.calls.append(name)
        if self.dead:
            raise RuntimeError("Connection closed")
        if name in self._script.gang_tools and not self._script.gang_done:
            await self._script.join_gang()
            self.dead = True
            raise RuntimeError("Connection closed")
        if name in self._script.hang_tools:
            await asyncio.sleep(30)
        if name in self._script.killer_tools:
            self.dead = True
            raise RuntimeError("Connection closed")
        if name in self._script.error_tools:
            return _PydanticLike(content=[_text_block("bad sql")], is_error=True)
        if name in self._script.raise_tools:
            raise ValueError("syntax error near 'SELEC'")
        return _PydanticLike(
            content=[_text_block(f"ok:{name}")], structured_content=None, is_error=False
        )

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _Script:
    """一次测试里所有连接共享的剧本与调用流水。"""

    def __init__(
        self,
        *,
        killer_tools: tuple[str, ...] = (),
        error_tools: tuple[str, ...] = (),
        raise_tools: tuple[str, ...] = (),
        hang_tools: tuple[str, ...] = (),
        gang_tools: tuple[str, ...] = (),
        gang_size: int = 0,
        connect_fails_after: int = 10_000,
    ) -> None:
        self.killer_tools = set(killer_tools)
        self.error_tools = set(error_tools)
        self.raise_tools = set(raise_tools)
        self.hang_tools = set(hang_tools)
        self.gang_tools = set(gang_tools)
        self.gang_size = gang_size
        self.gang_done = False
        self.connect_fails_after = connect_fails_after
        self.calls: list[str] = []
        self.connects = 0
        self._gang_seen = 0
        self._gang_gate: asyncio.Event | None = None

    async def join_gang(self) -> None:
        """卡住调用直到 gang_size 个线程都进来，再一起报连接断开。

        并发回归必须**确定性地**造出「多个线程撞到同一次断开」：
        让它们自然抢的话，运行时的锁会把后来者挡到重连之后，
        去重逻辑一次都跑不到（本轮就是这么漏过一次变异验证的）。
        """
        if self._gang_gate is None:
            self._gang_gate = asyncio.Event()
        self._gang_seen += 1
        if self._gang_seen >= self.gang_size:
            self.gang_done = True
            self._gang_gate.set()
        await self._gang_gate.wait()


class _FakeRuntime(_McpServerRuntime):
    """把真实 stdio 连接换成假 session，其余状态机（重连/封停）走生产代码。"""

    def __init__(self, script: _Script, **kwargs: object) -> None:
        config = McpServerConfig(
            name="probe", command="true", args=[], env={}, cwd=None, enabled=True
        )
        super().__init__(config, **kwargs)  # type: ignore[arg-type]
        self._script = script

    async def _connect(self) -> None:
        self._script.connects += 1
        if self._script.connects > self._script.connect_fails_after:
            raise RuntimeError("spawn failed")
        self._session = _FakeSession(self._script)
        self._tool_desc = {
            "read_query": "执行只读 SQL 查询",
            "write_row": "insert a row",
            "boom": "crash the server",
            "hang": "never returns",
        }


@pytest.fixture
def script_runtime():
    started: list[_FakeRuntime] = []

    def _make(script: _Script, **kwargs: object) -> _FakeRuntime:
        runtime = _FakeRuntime(script, timeout_sec=5.0, **kwargs)
        assert runtime.start() is True
        started.append(runtime)
        return runtime

    yield _make
    for runtime in started:
        runtime.stop()


def test_healthy_call_returns_body(script_runtime) -> None:
    script = _Script()
    runtime = script_runtime(script)
    assert runtime.call_tool_sync("read_query", {}) == "ok:read_query"
    assert script.connects == 1


def test_tool_error_does_not_trigger_reconnect(script_runtime) -> None:
    script = _Script(error_tools=("read_query",))
    runtime = script_runtime(script)
    out = runtime.call_tool_sync("read_query", {})
    assert out.startswith("MCP 错误")
    assert script.connects == 1
    assert runtime._reconnects == 0


def test_non_transport_exception_does_not_trigger_reconnect(script_runtime) -> None:
    script = _Script(raise_tools=("read_query",))
    runtime = script_runtime(script)
    out = runtime.call_tool_sync("read_query", {})
    assert "MCP 调用失败" in out
    assert script.connects == 1
    assert runtime._reconnects == 0


def test_dead_connection_reconnects_and_replays_read_tool(script_runtime) -> None:
    script = _Script()
    runtime = script_runtime(script)
    # 模拟「上一次调用已经把连接弄没了」：session 还在但已判死
    runtime._session.dead = True

    out = runtime.call_tool_sync("read_query", {})
    assert out == "ok:read_query"
    assert script.connects == 2
    assert runtime._reconnects == 1


def test_write_tool_is_reconnected_but_never_auto_replayed(script_runtime) -> None:
    script = _Script()
    runtime = script_runtime(script)
    runtime._session.dead = True

    out = runtime.call_tool_sync("write_row", {})
    assert "是否已在服务端生效未知" in out
    assert "不要直接重试" in out
    # 连接重建了，但这次写调用只发过一次
    assert script.connects == 2
    assert script.calls.count("write_row") == 1

    # 下一次写调用走的是新连接，正常执行
    assert runtime.call_tool_sync("write_row", {}) == "ok:write_row"


def test_crashing_tool_is_banned_alone_and_server_stays_usable(script_runtime) -> None:
    script = _Script(killer_tools=("boom",))
    runtime = script_runtime(script)

    out = runtime.call_tool_sync("boom", {})
    assert "每次调用都会让 MCP Server" in out
    assert "其它工具仍然可用" in out

    # 同 Server 的其它工具照用（会先把连接重建起来）
    assert runtime.call_tool_sync("read_query", {}) == "ok:read_query"

    # 再点这个坏工具：立刻封停，不再执行、不再烧重连预算
    before_calls = len(script.calls)
    before_connects = script.connects
    again = runtime.call_tool_sync("boom", {})
    assert "本会话不再执行它" in again
    assert len(script.calls) == before_calls
    assert script.connects == before_connects


def test_reconnect_budget_zero_marks_server_unavailable(script_runtime) -> None:
    script = _Script()
    runtime = script_runtime(script, max_reconnects=0)
    runtime._session.dead = True

    out = runtime.call_tool_sync("read_query", {})
    assert "本会话不再调用该 Server" in out
    assert script.connects == 1

    # 之后所有工具都快速失败，不再碰连接
    before = len(script.calls)
    assert "本会话不再调用该 Server" in runtime.call_tool_sync("write_row", {})
    assert len(script.calls) == before


def test_unavailable_server_fast_fails_without_waiting(script_runtime) -> None:
    script = _Script()
    runtime = script_runtime(script, max_reconnects=0)
    runtime._session.dead = True
    runtime.call_tool_sync("read_query", {})

    started = time.monotonic()
    for _ in range(20):
        out = runtime.call_tool_sync("read_query", {})
        assert "不要重复调用" in out
    # 改前这里是每次都等满 timeout_sec；现在必须是纯内存返回
    assert time.monotonic() - started < 1.0


def test_always_dying_server_terminates_within_reconnect_budget(script_runtime) -> None:
    """每次调用都把进程打挂的 server：重连次数必须有界，不能无限重启。"""
    script = _Script(killer_tools=("read_query",))
    runtime = script_runtime(script, max_reconnects=3)

    outs = [runtime.call_tool_sync("read_query", {}) for _ in range(12)]
    assert runtime._reconnects <= 3
    assert script.connects <= 4
    # 收尾一定落在明确的「别再调了」上
    assert "不再执行它" in outs[-1] or "不再调用该 Server" in outs[-1]


def test_failed_reconnect_marks_unavailable(script_runtime) -> None:
    script = _Script(connect_fails_after=1)
    runtime = script_runtime(script)
    runtime._session.dead = True

    out = runtime.call_tool_sync("read_query", {})
    assert "重连也失败" in out
    assert "本会话不再调用该 Server" in out


def test_concurrent_failures_burn_one_reconnect(script_runtime) -> None:
    """4 个线程撞到同一次断开：只重启一次进程，也不能把工具算成坏工具。"""
    workers = 4
    script = _Script(gang_tools=("read_query",), gang_size=workers)
    runtime = script_runtime(script, max_reconnects=3)

    results: list[str] = []
    lock = threading.Lock()

    def _worker() -> None:
        out = runtime.call_tool_sync("read_query", {})
        with lock:
            results.append(out)

    threads = [threading.Thread(target=_worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert len(results) == workers
    # 一次断开不该让任何一个线程看到「这个工具坏了」
    assert all(out == "ok:read_query" for out in results), results
    assert runtime._reconnects == 1
    assert script.connects == 2
    assert runtime._tool_crashes.get("read_query") == 1


def test_call_timeout_reconnects_without_replay(script_runtime) -> None:
    script = _Script(hang_tools=("hang",))
    runtime = _FakeRuntime(script, timeout_sec=1.0, max_reconnects=2)
    assert runtime.start() is True
    try:
        out = runtime.call_tool_sync("hang", {})
        assert "超时" in out
        assert "是否已在服务端生效未知" in out
        # 超时不许自动重放：只发过一次
        assert script.calls.count("hang") == 1
        assert script.connects == 2
    finally:
        runtime.stop()


def test_stop_returns_before_timeout_elapses(script_runtime, monkeypatch) -> None:
    """退避/等待期间按 Stop 要立刻返回，不能等满 timeout_sec。"""
    script = _Script(hang_tools=("hang",))
    runtime = _FakeRuntime(script, timeout_sec=30.0, max_reconnects=0)
    assert runtime.start() is True
    monkeypatch.setattr("llgraph.core.mcp_tools._cancel_requested", lambda: True)
    try:
        started = time.monotonic()
        out = runtime.call_tool_sync("hang", {})
        elapsed = time.monotonic() - started
        assert "已停止当前生成" in out
        assert elapsed < 5.0
        assert runtime._reconnects == 0
    finally:
        runtime.stop()


def test_max_reconnects_is_configurable(tmp_path) -> None:
    (tmp_path / ".llgraph").mkdir()
    (tmp_path / ".llgraph" / "mcp.json").write_text(
        '{"defaults": {"max_reconnects": 1}, "servers": {}}', encoding="utf-8"
    )
    assert resolve_mcp_settings(tmp_path).max_reconnects == 1


def test_max_reconnects_defaults_and_clamps(tmp_path) -> None:
    assert resolve_mcp_settings(tmp_path).max_reconnects == 3
    (tmp_path / ".llgraph").mkdir()
    (tmp_path / ".llgraph" / "mcp.json").write_text(
        '{"defaults": {"max_reconnects": 999}, "servers": {}}', encoding="utf-8"
    )
    assert resolve_mcp_settings(tmp_path).max_reconnects == 10
