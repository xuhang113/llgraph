"""工具调用的三段状态：pending（模型决定）→ in_progress（真开跑）→ completed（跑完）。

trace 的步骤只在工具跑完后才登记，所以编辑器里长命令期间看不到动静。
这里钉住补上的两条来源：模型决策时的 pending 通知，与 ToolNode 每次调用前的起步通知。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from llgraph.core.tool_invoke_timing import (
    lookup_tool_timing,
    reset_tool_timings,
    wrap_tool_node_with_timing,
)
from llgraph.core.tool_progress import (
    current_tool_start_observer,
    notify_tool_started,
    use_tool_start_observer,
)
from llgraph.display.trace_display import TraceSession, TurnTracePrinter
from llgraph.editor.acp.sink import AcpTraceSink


class _FakeSink:
    """只收工具相关事件的 Sink（不实现 TraceSink 全部方法也能被 hasattr 探到）。"""

    def __init__(self) -> None:
        self.planned: list[list[dict[str, Any]]] = []
        self.steps: list[Any] = []

    def line(self, text: str) -> None:
        pass

    def stream(self, text: str) -> None:
        pass

    def stream_end(self) -> None:
        pass

    def step_added(self, step: Any) -> None:
        self.steps.append(step)

    def step_selected(self, step_id: int) -> None:
        pass

    def tool_calls_planned(self, calls: list[dict[str, Any]]) -> None:
        self.planned.append(calls)


class _FakeToolNode:
    """够用的 ToolNode 替身：只有计时包装要接的那两个方法。"""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def _run_one(self, call: dict[str, Any], input_type: Any, tool_runtime: Any) -> str:
        self.ran.append(str(call.get("id")))
        return "ok"

    async def _arun_one(self, call: dict[str, Any], input_type: Any, tool_runtime: Any) -> str:
        self.ran.append(str(call.get("id")))
        return "ok"


def _tool_step(**kwargs: Any) -> Any:
    from llgraph.display.trace_display import TraceStepRecord

    base: dict[str, Any] = {
        "step_id": 1,
        "kind": "tool",
        "title": "执行 read_file(a.txt)",
        "elapsed": 0.2,
        "summary": "1 行输出",
        "body_lines": ["ok"],
    }
    base.update(kwargs)
    return TraceStepRecord(**base)


# ---- 观察者语义 ----


def test_no_observer_registered_is_a_noop() -> None:
    assert current_tool_start_observer() is None
    notify_tool_started("call_1", "read_file")  # 不应抛


def test_observer_scope_ends_with_the_with_block() -> None:
    seen: list[tuple[str, str]] = []
    with use_tool_start_observer(lambda cid, name: seen.append((cid, name))):
        notify_tool_started("c1", "read_file")
    notify_tool_started("c2", "read_file")
    assert seen == [("c1", "read_file")]


def test_observer_exception_does_not_escape() -> None:
    def boom(cid: str, name: str) -> None:
        raise RuntimeError("界面炸了不能带崩工具")

    with use_tool_start_observer(boom):
        notify_tool_started("c1", "read_file")


def test_blank_tool_call_id_is_ignored() -> None:
    seen: list[str] = []
    with use_tool_start_observer(lambda cid, name: seen.append(cid)):
        notify_tool_started("  ", "read_file")
    assert seen == []


# ---- ToolNode 每次调用之前叫一声 ----


def test_timing_wrapper_notifies_before_running() -> None:
    node = _FakeToolNode()
    wrap_tool_node_with_timing(node)
    reset_tool_timings()

    order: list[str] = []
    node_ran = node.ran

    def observer(cid: str, name: str) -> None:
        # 通知必须发生在工具真正跑之前，否则「正在跑」晚于「已跑完」
        order.append(f"start:{cid}:{name}:{len(node_ran)}")

    with use_tool_start_observer(observer):
        node._run_one({"id": "c1", "name": "read_file"}, None, None)

    assert order == ["start:c1:read_file:0"]
    assert node.ran == ["c1"]
    assert lookup_tool_timing("c1") is not None


def test_timing_wrapper_notifies_on_the_async_path_too() -> None:
    node = _FakeToolNode()
    wrap_tool_node_with_timing(node)
    reset_tool_timings()

    seen: list[str] = []
    with use_tool_start_observer(lambda cid, name: seen.append(cid)):
        asyncio.run(node._arun_one({"id": "c2", "name": "grep"}, None, None))

    assert seen == ["c2"]
    assert lookup_tool_timing("c2") is not None


# ---- Sink：三段串成一条 ----


def test_sink_reports_pending_then_in_progress_then_completed() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append, id_prefix="t2_")

    sink.tool_calls_planned(
        [{"id": "raw-1", "name": "read_file", "title": "执行 read_file(a.txt)"}]
    )
    sink.tool_started("raw-1", "read_file")
    sink.step_added(_tool_step(tool_call_id="raw-1"))

    assert [u["sessionUpdate"] for u in updates] == [
        "tool_call",
        "tool_call_update",
        "tool_call_update",
    ]
    assert [u["status"] for u in updates] == ["pending", "in_progress", "completed"]
    assert {u["toolCallId"] for u in updates} == {"t2_call_1"}
    assert updates[0]["title"] == "执行 read_file(a.txt)"
    assert updates[0]["kind"] == "read"
    assert "ok" in updates[-1]["content"][0]["content"]["text"]
    # 收尾不重发标题：工具节点的输出里没有调用参数，重算会退化成「执行 read_file」
    assert "title" not in updates[-1]


def test_sink_keeps_one_id_per_call_across_parallel_tools() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.tool_calls_planned(
        [
            {"id": "raw-1", "name": "read_file", "title": "执行 read_file(a.txt)"},
            {"id": "raw-2", "name": "grep", "title": "执行 grep(x)"},
        ]
    )
    # 并行工具的完成顺序可能与规划顺序相反，id 必须跟着 tool_call_id 走
    sink.step_added(_tool_step(step_id=1, tool_call_id="raw-2", title="执行 grep(x)"))
    sink.step_added(_tool_step(step_id=2, tool_call_id="raw-1"))

    by_id = [(u["toolCallId"], u["status"]) for u in updates]
    assert by_id == [
        ("call_1", "pending"),
        ("call_2", "pending"),
        ("call_2", "completed"),
        ("call_1", "completed"),
    ]
    assert [u.get("title") for u in updates[:2]] == ["执行 read_file(a.txt)", "执行 grep(x)"]


def test_sink_ignores_duplicate_plan_notifications() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)
    call = {"id": "raw-1", "name": "read_file", "title": "执行 read_file(a.txt)"}

    sink.tool_calls_planned([call])
    sink.tool_calls_planned([call])

    assert len(updates) == 1


def test_sink_does_not_report_progress_for_unannounced_calls() -> None:
    """没报过 pending 的调用（trace 静默 / explore 那类）不该冒出一条转圈的记录。"""
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.tool_started("raw-unknown", "read_file")

    assert updates == []


def test_sink_does_not_reopen_a_finished_call() -> None:
    """同一 id 的迟到起步通知（工具重试 / 包装层重复调用）不该把完成态推回进行中。"""
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.tool_calls_planned(
        [{"id": "raw-1", "name": "read_file", "title": "执行 read_file(a.txt)"}]
    )
    sink.step_added(_tool_step(tool_call_id="raw-1"))
    sink.tool_started("raw-1", "read_file")

    assert [u["status"] for u in updates] == ["pending", "completed"]


def test_sink_still_creates_a_tool_call_for_steps_without_a_call_id() -> None:
    """explore 步骤没有 tool_call_id（它由 emit_explore_trace_step 登记），仍按新建发。"""
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.step_added(_tool_step(kind="explore", title="Explore", tool_call_id=None))

    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "tool_call"
    assert updates[0]["status"] == "completed"
    assert updates[0]["kind"] == "think"


def test_non_tool_steps_do_not_consume_tool_call_ids() -> None:
    """模型决策 / 回复也会走 step_added，但它们不是工具调用，不该占号。"""
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.step_added(_tool_step(kind="plan", title="模型决策", tool_call_id=None))
    sink.tool_calls_planned(
        [{"id": "raw-1", "name": "read_file", "title": "执行 read_file(a.txt)"}]
    )

    assert [u["toolCallId"] for u in updates] == ["call_1"]


def test_sink_marks_leftover_calls_failed_when_the_turn_gives_up() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.tool_calls_planned(
        [
            {"id": "raw-1", "name": "read_file", "title": "执行 read_file(a.txt)"},
            {"id": "raw-2", "name": "grep", "title": "执行 grep(x)"},
        ]
    )
    sink.tool_started("raw-1", "read_file")
    sink.step_added(_tool_step(tool_call_id="raw-1"))
    # raw-2 被取消，再也不会有人给它收尾
    sink.abandon_open_tool_calls()
    sink.abandon_open_tool_calls()

    failed = [u for u in updates if u.get("status") == "failed"]
    assert [u["toolCallId"] for u in failed] == ["call_2"]


def test_sink_abandon_survives_a_closed_connection() -> None:
    def broken(update: dict[str, Any]) -> None:
        if update.get("status") == "failed":
            raise OSError("连接已关闭")

    sink = AcpTraceSink(broken)
    sink.tool_calls_planned(
        [{"id": "raw-1", "name": "read_file", "title": "执行 read_file(a.txt)"}]
    )
    sink.abandon_open_tool_calls()


# ---- trace 那两个来源 ----


def test_planned_notification_carries_ids_titles_and_skips_subagents() -> None:
    session = TraceSession()
    sink = _FakeSink()
    session.trace_sink = sink
    printer = TurnTracePrinter(session)

    printer.on_agent_update(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "a.txt"}, "id": "raw-1"},
                    {"name": "spawn_subagent", "args": {"task": "查一下"}, "id": "raw-2"},
                    {"name": "grep", "args": {}, "id": ""},
                ],
            )
        ]
    )

    assert len(sink.planned) == 1
    planned = sink.planned[0]
    # spawn_subagent 的完成态走 explore 步骤，报了 pending 就再没人收尾；无 id 的也发不出更新
    assert [c["id"] for c in planned] == ["raw-1"]
    assert planned[0]["name"] == "read_file"
    assert "read_file" in planned[0]["title"]
    assert "a.txt" in planned[0]["title"]


def test_verbose_mode_announces_nothing() -> None:
    """trace all 下工具步骤按行打印、不按工具登记，报 pending 就没人收尾。"""
    from llgraph.display.trace_display import TraceMode

    session = TraceSession(mode=TraceMode.ALL)
    sink = _FakeSink()
    session.trace_sink = sink
    printer = TurnTracePrinter(session)

    printer.on_agent_update(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "a.txt"}, "id": "raw-1"}],
            )
        ]
    )

    assert sink.planned == []


def test_reply_turn_notifies_nothing() -> None:
    session = TraceSession()
    sink = _FakeSink()
    session.trace_sink = sink
    printer = TurnTracePrinter(session)

    printer.on_agent_update([AIMessage(content="就这样，没有工具要调。")])

    assert sink.planned == []


def test_tool_step_records_the_model_tool_call_id() -> None:
    session = TraceSession()
    sink = _FakeSink()
    session.trace_sink = sink
    printer = TurnTracePrinter(session)

    printer.on_tools_update(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "a.txt"}, "id": "raw-1"}
                ],
            ),
            ToolMessage(content="ok", name="read_file", tool_call_id="raw-1"),
        ]
    )

    tool_steps = [s for s in sink.steps if s.kind == "tool"]
    assert [s.tool_call_id for s in tool_steps] == ["raw-1"]
