"""工具跑完那一下：失败态（failed）与受影响文件（locations）。

工具不抛异常也可能是失败的（参数校验错、old_string 没匹配上），一律报 completed
编辑器里就看不出来；locations 则决定那一行能不能点开跳到文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from llgraph.display.trace_display import TraceSession, TurnTracePrinter
from llgraph.editor.acp.sink import AcpTraceSink
from llgraph.editor.acp.updates import (
    tool_call_from_step,
    tool_call_locations,
    tool_call_pending,
)


class _FakeSink:
    """只收工具相关事件的 Sink。"""

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


def _printer() -> tuple[TurnTracePrinter, _FakeSink]:
    session = TraceSession()
    sink = _FakeSink()
    session.trace_sink = sink
    return TurnTracePrinter(session), sink


def _tool_step(**kwargs: Any) -> Any:
    from llgraph.display.trace_display import TraceStepRecord

    base: dict[str, Any] = {
        "step_id": 1,
        "kind": "tool",
        "title": "执行 search_replace(a.txt)",
        "elapsed": 0.2,
        "summary": "1 行输出",
        "body_lines": ["错误: 未找到 old_string"],
    }
    base.update(kwargs)
    return TraceStepRecord(**base)


# ---- trace 侧：步骤记下「这次像是失败了」 ----


def test_error_looking_tool_output_is_recorded_on_the_step() -> None:
    printer, sink = _printer()

    printer.on_tools_update(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_replace", "args": {"path": "a.txt"}, "id": "raw-1"}
                ],
            ),
            ToolMessage(
                content="错误: search_replace 未找到 old_string",
                name="search_replace",
                tool_call_id="raw-1",
            ),
        ]
    )

    tool_steps = [s for s in sink.steps if s.kind == "tool"]
    assert [s.tool_failed for s in tool_steps] == [True]


def test_normal_tool_output_is_not_marked_failed() -> None:
    printer, sink = _printer()

    printer.on_tools_update(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "a.txt"}, "id": "raw-1"}
                ],
            ),
            ToolMessage(content="hello", name="read_file", tool_call_id="raw-1"),
        ]
    )

    tool_steps = [s for s in sink.steps if s.kind == "tool"]
    assert [s.tool_failed for s in tool_steps] == [False]


# ---- trace 侧：受影响文件从参数里取 ----


def test_planned_notification_carries_the_files_a_call_touches() -> None:
    printer, sink = _printer()

    printer.on_agent_update(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "write_file", "args": {"path": "src/a.py"}, "id": "raw-1"},
                    {
                        "name": "read_files",
                        "args": {"paths": ["b.py", "c.py", "b.py"]},
                        "id": "raw-2",
                    },
                    # 搜索类的 path 是扫描范围，不是「这次动的文件」
                    {
                        "name": "grep_files",
                        "args": {"pattern": "x", "path": "src"},
                        "id": "raw-3",
                    },
                    {"name": "run_shell_command", "args": {"command": "ls"}, "id": "raw-4"},
                ],
            )
        ]
    )

    by_id = {c["id"]: c["paths"] for c in sink.planned[0]}
    assert by_id["raw-1"] == ["src/a.py"]
    # 同一个文件报两次，编辑器里就会挂两条一样的跳转
    assert by_id["raw-2"] == ["b.py", "c.py"]
    assert by_id["raw-3"] == []
    assert by_id["raw-4"] == []


def test_workspace_root_is_not_a_location() -> None:
    """path="." 指的是整个工作区，点开它没有意义。"""
    printer, sink = _printer()

    printer.on_agent_update(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "."}, "id": "raw-1"}],
            )
        ]
    )

    assert sink.planned[0][0]["paths"] == []


# ---- 载荷：failed 与 locations ----


def test_step_that_looks_like_an_error_is_reported_failed() -> None:
    step = {
        "kind": "tool",
        "title": "执行 search_replace(a.txt)",
        "body_lines": ["错误: 未找到 old_string"],
        "tool_failed": True,
    }

    created = tool_call_from_step(step, tool_call_id="call_1")
    updated = tool_call_from_step(step, tool_call_id="call_1", as_update=True)

    assert created["status"] == "failed"
    assert updated["status"] == "failed"
    # 失败也要带上输出：编辑器里那行点开就是失败原因
    assert "old_string" in created["content"][0]["content"]["text"]


def test_step_without_the_failure_flag_stays_completed() -> None:
    step = {"kind": "tool", "title": "执行 read_file(a.txt)", "body_lines": ["ok"]}

    assert tool_call_from_step(step, tool_call_id="call_1")["status"] == "completed"


def test_locations_are_absolute(tmp_path: Path) -> None:
    locations = tool_call_locations(
        ["src/a.py", str(tmp_path / "b.py"), "src/a.py"], tmp_path
    )

    assert locations == [
        {"path": str(tmp_path / "src" / "a.py")},
        {"path": str(tmp_path / "b.py")},
    ]


def test_relative_locations_are_dropped_without_a_workspace(tmp_path: Path) -> None:
    """ACP 只认绝对路径；补不出来的宁可不报，免得编辑器点下去是个报错。"""
    assert tool_call_locations(["src/a.py"], None) == []
    assert tool_call_locations([str(tmp_path / "b.py")], None) == [
        {"path": str(tmp_path / "b.py")}
    ]


def test_locations_ignore_junk_input() -> None:
    assert tool_call_locations(None, Path("/ws")) == []
    assert tool_call_locations([" ", 3, None], Path("/ws")) == []


def test_pending_payload_omits_empty_locations() -> None:
    payload = tool_call_pending("call_1", title="执行 ls", tool_name="list_directory")

    assert "locations" not in payload


# ---- Sink：把两件事串起来 ----


def test_sink_puts_absolute_locations_on_the_pending_call(tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append, workspace=tmp_path)

    sink.tool_calls_planned(
        [
            {
                "id": "raw-1",
                "name": "write_file",
                "title": "执行 write_file(src/a.py)",
                "paths": ["src/a.py"],
            }
        ]
    )

    assert updates[0]["locations"] == [{"path": str(tmp_path / "src" / "a.py")}]


def test_sink_finishes_a_failed_call_as_failed(tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append, workspace=tmp_path)

    sink.tool_calls_planned(
        [
            {
                "id": "raw-1",
                "name": "search_replace",
                "title": "执行 search_replace(a.txt)",
                "paths": ["a.txt"],
            }
        ]
    )
    sink.tool_started("raw-1", "search_replace")
    sink.step_added(_tool_step(tool_call_id="raw-1", tool_failed=True))

    assert [u["status"] for u in updates] == ["pending", "in_progress", "failed"]
    # 失败也要收尾在同一行上，且不该再被收场逻辑标一次
    assert len({u["toolCallId"] for u in updates}) == 1
    sink.abandon_open_tool_calls()
    assert len(updates) == 3


def test_sink_without_a_workspace_skips_relative_locations() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.tool_calls_planned(
        [
            {
                "id": "raw-1",
                "name": "read_file",
                "title": "执行 read_file(a.txt)",
                "paths": ["a.txt"],
            }
        ]
    )

    assert "locations" not in updates[0]
