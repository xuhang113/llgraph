"""ACP 计划清单：``todo_write`` 的待办推成编辑器里的 ``plan``。

清单此前只有 CLI / Web Console 看得到，编辑器里一条都不显示——模型按一张表干活，
用户却看不见那张表。这里钉住三件事：落盘那一刻报出整张表、载荷按 ACP 的
``PlanEntry`` 成形（三档状态 + 必填 priority）、以及续聊时把清单接回来。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.context.runtime_context import set_active_thread_id
from llgraph.core.todo_store import load_todo_state, save_todo_state
from llgraph.core.tool_progress import (
    PlanItem,
    current_plan_observer,
    notify_plan_updated,
    use_plan_observer,
)
from llgraph.core.todo_tools import create_todo_tools
from llgraph.editor.acp.replay import plan_update
from llgraph.editor.acp.sink import AcpTraceSink
from llgraph.editor.acp.updates import (
    CANCELLED_PLAN_PREFIX,
    MAX_PLAN_ENTRIES,
    plan_entries,
    session_plan,
)


def _todo_write(workspace: Path):
    """@param workspace 工作区根 @return todo_write 的可调用体"""
    tool = create_todo_tools(workspace)[0]
    return tool.func


def _plans(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [u for u in updates if u.get("sessionUpdate") == "plan"]


# ---- 载荷成形 ----


def test_plan_entries_map_the_four_statuses_into_acp_three() -> None:
    entries = plan_entries(
        [
            {"content": "定位入口", "status": "completed"},
            {"content": "改 sink", "status": "in_progress"},
            {"content": "跑测试", "status": "pending"},
        ]
    )
    assert [e["status"] for e in entries] == ["completed", "in_progress", "pending"]
    assert [e["content"] for e in entries] == ["定位入口", "改 sink", "跑测试"]
    # priority 是 ACP 必填项，llgraph 没有这个字段，一律中档（不凭空造高低）
    assert {e["priority"] for e in entries} == {"medium"}


def test_cancelled_shows_as_done_but_says_so() -> None:
    """ACP 没有 cancelled：留成 pending 会让那张表永远显示「还有活没干」。"""
    entries = plan_entries([{"content": "顺手加缓存", "status": "cancelled"}])
    assert entries[0]["status"] == "completed"
    assert entries[0]["content"] == f"{CANCELLED_PLAN_PREFIX}顺手加缓存"


def test_plan_entries_drop_junk_and_cap_the_table() -> None:
    assert plan_entries(None) == []
    assert plan_entries("t1") == []
    assert plan_entries([None, 7, {"content": "   "}, {"status": "pending"}]) == []
    # 认不出的状态按待做算，而不是丢掉这一条
    assert plan_entries([{"content": "x", "status": "???"}])[0]["status"] == "pending"
    # 正文里的换行折成一行：编辑器里一条就是一行
    assert plan_entries([{"content": "读\n  改\n跑", "status": "pending"}])[0][
        "content"
    ] == "读 改 跑"
    many = [{"content": f"第 {i} 步", "status": "pending"} for i in range(MAX_PLAN_ENTRIES + 5)]
    assert len(plan_entries(many)) == MAX_PLAN_ENTRIES


def test_session_plan_is_a_full_snapshot() -> None:
    payload = session_plan([{"content": "一步", "status": "pending"}])
    assert payload["sessionUpdate"] == "plan"
    assert payload["entries"] == [
        {"content": "一步", "priority": "medium", "status": "pending"}
    ]
    # 清空清单发的是空快照，而不是什么都不发：那张表该跟着清掉
    assert session_plan([]) == {"sessionUpdate": "plan", "entries": []}


# ---- 通知来源 ----


def test_plan_observer_scope_and_silence() -> None:
    assert current_plan_observer() is None
    seen: list[list[PlanItem]] = []
    with use_plan_observer(seen.append):
        assert current_plan_observer() is not None
        notify_plan_updated([PlanItem(id="t1", content="一步", status="pending")])
        # 空表也要报：清空清单同样是一次变化
        notify_plan_updated([])
    assert current_plan_observer() is None
    # 作用域外没人收，也不许抛
    notify_plan_updated([PlanItem(id="t1", content="一步", status="pending")])
    assert [len(items) for items in seen] == [1, 0]
    assert seen[0][0].content == "一步"


def test_plan_observer_exception_does_not_break_the_tool() -> None:
    def boom(items: list[PlanItem]) -> None:
        raise RuntimeError("观察者炸了")

    with use_plan_observer(boom):
        notify_plan_updated([PlanItem(id="t1", content="一步", status="pending")])


def test_todo_write_reports_the_whole_table(tmp_path: Path) -> None:
    """落盘之后报一次，给的是整张表（不是本次提交的那几条）。"""
    set_active_thread_id("cli-acpplan1")
    try:
        write = _todo_write(tmp_path)
        seen: list[list[PlanItem]] = []
        with use_plan_observer(seen.append):
            write(todos=[{"content": "定位入口", "status": "in_progress"}], merge=False)
            write(
                todos=[
                    {"id": "t1", "content": "定位入口", "status": "completed"},
                    {"content": "跑测试", "status": "in_progress"},
                ],
                merge=True,
            )
        assert [len(items) for items in seen] == [1, 2]
        assert [(i.id, i.status) for i in seen[1]] == [
            ("t1", "completed"),
            ("t2", "in_progress"),
        ]
        # 报的和落盘的是同一份
        state = load_todo_state(tmp_path, "cli-acpplan1")
        assert [(i.id, i.status) for i in state.todos] == [
            (i.id, i.status) for i in seen[1]
        ]
    finally:
        set_active_thread_id(None)


def test_todo_write_without_changes_reports_nothing(tmp_path: Path) -> None:
    """``merge=true`` 不给 todos 只是查一眼当前清单，没落盘就不该报。"""
    set_active_thread_id("cli-acpplan2")
    try:
        write = _todo_write(tmp_path)
        write(todos=[{"content": "一步", "status": "pending"}], merge=False)
        seen: list[list[PlanItem]] = []
        with use_plan_observer(seen.append):
            write(merge=True)
        assert seen == []
    finally:
        set_active_thread_id(None)


def test_todo_write_clears_the_plan(tmp_path: Path) -> None:
    set_active_thread_id("cli-acpplan3")
    try:
        write = _todo_write(tmp_path)
        write(todos=[{"content": "一步", "status": "pending"}], merge=False)
        seen: list[list[PlanItem]] = []
        with use_plan_observer(seen.append):
            write(todos=[], merge=False)
        assert seen == [[]]
    finally:
        set_active_thread_id(None)


def test_no_observer_means_cli_behaviour_is_unchanged(tmp_path: Path) -> None:
    set_active_thread_id("cli-acpplan4")
    try:
        result = _todo_write(tmp_path)(
            todos=[{"content": "一步", "status": "pending"}], merge=False
        )
        assert "任务清单" in result
        assert load_todo_state(tmp_path, "cli-acpplan4").todos
    finally:
        set_active_thread_id(None)


# ---- Sink ----


def test_sink_sends_the_plan_on_every_change() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.todo_plan([PlanItem(id="t1", content="定位入口", status="in_progress")])
    sink.todo_plan(
        [
            PlanItem(id="t1", content="定位入口", status="completed"),
            PlanItem(id="t2", content="跑测试", status="in_progress"),
        ]
    )
    plans = _plans(updates)
    assert len(plans) == 2
    assert [e["status"] for e in plans[0]["entries"]] == ["in_progress"]
    assert [e["status"] for e in plans[1]["entries"]] == ["completed", "in_progress"]


def test_sink_skips_an_identical_table() -> None:
    """模型原样写回同一份清单时不必让编辑器白刷一次。"""
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)
    items = [PlanItem(id="t1", content="一步", status="pending")]

    sink.todo_plan(items)
    sink.todo_plan(list(items))
    # id 变了但正文与状态没变：ACP 的 plan 里本来就没有 id，这仍是同一张表
    sink.todo_plan([PlanItem(id="t9", content="一步", status="pending")])
    assert len(_plans(updates)) == 1

    sink.todo_plan([PlanItem(id="t1", content="一步", status="completed")])
    assert len(_plans(updates)) == 2


def test_sink_sends_an_empty_plan_once() -> None:
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append)

    sink.todo_plan([PlanItem(id="t1", content="一步", status="pending")])
    sink.todo_plan([])
    sink.todo_plan([])
    plans = _plans(updates)
    assert len(plans) == 2
    assert plans[1]["entries"] == []


def test_sink_plan_does_not_touch_tool_call_rows() -> None:
    """计划是会话级的一张表，不挂在任何一行工具调用上。"""
    updates: list[dict[str, Any]] = []
    sink = AcpTraceSink(updates.append, id_prefix="t1_")

    sink.tool_calls_planned([{"id": "call_a", "name": "todo_write", "title": "执行 todo_write"}])
    sink.todo_plan([PlanItem(id="t1", content="一步", status="pending")])
    plan = _plans(updates)[0]
    assert "toolCallId" not in plan
    assert "status" not in plan
    # 工具那一行照旧走自己的三段状态
    assert updates[0]["sessionUpdate"] == "tool_call"


# ---- 续聊 ----


def test_loading_a_session_brings_the_plan_back(tmp_path: Path) -> None:
    from llgraph.core.todo_store import TodoItem, TodoState

    save_todo_state(
        tmp_path,
        "cli-acpplan5",
        TodoState(
            todos=[
                TodoItem(id="t1", content="定位入口", status="completed"),
                TodoItem(id="t2", content="改 sink", status="in_progress"),
            ]
        ),
    )
    payloads = plan_update(tmp_path, "cli-acpplan5")
    assert len(payloads) == 1
    assert [e["content"] for e in payloads[0]["entries"]] == ["定位入口", "改 sink"]


def test_loading_a_session_without_a_plan_sends_nothing(tmp_path: Path) -> None:
    assert plan_update(tmp_path, "cli-acpplan6") == []


def test_load_session_updates_appends_the_plan_after_history(tmp_path: Path) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from llgraph.core.todo_store import TodoItem, TodoState
    from llgraph.editor.acp.replay import load_session_updates
    from llgraph.session.session_file_store import save_session_messages

    workspace = tmp_path / "ws"
    workspace.mkdir()
    save_session_messages(
        workspace,
        "cli-acpplan7",
        [HumanMessage(content="接着改"), AIMessage(content="好")],
    )
    save_todo_state(
        workspace,
        "cli-acpplan7",
        TodoState(todos=[TodoItem(id="t1", content="改 sink", status="in_progress")]),
    )

    updates = load_session_updates(workspace, "cli-acpplan7")
    kinds = [u["sessionUpdate"] for u in updates]
    assert kinds[-1] == "plan"
    assert "user_message_chunk" in kinds
    assert updates[-1]["entries"][0]["status"] == "in_progress"
