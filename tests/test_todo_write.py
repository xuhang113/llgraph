"""会话任务清单：对标 Cursor / Claude Code TodoWrite。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.context_builder import build_workspace_context_block
from llgraph.context.investigate_harness import is_ephemeral_harness_human
from llgraph.context.context_session import ContextSession
from llgraph.context.runtime_context import set_active_thread_id
from llgraph.core.agent_turn import route_after_agent
from llgraph.core.todo_store import (
    TODO_NUDGE_MARKER,
    TODO_TOOL_NAME,
    apply_todo_write,
    format_todo_nudge,
    format_todo_tool_result,
    load_todo_state,
    parse_todo_inputs,
    save_todo_state,
    should_nudge_open_todos,
)
from llgraph.core.todo_tools import create_todo_tools
from llgraph.session.user_storage import session_todos_path


def test_merge_upsert_and_single_in_progress(tmp_path: Path) -> None:
    first, _ = apply_todo_write(
        load_todo_state(tmp_path, "cli-test"),
        parse_todo_inputs(
            [
                {"id": "t1", "content": "定位入口", "status": "completed"},
                {"id": "t2", "content": "改 search_replace", "status": "in_progress"},
                {"id": "t3", "content": "跑测试", "status": "pending"},
            ]
        ),
        merge=False,
    )
    save_todo_state(tmp_path, "cli-test", first)
    assert session_todos_path(tmp_path, "cli-test").is_file()

    second, notes = apply_todo_write(
        first,
        parse_todo_inputs(
            [
                {"id": "t2", "content": "改 search_replace", "status": "completed"},
                {"content": "跑测试", "status": "in_progress"},
                {"content": "更新提示词", "status": "in_progress"},
            ]
        ),
        merge=True,
    )
    assert [item.id for item in second.todos] == ["t1", "t2", "t3", "t4"]
    in_prog = [item for item in second.todos if item.status == "in_progress"]
    assert len(in_prog) == 1
    assert in_prog[0].id == "t4"
    assert any("in_progress" in n for n in notes)
    assert second.open_items()[0].id == "t3"


def test_replace_clears_and_tool_roundtrip(tmp_path: Path) -> None:
    set_active_thread_id("cli-tool")
    tools = create_todo_tools(tmp_path)
    fn = tools[0].func
    out = fn(
        todos=[
            {"content": "读文件", "status": "completed"},
            {"content": "提交补丁", "status": "pending"},
        ],
        merge=False,
    )
    assert "1/2 完成" in out
    state = load_todo_state(tmp_path, "cli-tool")
    assert len(state.todos) == 2
    cleared = fn(todos=[], merge=False)
    assert "已清空" in cleared or "0/0" in format_todo_tool_result(
        load_todo_state(tmp_path, "cli-tool"), []
    )
    assert load_todo_state(tmp_path, "cli-tool").todos == []
    set_active_thread_id(None)


def test_workspace_context_pins_todos_and_edits(tmp_path: Path) -> None:
    set_active_thread_id("cli-ctx")
    state, _ = apply_todo_write(
        load_todo_state(tmp_path, "cli-ctx"),
        parse_todo_inputs([{"content": "修诊断回注", "status": "in_progress"}]),
        merge=False,
    )
    save_todo_state(tmp_path, "cli-ctx", state)
    block = build_workspace_context_block(
        tmp_path,
        ContextSession(),
        "继续改",
        allow_write=True,
        recent_messages=[],
        edited_paths=["llgraph/core/edit_apply.py"],
    )
    assert "本轮任务清单" in block
    assert "修诊断回注" in block
    assert "本会话已改文件" in block
    assert "edit_apply.py" in block
    set_active_thread_id(None)


def test_open_todos_nudge_after_todo_write(tmp_path: Path) -> None:
    set_active_thread_id("cli-nudge")
    state, _ = apply_todo_write(
        load_todo_state(tmp_path, "cli-nudge"),
        parse_todo_inputs(
            [
                {"id": "t1", "content": "改代码", "status": "in_progress"},
                {"id": "t2", "content": "跑测试", "status": "pending"},
            ]
        ),
        merge=False,
    )
    save_todo_state(tmp_path, "cli-nudge", state)
    messages = [
        HumanMessage(content="把编辑诊断接到写工具上"),
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": TODO_TOOL_NAME, "args": {}}],
        ),
        ToolMessage(content="ok", tool_call_id="c1", name=TODO_TOOL_NAME),
        AIMessage(content="已经改完了。"),
    ]
    assert should_nudge_open_todos(messages, tmp_path, remaining_steps=10)
    assert route_after_agent(
        {"messages": messages, "remaining_steps": 10},
        workspace=tmp_path,
    ) == "todo_nudge"
    nudged = [*messages, HumanMessage(content=format_todo_nudge(tmp_path))]
    assert TODO_NUDGE_MARKER in nudged[-1].content
    assert is_ephemeral_harness_human(nudged[-1])
    assert not should_nudge_open_todos(nudged, tmp_path, remaining_steps=10)
    assert route_after_agent(
        {"messages": nudged, "remaining_steps": 10},
        workspace=tmp_path,
    ) == "__end__"
    set_active_thread_id(None)


def test_no_nudge_without_todo_write_this_turn(tmp_path: Path) -> None:
    set_active_thread_id("cli-plain")
    state, _ = apply_todo_write(
        load_todo_state(tmp_path, "cli-plain"),
        parse_todo_inputs([{"content": "旧任务", "status": "pending"}]),
        merge=False,
    )
    save_todo_state(tmp_path, "cli-plain", state)
    messages = [
        HumanMessage(content="现在几点？"),
        AIMessage(content="现在是下午。"),
    ]
    assert not should_nudge_open_todos(messages, tmp_path, remaining_steps=10)
    assert route_after_agent(
        {"messages": messages, "remaining_steps": 10},
        workspace=tmp_path,
    ) == "__end__"
    set_active_thread_id(None)
