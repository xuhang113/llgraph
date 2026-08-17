"""ReAct 步间批量提醒单测。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.react_step_reminder import (
    append_react_step_reminder_for_dispatch,
    should_inject_react_step_reminder,
)


def _ai_with_tools(n: int) -> AIMessage:
    calls = [{"id": f"c{i}", "name": "grep_files", "args": {"pattern": "x"}} for i in range(n)]
    return AIMessage(content="", tool_calls=calls)


def test_no_reminder_before_first_tool_round():
    msgs = [
        HumanMessage(content="查 tb_foo 和 bar_id"),
        _ai_with_tools(1),
    ]
    assert should_inject_react_step_reminder(msgs) is False


def test_reminder_after_tool_returns():
    msgs = [
        HumanMessage(content="查表"),
        _ai_with_tools(1),
        ToolMessage(content="hit", tool_call_id="c0", name="grep_files"),
    ]
    assert should_inject_react_step_reminder(msgs) is True


def test_single_tool_gets_stronger_nudge():
    msgs = [
        HumanMessage(content="查表"),
        _ai_with_tools(1),
        ToolMessage(content="hit", tool_call_id="c0", name="grep_files"),
    ]
    out = append_react_step_reminder_for_dispatch(msgs)
    assert len(out) == len(msgs) + 1
    text = str(out[-1].content)
    assert "<system-reminder>" in text
    assert "工具往返预算：" in text
    assert "仅 1 个 tool_call" in text or "1 个工具" in text


def test_multi_tool_gets_general_reminder():
    msgs = [
        HumanMessage(content="查表"),
        _ai_with_tools(3),
        ToolMessage(content="a", tool_call_id="c0", name="grep_files"),
        ToolMessage(content="b", tool_call_id="c1", name="grep_files"),
        ToolMessage(content="c", tool_call_id="c2", name="grep_files"),
    ]
    out = append_react_step_reminder_for_dispatch(msgs)
    text = str(out[-1].content)
    assert "<system-reminder>" in text
    assert "工具往返预算：" in text
    assert "批量" in text or "终答" in text


def test_budget_injected_even_without_tool_round():
    msgs = [
        HumanMessage(content="你好"),
        AIMessage(content="完成"),
    ]
    assert should_inject_react_step_reminder(msgs) is False
    out = append_react_step_reminder_for_dispatch(msgs)
    assert "工具往返预算：" in str(out[-1].content)
    assert "已用 0" in str(out[-1].content)


def test_investigate_query_gets_same_generic_reminder():
    """排查口吻不再注入归因/提纲分流文案。"""
    msgs = [
        HumanMessage(content="滚动条变小且确认后无法删除，是前端bug吗"),
        _ai_with_tools(1),
        ToolMessage(content="hit", tool_call_id="c0", name="grep_files"),
    ]
    out = append_react_step_reminder_for_dispatch(msgs)
    text = str(out[-1].content)
    assert "工具往返预算：" in text
    assert "归因提示" not in text
    assert "排查提示" not in text
    assert "提纲" not in text


def test_no_reminder_when_ending_with_ai_text():
    msgs = [
        HumanMessage(content="你好"),
        AIMessage(content="完成"),
    ]
    assert should_inject_react_step_reminder(msgs) is False
