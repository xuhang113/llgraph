"""agent_turn 路由谓词单测。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.core.agent_turn import (
    ai_message_has_visible_text,
    pending_tool_calls,
    route_after_agent,
)


def test_plan_line_only_loops_agent() -> None:
    msg = AIMessage(content="【规划】先 grep 再 read。")
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 10}
    assert not ai_message_has_visible_text(msg)
    assert route_after_agent(state) == "think_nudge"


def test_visible_text_only_ends_turn() -> None:
    msg = AIMessage(content=[{"type": "text", "text": "你好"}])
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 10}
    assert route_after_agent(state) == "__end__"
    assert ai_message_has_visible_text(msg)


def test_thinking_only_loops_agent() -> None:
    msg = AIMessage(content=[{"type": "thinking", "thinking": "内部推理"}])
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 10}
    assert route_after_agent(state) == "think_nudge"
    assert not ai_message_has_visible_text(msg)


def test_tool_calls_route_to_tools() -> None:
    msg = AIMessage(
        content=[{"type": "thinking", "thinking": "plan"}],
        tool_calls=[{"id": "c1", "name": "read_file", "args": {"path": "a.txt"}}],
    )
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 10}
    assert route_after_agent(state) == "tools"
    assert pending_tool_calls(state["messages"])  # type: ignore[arg-type]


def test_pending_tool_after_partial_results() -> None:
    ai = AIMessage(
        content="",
        tool_calls=[
            {"id": "c1", "name": "read_file", "args": {}},
            {"id": "c2", "name": "grep_files", "args": {}},
        ],
    )
    done = ToolMessage(content="ok", tool_call_id="c1", name="read_file")
    state = {"messages": [ai, done], "remaining_steps": 5}
    pending = pending_tool_calls(state["messages"])  # type: ignore[arg-type]
    assert len(pending) == 1
    assert pending[0]["id"] == "c2"


def test_reused_tool_call_id_on_new_ai_still_routes_to_tools() -> None:
    """Kimi 常复用 functions.grep_files:1；新 AI 上的同 id 仍应 pending。"""
    earlier_ai = AIMessage(
        content="",
        tool_calls=[{"id": "functions.grep_files:1", "name": "grep_files", "args": {"pattern": "a"}}],
    )
    earlier_tool = ToolMessage(
        content="hits-a",
        tool_call_id="functions.grep_files:1",
        name="grep_files",
    )
    latest_ai = AIMessage(
        content="【规划】继续检索过滤逻辑。",
        tool_calls=[{"id": "functions.grep_files:1", "name": "grep_files", "args": {"pattern": "b"}}],
    )
    messages = [
        HumanMessage(content="q"),
        earlier_ai,
        earlier_tool,
        latest_ai,
    ]
    pending = pending_tool_calls(messages)
    assert len(pending) == 1
    assert pending[0]["args"]["pattern"] == "b"
    assert route_after_agent({"messages": messages, "remaining_steps": 10}) == "tools"


def test_thinking_only_exhausted_steps_goes_fallback() -> None:
    msg = AIMessage(content=[{"type": "thinking", "thinking": "still thinking"}])
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 2}
    assert route_after_agent(state) == "turn_fallback"
    state["remaining_steps"] = 1
    assert route_after_agent(state) == "turn_fallback"


def test_planner_visible_json_ends_turn() -> None:
    from llgraph.plan.subgraphs.routing import planner_deliverable_complete

    plan_json = (
        '{"title": "T", "tasks": [{"id": "w1", "title": "a"}, {"id": "w2", "title": "b"}]}'
    )
    msg = AIMessage(
        content=[{"type": "text", "text": f"```json\n{plan_json}\n```"}],
    )
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 2}
    assert planner_deliverable_complete(msg)
    assert route_after_agent(state, complete_on_thinking_if=planner_deliverable_complete) == "__end__"


def test_planner_thinking_json_does_not_end_turn() -> None:
    from llgraph.plan.subgraphs.routing import planner_deliverable_complete

    plan_json = (
        '{"title": "T", "tasks": [{"id": "w1", "title": "a"}, {"id": "w2", "title": "b"}]}'
    )
    msg = AIMessage(content=[{"type": "thinking", "thinking": f"plan:\n```json\n{plan_json}\n```"}])
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 10}
    assert not planner_deliverable_complete(msg)
    assert route_after_agent(state, complete_on_thinking_if=planner_deliverable_complete) == "think_nudge"


def test_planner_thinking_without_json_still_loops() -> None:
    from llgraph.plan.subgraphs.routing import planner_deliverable_complete

    msg = AIMessage(content=[{"type": "thinking", "thinking": "还在调研，没有 JSON"}])
    state = {"messages": [HumanMessage(content="hi"), msg], "remaining_steps": 10}
    assert not planner_deliverable_complete(msg)
    assert route_after_agent(state, complete_on_thinking_if=planner_deliverable_complete) == "think_nudge"
