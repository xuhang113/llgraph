"""Plan 子图 visible 结构化 END 路由。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from llgraph.core.agent_turn import FALLBACK_INCOMPLETE_TURN, route_after_agent
from llgraph.plan.subgraphs.routing import (
    extract_structured_deliverable_text,
    planner_deliverable_complete,
)


def test_thinking_only_does_not_deliver() -> None:
    plan_json = (
        '{"title": "促销", "tasks": [{"id": "w1", "title": "a"}, {"id": "w2", "title": "b"}]}'
    )
    msg = AIMessage(content=[{"type": "thinking", "thinking": f"```json\n{plan_json}\n```"}])
    messages = [HumanMessage(content="goal"), msg]
    assert not planner_deliverable_complete(msg)
    assert extract_structured_deliverable_text(messages, subgraph_kind="planner") == ""
    state = {"messages": messages, "remaining_steps": 10}
    assert route_after_agent(state, complete_on_thinking_if=planner_deliverable_complete) == "think_nudge"


def test_llgraph_thinking_meta_does_not_deliver() -> None:
    plan_json = (
        '{"title": "促销", "tasks": [{"id": "w1", "title": "a"}, {"id": "w2", "title": "b"}]}'
    )
    msg = AIMessage(
        content="",
        additional_kwargs={"llgraph": {"thinking_text": f"```json\n{plan_json}\n```"}},
    )
    assert not planner_deliverable_complete(msg)


def test_visible_json_delivers() -> None:
    plan_json = (
        '{"title": "促销", "tasks": [{"id": "w1", "title": "a"}, {"id": "w2", "title": "b"}]}'
    )
    msg = AIMessage(
        content=[{"type": "text", "text": f"计划如下：\n```json\n{plan_json}\n```"}],
    )
    messages = [HumanMessage(content="goal"), msg]
    assert planner_deliverable_complete(msg)
    text = extract_structured_deliverable_text(messages, subgraph_kind="planner")
    assert "w1" in text
    state = {"messages": messages, "remaining_steps": 2}
    assert route_after_agent(state, complete_on_thinking_if=planner_deliverable_complete) == "__end__"


def test_fallback_visible_skipped() -> None:
    plan_json = (
        '{"title": "促销", "tasks": [{"id": "w1", "title": "a"}, {"id": "w2", "title": "b"}]}'
    )
    good = AIMessage(
        content=[{"type": "text", "text": f"```json\n{plan_json}\n```"}],
    )
    bad = AIMessage(content=FALLBACK_INCOMPLETE_TURN)
    messages = [HumanMessage(content="goal"), good, bad]
    text = extract_structured_deliverable_text(messages, subgraph_kind="planner")
    assert "w1" in text
    assert not planner_deliverable_complete(bad)
