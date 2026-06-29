"""Plan checkpoint 与 parallel search 守卫测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.core.react_tools import guard_parallel_search_code_parallel
from llgraph.plan.runner import _merge_checkpoint_state
from llgraph.plan.state import PlanPhase


def test_plan_cancel_not_revived_by_checkpoint() -> None:
    disk_state = {
        "phase": PlanPhase.CANCELLED,
        "cancel_requested": True,
        "parallel_batch": [],
        "workflow_snapshot": {"nodes": []},
        "plan": {"plan_id": "p1", "phase": PlanPhase.CANCELLED, "tasks": []},
    }
    checkpoint_state = {
        "phase": "executing",
        "cancel_requested": False,
        "parallel_batch": ["w1", "w2"],
        "workflow_snapshot": {"nodes": [{"id": "w1", "status": "running"}]},
        "plan": {"plan_id": "p1", "phase": "executing", "tasks": [{"id": "w1", "status": "running"}]},
    }
    graph = MagicMock()
    graph.get_state.return_value = MagicMock(values=checkpoint_state)

    ctx = MagicMock()
    merged = _merge_checkpoint_state(ctx, graph, disk_state)

    assert merged["phase"] == PlanPhase.CANCELLED
    assert merged["cancel_requested"] is True
    assert merged["plan"]["phase"] == PlanPhase.CANCELLED
    assert merged["parallel_batch"] == []
    assert merged["workflow_snapshot"] == {"nodes": []}


def test_parallel_search_code_parallel_batch_guard() -> None:
    ai = AIMessage(
        content="",
        tool_calls=[
            {"id": "p1", "name": "search_code_parallel", "args": {"query": "foo bar baz qux quux"}},
            {"id": "p2", "name": "search_code_parallel", "args": {"query": "other"}},
            {"id": "g1", "name": "grep_files", "args": {"pattern": "Foo", "path": "."}},
        ],
    )
    state = {"messages": [HumanMessage(content="hi"), ai]}
    guarded, blocked = guard_parallel_search_code_parallel(state)

    msgs = guarded["messages"]
    assert isinstance(msgs[-1], AIMessage)
    kept = msgs[-1].tool_calls or []
    names = [c["name"] for c in kept]
    assert names.count("search_code_parallel") == 1
    assert "grep_files" in names
    assert len(blocked) == 1
    assert blocked[0].name == "search_code_parallel"
    assert "拦截" in str(blocked[0].content)
