"""Plan 子图注册表（供 workflow 视图与 Web 流式扩展）。"""

from __future__ import annotations

from typing import Any

from llgraph.plan.subgraphs.planner import PLANNER_SUBGRAPH_SPEC
from llgraph.plan.subgraphs.worker import WORKER_SUBGRAPH_SPEC

PLAN_SUBGRAPH_REGISTRY: dict[str, dict[str, Any]] = {
    "planner": {
        "spec": PLANNER_SUBGRAPH_SPEC,
        "engine": "langgraph_react",
        "state_schema": "MessagesState",
        "parent_node": "planner",
        "thread_pattern": "{thread_id}:planner:v{version}",
    },
    "worker": {
        "spec": WORKER_SUBGRAPH_SPEC,
        "engine": "langgraph_react",
        "state_schema": "MessagesState",
        "parent_node": "worker",
        "thread_pattern": "{thread_id}:worker:{task_id}",
    },
}


def subgraph_definitions_for_workflow() -> list[dict[str, Any]]:
    """
    返回可嵌入 workflow_snapshot / Web API 的子图定义列表。

    @return 子图元数据列表
    """
    items: list[dict[str, Any]] = []
    for kind, meta in PLAN_SUBGRAPH_REGISTRY.items():
        spec = meta["spec"]
        items.append(
            {
                "id": kind,
                "kind": spec.subgraph_kind,
                "parent_node": spec.node_id,
                "engine": meta["engine"],
                "state_schema": meta["state_schema"],
                "thread_pattern": meta["thread_pattern"],
            }
        )
    return items
