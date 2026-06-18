"""plan_state.json 持久化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llgraph.plan.plan_registry import plan_state_json_path
from llgraph.plan.state import PlanState


def load_plan_state(workspace: Path, thread_id: str) -> dict[str, Any]:
    """
    读取 plan_state.json。

    @param workspace 工作区根
    @param thread_id plan thread
    @return state dict
    """
    path = plan_state_json_path(workspace, thread_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_plan_state(workspace: Path, thread_id: str, state: PlanState | dict[str, Any]) -> None:
    """
    写入 plan_state.json。

    @param workspace 工作区根
    @param thread_id plan thread
    @param state PlanState 或 dict
    """
    path = plan_state_json_path(workspace, thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(state)
    if "user_messages" in serializable and isinstance(serializable["user_messages"], list):
        serializable["user_messages"] = list(serializable["user_messages"])
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
