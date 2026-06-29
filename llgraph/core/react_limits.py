"""ReAct 图 recursion_limit（agent / plan 子图共用）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.core.agent_config import load_agent_config

DEFAULT_REACT_MAX_TURNS = 500
REACT_MAX_TURNS_CAP = 500


def parse_react_max_turns(raw: object, *, default: int = DEFAULT_REACT_MAX_TURNS) -> int:
    """
    解析 max_turns 配置项。

    @param raw agent.json 原始值
    @param default 缺省时的默认值
    @return 1～REACT_MAX_TURNS_CAP 之间的整数
    """
    if raw is None:
        return default
    try:
        return max(1, min(REACT_MAX_TURNS_CAP, int(raw)))
    except (TypeError, ValueError):
        return default


def resolve_agent_max_turns(workspace: Path) -> int:
    """
    主 Chat Agent ReAct 步数上限。

    @param workspace 工作区根
    @return recursion_limit
    """
    cfg = load_agent_config(workspace)
    agent = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    return parse_react_max_turns(agent.get("max_turns"))
