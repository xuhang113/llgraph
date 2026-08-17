"""Agent LRU 保活池配置（agent.json → agent_pool 段）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.core.agent_config import load_agent_config

DEFAULT_POOL_SIZE = 5
DEFAULT_WARM_RECENT_LIMIT = 3
DEFAULT_WARM_RECENT_DAYS = 7
MAX_POOL_SIZE = 32


@dataclass(frozen=True)
class AgentPoolSettings:
    """Web Agent 会话保活池参数。"""

    pool_size: int
    warm_recent_limit: int
    warm_recent_days: int


def _parse_bounded_int(
    raw: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if raw is None:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except (TypeError, ValueError):
        return default


def resolve_agent_pool_settings(workspace: Path | None = None) -> AgentPoolSettings:
    """
    解析 agent_pool 配置（用户 ~/.llgraph/agent.json 为底，工作区覆盖）。

    @param workspace 工作区根；None 时仅用户级
    @return AgentPoolSettings
    """
    cfg = load_agent_config(workspace)
    section = cfg.get("agent_pool") if isinstance(cfg.get("agent_pool"), dict) else {}

    pool_size = _parse_bounded_int(
        section.get("pool_size"),
        default=DEFAULT_POOL_SIZE,
        minimum=1,
        maximum=MAX_POOL_SIZE,
    )
    warm_recent_limit = _parse_bounded_int(
        section.get("warm_recent_limit"),
        default=DEFAULT_WARM_RECENT_LIMIT,
        minimum=0,
        maximum=MAX_POOL_SIZE,
    )
    warm_recent_days = _parse_bounded_int(
        section.get("warm_recent_days"),
        default=DEFAULT_WARM_RECENT_DAYS,
        minimum=1,
        maximum=30,
    )

    return AgentPoolSettings(
        pool_size=pool_size,
        warm_recent_limit=warm_recent_limit,
        warm_recent_days=warm_recent_days,
    )
