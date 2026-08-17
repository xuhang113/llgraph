"""agent_pool 配置解析测试。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.config.agent_pool_settings import (
    DEFAULT_POOL_SIZE,
    DEFAULT_WARM_RECENT_DAYS,
    DEFAULT_WARM_RECENT_LIMIT,
    resolve_agent_pool_settings,
)


def test_agent_pool_defaults_when_missing(tmp_path: Path) -> None:
    settings = resolve_agent_pool_settings(tmp_path)
    assert settings.pool_size == DEFAULT_POOL_SIZE
    assert settings.warm_recent_limit == DEFAULT_WARM_RECENT_LIMIT
    assert settings.warm_recent_days == DEFAULT_WARM_RECENT_DAYS


def test_agent_pool_workspace_override(tmp_path: Path) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text(
        json.dumps(
            {
                "agent_pool": {
                    "pool_size": 8,
                    "warm_recent_limit": 0,
                    "warm_recent_days": 14,
                }
            }
        ),
        encoding="utf-8",
    )
    settings = resolve_agent_pool_settings(tmp_path)
    assert settings.pool_size == 8
    assert settings.warm_recent_limit == 0
    assert settings.warm_recent_days == 14
