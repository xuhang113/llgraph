"""代码索引是否可用于 Agent 检索。"""

from __future__ import annotations

from pathlib import Path

from llgraph.code_index.store import get_index_status


def code_index_is_ready(workspace: Path) -> bool:
    """
    工作区是否已完成向量化索引（chunk_count > 0）。

    @param workspace 工作区根
    @return 是否可用 hybrid/semantic 检索
    """
    try:
        status = get_index_status(workspace.expanduser().resolve())
    except Exception:
        return False
    return bool(status.exists and status.chunk_count > 0)
