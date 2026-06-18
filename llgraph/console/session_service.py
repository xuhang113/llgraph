"""Console 会话删除：经 Control gateway 委托 llgraph 核心。"""

from __future__ import annotations

from typing import Any

from llgraph.gateway import get_control_gateway


def delete_session_for_web(workspace_path: str, thread_id: str) -> dict[str, Any]:
    """
    删除 Agent 或 Plan 会话（含 Plan Worker 级联）。

    @param workspace_path 工作区根路径
    @param thread_id 会话 ID
    @return API 响应 dict
    """
    record = get_control_gateway().delete_session(workspace_path, thread_id)
    return record.to_dict()


def delete_sessions_for_web(workspace_path: str, thread_ids: list[str]) -> dict[str, Any]:
    """
    批量删除 Agent / Plan 会话。

    @param workspace_path 工作区根路径
    @param thread_ids 会话 ID 列表
    @return 批量删除结果
    """
    from pathlib import Path

    from llgraph.session.session_delete import delete_sessions

    workspace = Path(workspace_path).expanduser().resolve()
    ids = [tid.strip() for tid in thread_ids if tid and tid.strip()]
    report = delete_sessions(workspace, ids)
    results = [
        {
            "thread_id": item.thread_id,
            "ok": item.ok,
            "removed_paths": list(item.removed_paths),
            "related_removed": list(item.related_removed),
            "error": item.error,
        }
        for item in report.results
    ]
    return {
        "ok": report.failure_count == 0,
        "success_count": report.success_count,
        "failure_count": report.failure_count,
        "results": results,
    }
