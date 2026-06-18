"""同进程 Control gateway（当前默认实现）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.gateway.protocol import ControlGateway
from llgraph.gateway.services import workspace_catalog
from llgraph.gateway.types import SessionDeleteRecord, WorkspaceRecord
from llgraph.session.session_delete import (
    delete_workspace_session,
    is_plan_main_thread,
)


def _session_kind(thread_id: str) -> str:
    """
    推断会话类型。

    @param thread_id 会话 ID
    @return agent | plan | worker
    """
    if is_plan_main_thread(thread_id):
        return "plan"
    if thread_id.startswith("plan-"):
        return "worker"
    return "agent"


class LocalControlGateway:
    """在本机进程内直接调用 llgraph 库。"""

    def list_workspaces(self) -> list[WorkspaceRecord]:
        """
        列举已注册工作区。

        @return 工作区列表
        """
        return workspace_catalog.list_workspaces()

    def register_workspace(self, path: str) -> WorkspaceRecord:
        """
        注册工作区。

        @param path 工作区绝对路径
        @return 工作区摘要
        """
        return workspace_catalog.register_workspace(path)

    def resolve_workspace_path(self, slug: str) -> str:
        """
        解析 slug 对应工作区路径。

        @param slug context 目录名
        @return 工作区绝对路径
        """
        return workspace_catalog.resolve_workspace_path(slug)

    def dismiss_workspace_from_recent(self, slug: str) -> None:
        """
        从最近列表隐藏工作区。

        @param slug context 目录名
        """
        workspace_catalog.dismiss_workspace_from_recent(slug)

    def touch_workspace_opened(self, slug: str) -> None:
        """
        记录工作区最近打开时间。

        @param slug context 目录名
        """
        workspace_catalog.touch_workspace_opened(slug)

    def delete_session(self, workspace_path: str, thread_id: str) -> SessionDeleteRecord:
        """
        删除会话落盘（Plan 含级联）。

        @param workspace_path 工作区根路径
        @param thread_id 会话 ID
        @return 删除结果
        """
        workspace = Path(workspace_path).expanduser().resolve()
        result = delete_workspace_session(workspace, thread_id)
        return SessionDeleteRecord(
            thread_id=result.thread_id,
            kind=_session_kind(result.thread_id),
            ok=result.ok,
            removed_paths=result.removed_paths,
            related_removed=result.related_removed,
            error=result.error,
        )
