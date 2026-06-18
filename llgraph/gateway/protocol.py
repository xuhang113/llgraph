"""Control gateway 协议（本地 / 远程统一接口）。"""

from __future__ import annotations

from typing import Protocol

from llgraph.gateway.types import SessionDeleteRecord, WorkspaceRecord


class ControlGateway(Protocol):
    """
    llgraph 控制面网关。

    Web 或其它客户端只依赖本协议；本地实现委托 llgraph 库，远程实现走 HTTP。
    """

    def list_workspaces(self) -> list[WorkspaceRecord]:
        """
        列举已注册工作区。

        @return 工作区列表
        """
        ...

    def register_workspace(self, path: str) -> WorkspaceRecord:
        """
        注册工作区到用户 context。

        @param path 工作区绝对路径
        @return 工作区摘要
        """
        ...

    def resolve_workspace_path(self, slug: str) -> str:
        """
        由 slug 解析工作区根路径。

        @param slug context 目录名
        @return 工作区绝对路径
        """
        ...

    def dismiss_workspace_from_recent(self, slug: str) -> None:
        """
        从最近工作区列表隐藏（不删除会话数据）。

        @param slug context 目录名
        """
        ...

    def touch_workspace_opened(self, slug: str) -> None:
        """
        记录工作区最近打开时间。

        @param slug context 目录名
        """
        ...

    def delete_session(self, workspace_path: str, thread_id: str) -> SessionDeleteRecord:
        """
        删除 Agent 或 Plan 会话（Plan 含 Worker 级联）。

        @param workspace_path 工作区根路径
        @param thread_id 会话 ID
        @return 删除结果
        """
        ...
