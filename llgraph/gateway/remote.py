"""远程 Control gateway（占位；后续 HTTP/WebSocket 实现）。"""

from __future__ import annotations

from llgraph.gateway.types import SessionDeleteRecord, WorkspaceRecord


class RemoteControlGateway:
    """
    通过 HTTP 调用远端 llgraph-gateway 服务。

    当前为占位实现；设置 ``LLGRAPH_CONTROL_GATEWAY=remote`` 时会提示未实现。
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        """Gateway 服务根 URL。"""
        return self._base_url

    def _not_implemented(self, method: str) -> None:
        raise NotImplementedError(
            f"RemoteControlGateway.{method} 尚未实现。"
            f" 请在执行节点部署 llgraph-gateway 服务，或设置 LLGRAPH_CONTROL_GATEWAY=local。"
            f" base_url={self._base_url}"
        )

    def list_workspaces(self) -> list[WorkspaceRecord]:
        """
        列举远端工作区。

        @return 工作区列表
        """
        self._not_implemented("list_workspaces")
        return []

    def register_workspace(self, path: str) -> WorkspaceRecord:
        """
        在远端注册工作区。

        @param path 工作区路径（远端可见）
        @return 工作区摘要
        """
        self._not_implemented("register_workspace")
        return WorkspaceRecord(slug="", path=path)

    def resolve_workspace_path(self, slug: str) -> str:
        """
        解析远端工作区路径。

        @param slug 工作区 slug
        @return 工作区路径
        """
        self._not_implemented("resolve_workspace_path")
        return ""

    def dismiss_workspace_from_recent(self, slug: str) -> None:
        """
        从远端最近列表隐藏工作区。

        @param slug 工作区 slug
        """
        self._not_implemented("dismiss_workspace_from_recent")

    def touch_workspace_opened(self, slug: str) -> None:
        """
        记录远端工作区最近打开时间。

        @param slug 工作区 slug
        """
        self._not_implemented("touch_workspace_opened")

    def delete_session(self, workspace_path: str, thread_id: str) -> SessionDeleteRecord:
        """
        在远端删除会话。

        @param workspace_path 工作区路径
        @param thread_id 会话 ID
        @return 删除结果
        """
        self._not_implemented("delete_session")
        return SessionDeleteRecord(
            thread_id=thread_id,
            kind="agent",
            ok=False,
            removed_paths=(),
            error="remote gateway not implemented",
        )
