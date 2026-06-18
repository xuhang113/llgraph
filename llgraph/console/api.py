"""
Console 公开库 API（集成请使用本模块，勿依赖 HTTP 路由）。

Agent/Plan 流式对话见 ``llgraph.console.runtime`` 或终端 ``llgraph`` CLI。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from llgraph.console import discovery
from llgraph.console.session_service import delete_session_for_web
from llgraph.gateway import get_control_gateway


class ConsoleEvent:
    """Console 流式事件（预留；runtime 模块产生同类 dict）。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        """
        转为 dict。

        @return 事件载荷
        """
        return dict(self.payload)


class Console:
    """
    llgraph 控制台 Python API。

    工作区 / 会话 / 删除等走 ControlGateway；读写详情走 discovery。
    """

    def list_workspaces(self) -> list[dict[str, Any]]:
        """
        列举已注册工作区。

        @return 工作区摘要列表
        """
        return [asdict(w) for w in discovery.discover_workspaces()]

    def register_workspace(self, path: str) -> dict[str, Any]:
        """
        注册工作区。

        @param path 工作区绝对路径
        @return 工作区摘要
        """
        return asdict(discovery.register_workspace_path(path))

    def resolve_workspace(self, slug: str) -> Path:
        """
        解析 slug 对应工作区路径。

        @param slug context slug
        @return 工作区 Path
        """
        return discovery.workspace_path_from_slug(slug)

    def session_tree(self, slug: str) -> dict[str, Any]:
        """
        构建 Agent + Plan 会话树。

        @param slug 工作区 slug
        @return agents / plans 树
        """
        workspace = self.resolve_workspace(slug)
        return discovery.build_session_tree(workspace)

    def delete_session(self, slug: str, thread_id: str) -> dict[str, Any]:
        """
        删除 Agent 或 Plan 会话（Plan 含 Worker 级联）。

        @param slug 工作区 slug
        @param thread_id 会话 ID
        @return 删除结果
        """
        workspace = self.resolve_workspace(slug)
        return delete_session_for_web(str(workspace), thread_id)

    def gateway(self):
        """
        获取 ControlGateway（高级用法）。

        @return ControlGateway 实例
        """
        return get_control_gateway()
