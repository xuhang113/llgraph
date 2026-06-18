"""
Control-plane gateway：供 Web / 远程客户端调用 llgraph 能力。

与 ``core.gateway_*``（AI 模型网关）无关；本包是 **会话/工作区控制面** 的稳定边界。
当前默认 ``LocalControlGateway``（同进程）；后续可切换 ``RemoteControlGateway``（HTTP）。
"""

from llgraph.gateway.factory import get_control_gateway, reset_control_gateway
from llgraph.gateway.protocol import ControlGateway
from llgraph.gateway.types import SessionDeleteRecord, WorkspaceRecord

__all__ = [
    "ControlGateway",
    "SessionDeleteRecord",
    "WorkspaceRecord",
    "get_control_gateway",
    "reset_control_gateway",
]
