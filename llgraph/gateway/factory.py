"""Control gateway 工厂。"""

from __future__ import annotations

import os
from typing import Literal

from llgraph.gateway.local import LocalControlGateway
from llgraph.gateway.protocol import ControlGateway
from llgraph.gateway.remote import RemoteControlGateway

_MODE_ENV = "LLGRAPH_CONTROL_GATEWAY"
_URL_ENV = "LLGRAPH_CONTROL_GATEWAY_URL"

GatewayMode = Literal["local", "remote"]

_instance: ControlGateway | None = None


def _parse_mode(raw: str) -> GatewayMode:
    mode = raw.strip().lower()
    if mode in ("local", "remote"):
        return mode
    raise ValueError(
        f"无效的 {_MODE_ENV}={raw!r}，可选: local | remote"
    )


def get_control_gateway(*, force_new: bool = False) -> ControlGateway:
    """
    获取 Control gateway 单例。

    环境变量:
      - LLGRAPH_CONTROL_GATEWAY: local（默认）| remote
      - LLGRAPH_CONTROL_GATEWAY_URL: remote 模式下的服务根 URL

    @param force_new 是否强制重建实例（测试用）
    @return ControlGateway 实现
    """
    global _instance
    if _instance is not None and not force_new:
        return _instance

    mode = _parse_mode(os.environ.get(_MODE_ENV, "local"))
    if mode == "remote":
        url = os.environ.get(_URL_ENV, "").strip()
        if not url:
            raise ValueError(
                f"remote 模式须设置 {_URL_ENV}，例如 http://127.0.0.1:8766"
            )
        _instance = RemoteControlGateway(url)
        return _instance

    _instance = LocalControlGateway()
    return _instance


def reset_control_gateway() -> None:
    """清除 gateway 单例（测试用）。"""
    global _instance
    _instance = None
