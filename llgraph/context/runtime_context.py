"""每轮 invoke 的运行时上下文（ContextVar），供工具读取当前 thread_id。"""

from __future__ import annotations

from contextvars import ContextVar

_active_thread_id: ContextVar[str | None] = ContextVar("llgraph_active_thread_id", default=None)


def set_active_thread_id(thread_id: str | None) -> None:
    """
    设置当前轮次会话 thread_id（invoke/stream 入口调用）。

    @param thread_id 线程 ID；None 表示清除
    """
    _active_thread_id.set(thread_id.strip() if thread_id else None)


def get_active_thread_id() -> str | None:
    """
    读取当前轮次 thread_id。

    @return thread_id 或 None
    """
    return _active_thread_id.get()


def require_active_thread_id() -> str:
    """
    读取 thread_id，缺失时抛出 ValueError。

    @return thread_id
    """
    tid = get_active_thread_id()
    if not tid:
        raise ValueError("当前无活动会话 thread_id（需在交互/记忆模式下调用）")
    return tid
