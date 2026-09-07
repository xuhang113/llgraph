"""交互模式的 Agent 预热句柄。

冷启动 90% 的时间是 import：`llgraph.core.agent` 会拉起 langgraph +
langchain_anthropic + anthropic SDK，约 0.5～1s，而这段时间对用户是纯等待。
把它挪到 banner 之后、首轮之前的空档（用户正在读 banner / 打字）里做，
终端就能立刻出提示符。

这里不做「后台线程 import 顺便省 CPU」的幻想：GIL 下 import 是 CPU 密集，
和主线程抢锁并不会更快。真正的收益来自「主线程空闲等输入」这段时间。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class LazyAgent:
    """按需构建的 Agent 代理：属性访问时阻塞到构建完成。

    只做转发，不缓存方法：`get_state` / `update_state` / `stream` 等
    都落到真实 CompiledStateGraph 上。
    """

    def __init__(self, builder: Callable[[], Any], *, background: bool = True) -> None:
        """
        @param builder 零参构建函数（通常是 build_agent 的闭包）
        @param background True 时立刻起后台线程构建
        """
        self._llgraph_builder = builder
        self._llgraph_agent: Any | None = None
        self._llgraph_lock = threading.Lock()
        # 与状态锁分开：构建期间不能占着状态锁，但也不许两路同时建
        self._llgraph_build_lock = threading.Lock()
        self._llgraph_thread: threading.Thread | None = None
        if background:
            self.start_background_build()

    def start_background_build(self) -> None:
        """起后台线程构建；已构建或已在构建时是空操作。"""
        with self._llgraph_lock:
            if self._llgraph_agent is not None or self._llgraph_thread is not None:
                return
            thread = threading.Thread(
                target=self._background_build,
                name="llgraph-agent-warmup",
                daemon=True,
            )
            self._llgraph_thread = thread
        thread.start()

    def _background_build(self) -> None:
        try:
            self._build_once()
        except BaseException:
            # 后台失败不吞不抛：resolve() 会在主线程重跑，异常照原样在调用点抛出，
            # 保证配置错误的提示位置与同步构建时一致。
            return

    def _build_once(self) -> Any:
        with self._llgraph_build_lock:
            with self._llgraph_lock:
                if self._llgraph_agent is not None:
                    return self._llgraph_agent
            agent = self._llgraph_builder()
            with self._llgraph_lock:
                if self._llgraph_agent is None:
                    self._llgraph_agent = agent
                return self._llgraph_agent

    @property
    def ready(self) -> bool:
        """是否已构建完成（不阻塞）。"""
        with self._llgraph_lock:
            return self._llgraph_agent is not None

    def resolve(self) -> Any:
        """返回真实 Agent；必要时阻塞等待后台线程或就地构建。"""
        with self._llgraph_lock:
            if self._llgraph_agent is not None:
                return self._llgraph_agent
            thread = self._llgraph_thread
        if thread is not None:
            thread.join()
        return self._build_once()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_llgraph_"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)


def unwrap_agent(agent: Any) -> Any:
    """取出真实 Agent；非 LazyAgent 原样返回。"""
    if isinstance(agent, LazyAgent):
        return agent.resolve()
    return agent
