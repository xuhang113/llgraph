"""父 Agent 运行时槽：工具闭包在 invoke 时读取最新 runtime（含 SSE）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llgraph.subagent.runtime import SubagentRuntime, runtime_from_agent_session


@dataclass
class SubagentParentSlot:
    """可变槽：build_agent 时创建，会话就绪 / 每轮对话前 bind。"""

    runtime: SubagentRuntime | None = None
    _lock: Any = field(default=None, repr=False)

    def bind_from_session(
        self,
        session: Any,
        *,
        sse_emit: Callable[[dict[str, Any]], None] | None = None,
        sse_loop: Any = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> SubagentRuntime:
        """
        用当前 AgentSessionContext 刷新父运行时。

        @param session AgentSessionContext
        @param sse_emit Web 轮次 emit（子会话双写）
        @param sse_loop asyncio loop
        @param cancel_check 父会话取消探测
        """
        rt = runtime_from_agent_session(session)
        if sse_emit is not None:
            rt.sse_emit = sse_emit
            rt.sse_loop = sse_loop
        if cancel_check is not None:
            rt.cancel_check = cancel_check
        self.runtime = rt
        return rt
