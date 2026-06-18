"""SSE 工具函数。"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator


def format_sse(event: dict[str, Any]) -> str:
    """
    格式化为 SSE data 行。

    @param event 事件 dict
    @return SSE 文本
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def merge_sse_streams(
    queue: asyncio.Queue[dict[str, Any]],
    *,
    timeout_sec: float = 3600.0,
) -> AsyncIterator[str]:
    """
    从队列读取事件并 yield SSE 行。

    @param queue 事件队列
    @param timeout_sec 总超时
    @yield SSE 行
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_sec
    while loop.time() < deadline:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            yield format_sse({"type": "ping"})
            continue
        if event.get("type") == "end":
            yield format_sse(event)
            break
        yield format_sse(event)
