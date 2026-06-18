"""Runtime 事件总线（SSE 广播）。"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from typing import Any


class EventHub:
    """按 channel 广播事件到多个 asyncio.Queue。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)

    def subscribe(self, channel: str) -> asyncio.Queue[dict[str, Any]]:
        """
        订阅 channel。

        @param channel 如 plan:plan-abc123
        @return 事件队列
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        with self._lock:
            self._queues[channel].append(queue)
        return queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """
        取消订阅。

        @param channel channel 名
        @param queue 队列实例
        """
        with self._lock:
            items = self._queues.get(channel, [])
            if queue in items:
                items.remove(queue)
            if not items and channel in self._queues:
                del self._queues[channel]

    def publish(self, channel: str, event: dict[str, Any]) -> None:
        """
        向 channel 所有订阅者推送事件（线程安全）。

        @param channel channel 名
        @param event 事件 dict
        """
        with self._lock:
            queues = list(self._queues.get(channel, []))
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def publish_sync(self, channel: str, event: dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """
        从工作线程向 asyncio 队列推送。

        @param channel channel 名
        @param event 事件 dict
        @param loop 主事件循环
        """

        def _put() -> None:
            self.publish(channel, event)

        loop.call_soon_threadsafe(_put)


HUB = EventHub()
