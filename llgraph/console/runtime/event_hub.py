"""Runtime 事件总线（SSE 广播）。"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from typing import Any

_BUFFER_MAX = 200
_SKIP_BUFFER_TYPES = frozenset({"ping", "subscribed"})


class EventHub:
    """按 channel 广播事件到多个 asyncio.Queue；断线重连时可回放本轮缓冲。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)
        self._turn_buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _record_turn_event(self, channel: str, event: dict[str, Any]) -> None:
        typ = str(event.get("type") or "")
        if typ in _SKIP_BUFFER_TYPES:
            return
        if typ == "turn_start":
            self._turn_buffers[channel] = []
        buf = self._turn_buffers[channel]
        buf.append(event)
        if len(buf) > _BUFFER_MAX:
            del buf[: len(buf) - _BUFFER_MAX]
        if typ == "end":
            self._turn_buffers[channel] = []

    def subscribe(self, channel: str) -> asyncio.Queue[dict[str, Any]]:
        """
        订阅 channel。

        @param channel 如 plan:plan-abc123
        @return 事件队列
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        with self._lock:
            self._queues[channel].append(queue)
            replay = list(self._turn_buffers.get(channel, []))
        for event in replay:
            replay_event = {**event, "replay": True}
            try:
                queue.put_nowait(replay_event)
            except asyncio.QueueFull:
                break
        return queue

    def has_subscribers(self, channel: str) -> bool:
        """channel 是否已有 SSE 订阅者。"""
        with self._lock:
            return bool(self._queues.get(channel))

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
            self._record_turn_event(channel, event)
            queues = list(self._queues.get(channel, []))
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def close_all(self) -> None:
        """
        进程 shutdown：向所有订阅队列投递 end，唤醒阻塞中的 SSE generator。

        配合 merge_sse_streams 与 Uvicorn graceful shutdown，避免关页/ Ctrl+C 后长时间挂起。
        """
        with self._lock:
            all_queues = [q for qs in self._queues.values() for q in qs]
            self._queues.clear()
            self._turn_buffers.clear()
        for queue in all_queues:
            try:
                queue.put_nowait({"type": "end"})
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
