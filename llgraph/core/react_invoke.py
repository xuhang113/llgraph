"""可中断 LLM 调用：Web Stop + thinking 流式超时。"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig

from llgraph.context.runtime_context import get_active_thread_id
from llgraph.core.agent_invoke_timing import AgentInvokeTiming
from llgraph.core.llm_settings import (
    DEFAULT_THINKING_STREAM_TIMEOUT_SEC,
    resolve_llm_settings,
)

_POLL_SEC = 0.05


def agent_cancel_requested() -> bool:
    """@return 当前活动 thread 是否已请求 Web Stop"""
    tid = get_active_thread_id()
    if not tid:
        return False
    from llgraph.console.runtime.agent_service import is_agent_cancel_requested

    return is_agent_cancel_requested(tid)


def _merge_stream_chunk(accumulated: Any, chunk: Any) -> Any:
    if accumulated is None:
        return chunk
    try:
        return accumulated + chunk
    except TypeError:
        return chunk


def _to_ai_message(response: Any) -> AIMessage:
    if isinstance(response, AIMessage):
        return response
    return AIMessage(content=str(response))


def _raise_if_cancelled() -> None:
    from llgraph.session.session_run_log import UserCancelledError

    if agent_cancel_requested():
        raise UserCancelledError("用户停止当前生成")


def _resolve_runnable_workspace(agent_runnable: Any) -> Path | None:
    """
    从 Runnable 链上解析 llgraph_workspace。

    @param agent_runnable prompt | bound_model 等
    @return 工作区根或 None
    """
    seen: set[int] = set()

    def walk(obj: Any) -> Path | None:
        if obj is None:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)
        ws = getattr(obj, "llgraph_workspace", None)
        if ws is not None:
            try:
                return Path(ws).expanduser().resolve()
            except (TypeError, ValueError, OSError):
                return None
        for attr in ("bound", "last", "first", "middle", "runnable"):
            child = getattr(obj, attr, None)
            if child is None:
                continue
            if isinstance(child, list):
                for item in child:
                    found = walk(item)
                    if found is not None:
                        return found
            else:
                found = walk(child)
                if found is not None:
                    return found
        steps = getattr(obj, "steps", None)
        if isinstance(steps, list):
            for item in steps:
                found = walk(item)
                if found is not None:
                    return found
        return None

    return walk(agent_runnable)


def _merged_response_progress(response: Any) -> tuple[bool, bool, bool]:
    """
    @param response 聚合中的 AIMessageChunk
    @return (has_tool_calls, has_visible_text, has_thinking)
    """
    if not isinstance(response, (AIMessage, AIMessageChunk)):
        return False, False, False
    has_tools = bool(getattr(response, "tool_calls", None))
    from llgraph.display.trace_display import (
        _extract_text_from_message_chunk,
        _extract_thinking_from_message_chunk,
    )

    has_text = bool(_extract_text_from_message_chunk(response).strip())
    has_thinking = bool(_extract_thinking_from_message_chunk(response).strip())
    return has_tools, has_text, has_thinking


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _consume_runnable_stream(
    agent_runnable: Any,
    state: Any,
    config: RunnableConfig,
) -> Any:
    """
    在后台线程拉取 stream，主线程轮询 cancel；Stop 时 close stream，不等待整包 invoke。

    @return 聚合后的响应 chunk
    @raises UserCancelledError 用户 Stop
    @raises ThinkingStreamTimeoutError thinking-only 流式超时
    """
    from llgraph.session.session_run_log import ThinkingStreamTimeoutError, UserCancelledError

    _raise_if_cancelled()

    ws = _resolve_runnable_workspace(agent_runnable)
    if ws is not None:
        thinking_timeout = resolve_llm_settings(ws).thinking_stream_timeout_sec
    else:
        thinking_timeout = DEFAULT_THINKING_STREAM_TIMEOUT_SEC

    stream = agent_runnable.stream(state, config)
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    closed = threading.Event()

    def _producer() -> None:
        try:
            for chunk in stream:
                if closed.is_set():
                    break
                events.put(("chunk", chunk))
            if not closed.is_set():
                events.put(("done", None))
        except Exception as exc:
            if not closed.is_set():
                events.put(("error", exc))
        finally:
            _close_stream(stream)

    thread = threading.Thread(target=_producer, daemon=True, name="llgraph-llm-stream")
    thread.start()

    response: Any = None
    thinking_only_since: float | None = None
    try:
        while True:
            try:
                kind, payload = events.get(timeout=_POLL_SEC)
            except queue.Empty:
                if agent_cancel_requested():
                    closed.set()
                    _close_stream(stream)
                    raise UserCancelledError("用户停止当前生成")
                if (
                    thinking_only_since is not None
                    and time.perf_counter() - thinking_only_since >= thinking_timeout
                ):
                    closed.set()
                    _close_stream(stream)
                    raise ThinkingStreamTimeoutError(
                        f"thinking 流式超过 {thinking_timeout:.0f}s，已中断以避免空转"
                    )
                continue

            if kind == "chunk":
                response = _merge_stream_chunk(response, payload)
                has_tools, has_text, has_thinking = _merged_response_progress(response)
                if has_tools or has_text:
                    thinking_only_since = None
                elif has_thinking:
                    if thinking_only_since is None:
                        thinking_only_since = time.perf_counter()
                    elif time.perf_counter() - thinking_only_since >= thinking_timeout:
                        closed.set()
                        _close_stream(stream)
                        raise ThinkingStreamTimeoutError(
                            f"thinking 流式超过 {thinking_timeout:.0f}s，已中断以避免空转"
                        )
            elif kind == "done":
                break
            elif kind == "error":
                raise payload
    finally:
        closed.set()
        thread.join(timeout=1.0)

    _raise_if_cancelled()
    if response is None:
        raise UserCancelledError("用户停止当前生成")
    return response


def invoke_agent_runnable_cancellable(
    agent_runnable: Any,
    state: Any,
    config: RunnableConfig,
    *,
    timing: AgentInvokeTiming | None = None,
) -> AIMessage:
    """
    可中断 LLM 调用：放弃未完成的大模型单次返回，不阻塞 invoke 至整包结束。

    @param agent_runnable prompt | bound_model
    @param state LangGraph state
    @param config RunnableConfig
    @param timing 可选；写入 http_sec
    @return AIMessage
    @raises UserCancelledError 用户 Stop
    @raises ThinkingStreamTimeoutError thinking-only 超时
    """
    http_start = time.perf_counter()
    response = _consume_runnable_stream(agent_runnable, state, config)
    if timing is not None:
        timing.http_sec = time.perf_counter() - http_start
    return _to_ai_message(response)


async def ainvoke_agent_runnable_cancellable(
    agent_runnable: Any,
    state: Any,
    config: RunnableConfig,
) -> AIMessage:
    """异步版：chunk 间检查 cancel（无 prefill 空窗轮询）。"""
    from llgraph.session.session_run_log import UserCancelledError

    _raise_if_cancelled()

    response: Any = None
    stream = agent_runnable.astream(state, config)
    try:
        async for chunk in stream:
            if agent_cancel_requested():
                raise UserCancelledError("用户停止当前生成")
            response = _merge_stream_chunk(response, chunk)
    finally:
        close = getattr(stream, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass

    _raise_if_cancelled()
    if response is None:
        raise UserCancelledError("用户停止当前生成")
    return _to_ai_message(response)
