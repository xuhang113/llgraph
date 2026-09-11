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


class EmptyGatewayStreamError(RuntimeError):
    """流式正常结束但一个 chunk 都没有（网关空响应）。"""


class StreamAttemptProgress:
    """一次流式尝试里已经外发给终端 / Web 的内容（决定还能不能重放）。"""

    __slots__ = ("emitted_visible",)

    def __init__(self) -> None:
        self.emitted_visible = False


def _stream_once(
    agent_runnable: Any,
    state: Any,
    config: RunnableConfig,
    *,
    thinking_timeout: float,
    progress: StreamAttemptProgress,
) -> Any:
    """
    在后台线程拉取 stream，主线程轮询 cancel；Stop 时 close stream，不等待整包 invoke。

    @param thinking_timeout thinking-only 空转上限
    @param progress 回填本次尝试是否已外发可见正文 / tool_call
    @return 聚合后的响应 chunk
    @raises UserCancelledError 用户 Stop
    @raises ThinkingStreamTimeoutError thinking-only 流式超时
    """
    from llgraph.session.session_run_log import ThinkingStreamTimeoutError, UserCancelledError

    _raise_if_cancelled()

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
                    progress.emitted_visible = True
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
        # 网关返回空 body：不是用户 Stop，交给外层按瞬时故障重放
        raise EmptyGatewayStreamError("网关未返回任何内容")
    return response


def _sleep_with_cancel(delay_sec: float) -> None:
    """
    退避期间保持 Stop 可响应。

    @param delay_sec 退避秒数
    @raises UserCancelledError 退避途中用户 Stop
    """
    deadline = time.perf_counter() + max(0.0, delay_sec)
    while True:
        _raise_if_cancelled()
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(_POLL_SEC, remaining))


def _log_stream_retry(
    workspace: Path | None,
    *,
    phase: str,
    detail: dict[str, Any],
) -> None:
    """把重放事件写进 run_log.jsonl，失败不影响主流程。"""
    thread_id = get_active_thread_id()
    if workspace is None or not thread_id:
        return
    try:
        from llgraph.session.session_run_log import log_react_phase

        log_react_phase(workspace, thread_id, phase=phase, detail=detail)
    except Exception:
        pass


def _consume_runnable_stream(
    agent_runnable: Any,
    state: Any,
    config: RunnableConfig,
) -> Any:
    """
    流式取一次 LLM 响应；网关瞬时故障时原样重放整次调用。

    只在「本次尝试还没外发可见正文 / tool_call」时重放：已经流到终端的内容
    不能重来一遍，否则用户会看到半截重复。thinking-only 的半截可以丢，
    重放比整轮 ReAct 作废便宜得多。

    @return 聚合后的响应 chunk
    @raises UserCancelledError 用户 Stop / 多次重放后网关仍返回空
    @raises ThinkingStreamTimeoutError thinking-only 流式超时
    """
    from llgraph.core.llm_retry import (
        classify_stream_failure,
        compute_retry_delay,
        resolve_llm_stream_retry_policy,
    )
    from llgraph.session.session_run_log import ThinkingStreamTimeoutError, UserCancelledError

    ws = _resolve_runnable_workspace(agent_runnable)
    if ws is not None:
        thinking_timeout = resolve_llm_settings(ws).thinking_stream_timeout_sec
    else:
        thinking_timeout = DEFAULT_THINKING_STREAM_TIMEOUT_SEC
    policy = resolve_llm_stream_retry_policy(ws)

    failures = 0
    while True:
        progress = StreamAttemptProgress()
        try:
            response = _stream_once(
                agent_runnable,
                state,
                config,
                thinking_timeout=thinking_timeout,
                progress=progress,
            )
            if failures:
                _log_stream_retry(
                    ws,
                    phase="llm_stream_retry_ok",
                    detail={"attempts": failures + 1},
                )
            return response
        except (UserCancelledError, ThinkingStreamTimeoutError):
            raise
        except BaseException as exc:  # noqa: BLE001 — 分类后决定重放还是原样抛出
            if isinstance(exc, EmptyGatewayStreamError):
                reason: str | None = "empty_stream"
            else:
                reason = classify_stream_failure(exc)
            failures += 1
            blocked = (
                reason is None
                or progress.emitted_visible
                or failures >= policy.max_attempts
                or not policy.enabled
            )
            if blocked:
                if reason is not None:
                    _log_stream_retry(
                        ws,
                        phase="llm_stream_retry_give_up",
                        detail={
                            "reason": reason,
                            "attempts": failures,
                            "emitted_visible": progress.emitted_visible,
                        },
                    )
                if isinstance(exc, EmptyGatewayStreamError):
                    raise UserCancelledError("网关未返回任何内容") from exc
                raise

            delay = compute_retry_delay(failures, policy)
            _log_stream_retry(
                ws,
                phase="llm_stream_retry",
                detail={
                    "reason": reason,
                    "attempt": failures,
                    "max_attempts": policy.max_attempts,
                    "delay_sec": delay,
                    "error_type": type(exc).__name__,
                },
            )
            from llgraph.terminal.ops_notice import ops_notice

            ops_notice(
                f"网关瞬时故障（{reason}），{delay:.1f}s 后重试 "
                f"{failures}/{policy.max_attempts - 1}"
            )
            _sleep_with_cancel(delay)


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
