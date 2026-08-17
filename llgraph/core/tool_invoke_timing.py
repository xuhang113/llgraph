"""单工具调用耗时（ToolNode 并行执行时按 tool_call_id 记录）。"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import ToolMessage

from llgraph.context.tool_call_id import canonical_tool_call_id

_tool_timings: ContextVar[dict[str, float] | None] = ContextVar(
    "llgraph_tool_timings",
    default=None,
)


def reset_tool_timings() -> None:
    """工具节点开始前清空计时表。"""
    _tool_timings.set({})


def record_tool_timing(tool_call_id: str, elapsed_sec: float) -> None:
    """
    记录单次工具调用耗时。

    @param tool_call_id 工具调用 ID
    @param elapsed_sec 耗时（秒）
    """
    cid = str(tool_call_id or "").strip()
    if not cid or elapsed_sec < 0:
        return
    timings = _tool_timings.get()
    if timings is None:
        timings = {}
        _tool_timings.set(timings)
    timings[cid] = elapsed_sec
    timings[canonical_tool_call_id(cid)] = elapsed_sec


def lookup_tool_timing(tool_call_id: str) -> float | None:
    """
    读取已记录的工具耗时。

    @param tool_call_id 工具调用 ID
    @return 秒数；未记录时 None
    """
    timings = _tool_timings.get() or {}
    cid = str(tool_call_id or "").strip()
    if not cid:
        return None
    if cid in timings:
        return timings[cid]
    return timings.get(canonical_tool_call_id(cid))


def read_tool_message_elapsed(msg: ToolMessage) -> float | None:
    """
    从 ToolMessage.response_metadata 读取单工具耗时。

    @param msg 工具消息
    @return 秒数；无记录时 None
    """
    meta = getattr(msg, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    raw = meta.get("elapsed_sec")
    if isinstance(raw, (int, float)) and raw >= 0:
        return float(raw)
    return None


def attach_tool_timings_to_output(out: dict[str, Any]) -> dict[str, Any]:
    """
    将 ContextVar 中的耗时写入 ToolMessage.response_metadata。

    @param out ToolNode 返回的状态片段
    @return 附带 elapsed_sec 的输出
    """
    timings = _tool_timings.get() or {}
    if not timings:
        return out
    msgs: list[Any] = []
    for msg in out.get("messages") or []:
        if not isinstance(msg, ToolMessage):
            msgs.append(msg)
            continue
        cid = str(getattr(msg, "tool_call_id", "") or "")
        elapsed = lookup_tool_timing(cid)
        if elapsed is None:
            msgs.append(msg)
            continue
        meta = dict(getattr(msg, "response_metadata", None) or {})
        meta["elapsed_sec"] = round(elapsed, 4)
        msgs.append(msg.model_copy(update={"response_metadata": meta}))
    return {**out, "messages": msgs}


def wrap_tool_node_with_timing(inner: Any) -> None:
    """
    为 LangGraph ToolNode 注入按 tool_call 计时的 _run_one / _arun_one 包装。

    @param inner ToolNode 实例（原地修改）
    """
    if getattr(inner, "_llgraph_timing_wrapped", False):
        return

    original_run = inner._run_one
    original_arun = inner._arun_one

    def timed_run_one(call: dict[str, Any], input_type: Any, tool_runtime: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_run(call, input_type, tool_runtime)
        finally:
            record_tool_timing(str(call.get("id") or ""), time.perf_counter() - started)

    async def timed_arun_one(call: dict[str, Any], input_type: Any, tool_runtime: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original_arun(call, input_type, tool_runtime)
        finally:
            record_tool_timing(str(call.get("id") or ""), time.perf_counter() - started)

    inner._run_one = timed_run_one  # type: ignore[method-assign]
    inner._arun_one = timed_arun_one  # type: ignore[method-assign]
    inner._llgraph_timing_wrapped = True  # type: ignore[attr-defined]
