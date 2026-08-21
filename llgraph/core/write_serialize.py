"""同 path 文件工具串行：并行 tool_calls 时按声明顺序落地。

对标 Claude Code / Cursor：
- 不同文件的读仍并行；
- 同一相对路径上的 write_file / append_file / search_replace 按 tool_calls 顺序排队；
- 同一路径上的 read_file 与写工具也按声明顺序排队，避免读到半写入内容或用过期快照去改。
"""

from __future__ import annotations

import threading
from typing import Any

from llgraph.core.write_failure_tracker import WRITE_TOOL_NAMES

_WRITE_WAIT_TIMEOUT_SEC = 180.0
_PATH_SERIAL_READ_TOOLS = frozenset({"read_file"})


def normalize_write_path(raw: object) -> str:
    """
    归一化写工具 path，用于判断是否指向同一文件。

    @param raw 工具 args.path
    @return 相对路径键；空则空串
    """
    text = str(raw or "").strip().replace("\\", "/")
    if not text:
        return ""
    while "//" in text:
        text = text.replace("//", "/")
    text = text.lstrip("/")
    if text.startswith("./"):
        text = text[2:]
    return text or "."


def _call_as_dict(call: Any) -> dict[str, Any]:
    if isinstance(call, dict):
        return call
    args = getattr(call, "args", None)
    return {
        "name": getattr(call, "name", None),
        "id": getattr(call, "id", None),
        "args": args if isinstance(args, dict) else {},
    }


def write_path_from_call(call: dict[str, Any]) -> str | None:
    """
    若该 tool_call 是写工具则返回归一化 path，否则 None。

    @param call LangGraph / OpenAI 风格 tool_call dict
    """
    name = str(call.get("name") or "").strip()
    if name not in WRITE_TOOL_NAMES:
        return None
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    path = normalize_write_path(args.get("path"))
    return path or None


def touched_path_from_call(call: dict[str, Any]) -> str | None:
    """
    写工具或同文件 read_file 的归一化 path。

    @param call tool_call dict
    @return 路径；无关调用为 None
    """
    written = write_path_from_call(call)
    if written:
        return written
    name = str(call.get("name") or "").strip()
    if name not in _PATH_SERIAL_READ_TOOLS:
        return None
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    path = normalize_write_path(args.get("path"))
    if not path or path == ".":
        return None
    return path


class _WritePathGate:
    """按 path 把冲突调用排成队：后者等待前者 mark_done。"""

    def __init__(self, path_ids: dict[str, list[str]]) -> None:
        self._order = {path: list(ids) for path, ids in path_ids.items() if len(ids) > 1}
        self._done: set[str] = set()
        self._cond = threading.Condition()
        self._id_to_path: dict[str, str] = {}
        for path, ids in self._order.items():
            for cid in ids:
                self._id_to_path[cid] = path

    def wait_turn(self, call: dict[str, Any]) -> None:
        cid = str(call.get("id") or "").strip()
        path = self._id_to_path.get(cid)
        if not path:
            return
        ids = self._order.get(path) or []
        if cid not in ids:
            return
        idx = ids.index(cid)
        if idx == 0:
            return
        predecessor = ids[idx - 1]
        with self._cond:
            ok = self._cond.wait_for(
                lambda: predecessor in self._done,
                timeout=_WRITE_WAIT_TIMEOUT_SEC,
            )
            if not ok:
                # 超时放行，避免死锁；后者仍可能因内容未更新而失败
                self._done.add(predecessor)

    def mark_done(self, call: dict[str, Any]) -> None:
        cid = str(call.get("id") or "").strip()
        if not cid or cid not in self._id_to_path:
            return
        with self._cond:
            self._done.add(cid)
            self._cond.notify_all()


def gate_from_tool_calls(calls: list[Any]) -> _WritePathGate:
    """
    根据本批 tool_calls 构建同 path 读写串行门闩。

    仅当同一 path 上出现 2+ 次写或「读+写」时才排队；互不冲突的读仍并行。

    @param calls assistant 消息中的 tool_calls
    @return 门闩（无冲突时 wait 为 no-op）
    """
    path_ids: dict[str, list[str]] = {}
    for call in calls:
        item = _call_as_dict(call)
        path = touched_path_from_call(item)
        if not path:
            continue
        cid = str(item.get("id") or "").strip()
        if not cid:
            continue
        path_ids.setdefault(path, []).append(cid)
    return _WritePathGate(path_ids)


def install_write_serialize_gate(inner: Any, calls: list[Any]) -> None:
    """把本批 tool_calls 的写串行门闩挂到 ToolNode 实例上（跨线程可见）。"""
    inner._llgraph_write_gate = gate_from_tool_calls(calls)


def clear_write_serialize_gate(inner: Any) -> None:
    """清除 ToolNode 上的写串行门闩。"""
    inner._llgraph_write_gate = None


def wrap_tool_node_with_write_serialize(inner: Any) -> None:
    """
    包装 ToolNode._run_one / _arun_one：同 path 写/read_file 按 tool_calls 顺序执行。

    应在 wrap_tool_node_with_timing 之后调用，使等待时间不计入单工具耗时。
    门闩挂在 ToolNode 实例上，避免 ContextVar 在未 copy_context 的线程里丢失。

    @param inner ToolNode 实例（原地修改）
    """
    if getattr(inner, "_llgraph_write_serialize_wrapped", False):
        return

    inner._llgraph_write_gate = None
    original_run = inner._run_one
    original_arun = inner._arun_one

    def gated_run_one(call: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        gate = getattr(inner, "_llgraph_write_gate", None)
        if gate is not None:
            gate.wait_turn(call)
        try:
            return original_run(call, *args, **kwargs)
        finally:
            if gate is not None:
                gate.mark_done(call)

    async def gated_arun_one(call: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        gate = getattr(inner, "_llgraph_write_gate", None)
        if gate is not None:
            await _wait_turn_async(gate, call)
        try:
            return await original_arun(call, *args, **kwargs)
        finally:
            if gate is not None:
                gate.mark_done(call)

    inner._run_one = gated_run_one  # type: ignore[method-assign]
    inner._arun_one = gated_arun_one  # type: ignore[method-assign]
    inner._llgraph_write_serialize_wrapped = True  # type: ignore[attr-defined]


async def _wait_turn_async(gate: _WritePathGate, call: dict[str, Any]) -> None:
    import asyncio

    await asyncio.to_thread(gate.wait_turn, call)
