"""修复 Kimi 等流式网关产生的畸形 tool_calls（空名、空 args、invalid 碎片）。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, RemoveMessage

_RE_TOOL_ID = re.compile(r"[^a-zA-Z0-9_]+")


def _normalize_tool_call_id(tool_call_id: str) -> str:
    """
    归一化 tool_call_id 便于 invalid 碎片与正式 call 对齐。

    @param tool_call_id 原始 id
    @return 仅含字母数字下划线的小写串
    """
    return _RE_TOOL_ID.sub("_", str(tool_call_id or "").strip()).strip("_").lower()


def _invalid_call_id(entry: dict[str, Any]) -> str:
    raw = entry.get("id")
    return str(raw).strip() if raw is not None else ""


def _reconstruct_args_from_invalid(invalid_calls: list[Any]) -> dict[str, dict[str, Any]]:
    """
    将 invalid_tool_calls 按 id 拼接 args 碎片并解析 JSON。

    @param invalid_calls LangChain invalid_tool_calls 列表
    @return 归一化 id -> 参数字典
    """
    fragments: dict[str, list[str]] = {}
    for entry in invalid_calls:
        if not isinstance(entry, dict):
            continue
        cid = _invalid_call_id(entry)
        frag = entry.get("args")
        if isinstance(frag, str):
            text = frag
        elif frag is None:
            continue
        else:
            text = str(frag)
        fragments.setdefault(cid, []).append(text)

    out: dict[str, dict[str, Any]] = {}
    for cid, parts in fragments.items():
        merged = "".join(parts).strip()
        if not merged:
            continue
        parsed = _parse_tool_args_json(merged)
        if parsed is not None:
            out[_normalize_tool_call_id(cid)] = parsed
    return out


def _parse_tool_args_json(text: str) -> dict[str, Any] | None:
    """
    解析工具参数字符串（容忍缺外层花括号）。

    @param text 拼接后的 JSON 或碎片
    @return 参数字典；无法解析则 None
    """
    raw = text.strip()
    if not raw:
        return None
    candidates = [raw]
    if not raw.startswith("{") and not raw.endswith("}"):
        candidates.append(f"{{{raw}}}")
    if not raw.startswith("{") and ":" in raw:
        candidates.append(f"{{{raw}}}")
        candidates.append('{"' + raw)
    if raw.startswith("{") and not raw.endswith("}"):
        candidates.append(raw + "}")
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _coerce_tool_call(call: Any) -> dict[str, Any]:
    if isinstance(call, dict):
        return dict(call)
    return {
        "name": getattr(call, "name", ""),
        "args": getattr(call, "args", {}),
        "id": getattr(call, "id", ""),
        "type": getattr(call, "type", "tool_call"),
    }


def repair_streaming_tool_calls_on_ai_message(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    清理 AIMessage 上畸形 tool_calls：去空名、去重 id、从 invalid 碎片恢复 args。

    @param msg assistant 消息
    @return (修复后消息, 是否改写)
    """
    invalid = list(getattr(msg, "invalid_tool_calls", None) or [])
    calls = list(msg.tool_calls or [])
    if not calls and not invalid:
        return msg, False

    reconstructed = _reconstruct_args_from_invalid(invalid)
    seen_ids: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    changed = bool(invalid)

    for call in calls:
        item = _coerce_tool_call(call)
        name = str(item.get("name") or "").strip()
        cid = str(item.get("id") or "").strip()

        if not name:
            changed = True
            continue

        if cid:
            if cid in seen_ids:
                changed = True
                continue
            seen_ids.add(cid)

        args = item.get("args")
        if not isinstance(args, dict):
            args = {}

        if not args and cid and reconstructed:
            norm = _normalize_tool_call_id(cid)
            if norm in reconstructed:
                args = dict(reconstructed[norm])
                changed = True
            else:
                for key, parsed in reconstructed.items():
                    if norm.endswith(key) or key.endswith(norm):
                        args = dict(parsed)
                        changed = True
                        break

        item["name"] = name
        item["args"] = args
        if not item.get("type"):
            item["type"] = "tool_call"
        cleaned.append(item)

    if not changed and cleaned == calls and not invalid:
        return msg, False

    return msg.model_copy(
        update={
            "tool_calls": cleaned,
            "invalid_tool_calls": [],
        },
    ), True


def repair_tool_calls_post_model_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph post_model_hook：修复本轮 AIMessage 的 tool_calls 再进入 tools 节点。

    @param state ReAct 图状态
    @return 状态增量；无改动时返回空 dict
    """
    messages = list(state.get("messages") or [])
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {}

    repaired, changed = repair_streaming_tool_calls_on_ai_message(last)
    if not changed:
        return {}

    if getattr(last, "id", None):
        return {"messages": [RemoveMessage(id=last.id), repaired]}
    return {"messages": [repaired]}
