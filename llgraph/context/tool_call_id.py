"""tool_call_id 规范化（inbound 流式修复 / 历史修链 / 出站网关共用）。"""

from __future__ import annotations

import re

_TOOL_CALL_ID_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def normalize_tool_call_id_raw(tool_call_id: object | None) -> str | None:
    """
    去空白后的原始 id；空则 None。

    @param tool_call_id 原始 id
    @return 去空白字符串或 None
    """
    if tool_call_id is None:
        return None
    text = str(tool_call_id).strip()
    return text or None


def canonical_tool_call_id(tool_call_id: object | None) -> str:
    """
    统一 tool_call_id 匹配键（functions.foo:2 与 functions_foo_2 等价）。

    @param tool_call_id 原始 id
    @return 规范化 id
    """
    text = normalize_tool_call_id_raw(tool_call_id)
    if not text:
        return "tool_call"
    safe = _TOOL_CALL_ID_UNSAFE.sub("_", text)
    return safe.replace(".", "_") or "tool_call"


def gateway_safe_tool_call_id(tool_call_id: object | None) -> str:
    """
    将 tool_call_id 规范为 Anthropic/Claude 网关可接受字符集。

    @param tool_call_id 原始 id
    @return 规范化 id
    """
    return canonical_tool_call_id(tool_call_id)
