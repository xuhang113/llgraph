"""Kimi K2 原生 tool call token → AIMessage.tool_calls。"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

from llgraph.adapters.inbound.streaming import parse_tool_args_json

_KIMI_TOOL_CALL_PATTERN = re.compile(
    r"<\|(?:redacted_)?tool_call_begin(?:_kimi)?\|>\s*"
    r"(?P<tool_call_id>[^\s<|]+)\s*"
    r"<\|(?:redacted_)?tool_call_argument_begin\|>\s*"
    r"(?P<function_arguments>\{.*?\})\s*"
    r"<\|(?:redacted_)?tool_call_end(?:_kimi)?\|>",
    re.DOTALL,
)
_KIMI_SECTION_PATTERN = re.compile(
    r"<\|(?:redacted_)?tool_calls_section_begin\|>.*?<\|(?:redacted_)?tool_calls_section_end\|>",
    re.DOTALL,
)
_KIMI_SECTION_BEGIN_MARKERS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_begin|>",
)
_KIMI_STRIP_TOKENS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_begin_kimi|>",
    "<|tool_call_end_kimi|>",
    "<|tool_call_argument_begin|>",
    "<|tool_call_argument_begin|>",
)


def ai_message_text_content(msg: AIMessage) -> str:
    """
    合并 AIMessage 中 type:text 块为字符串。

    @param msg assistant 消息
    @return 文本
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def content_has_kimi_tool_tokens(text: str) -> bool:
    """content 是否含 Kimi 原生 tool call 标记。"""
    if not text:
        return False
    return "<|tool_call" in text or "<|redacted_tool_call" in text


_PLAIN_FUNCTIONS_HEAD = re.compile(r"functions\.[A-Za-z0-9_]+:\d+")


def _skip_json_object(text: str, start: int) -> int:
    """
    从 start 处的 ``{`` 起跳过完整 JSON 对象。

    @param text 全文
    @param start ``{`` 下标
    @return 对象结束后的下标
    """
    if start >= len(text) or text[start] != "{":
        return start
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return len(text)


def strip_plain_functions_tool_calls(text: str) -> str:
    """
    剥离去 token 后的 ``functions.tool_name:1{...}`` 泄漏（Web 截图常见）。

    @param text 原始正文
    @return 清理后文本
    """
    if not text or "functions." not in text:
        return text
    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        match = _PLAIN_FUNCTIONS_HEAD.search(text, cursor)
        if not match:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor : match.start()])
        end = match.end()
        if end < len(text) and text[end] == "{":
            end = _skip_json_object(text, end)
        cursor = end
    cleaned = "".join(parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_kimi_tool_call_markup(text: str) -> str:
    """
    从文本中移除 Kimi 原生 tool call 特殊 token。

    @param text 原始文本
    @return 清理后文本
    """
    if not text:
        return ""
    if not content_has_kimi_tool_tokens(text):
        return text
    cleaned = _KIMI_SECTION_PATTERN.sub("", text)
    cleaned = _KIMI_TOOL_CALL_PATTERN.sub("", cleaned)
    for token in _KIMI_STRIP_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _parse_kimi_function_id(function_id: str) -> tuple[str, str]:
    fid = function_id.strip()
    if fid.startswith("functions."):
        rest = fid[len("functions.") :]
        if ":" in rest:
            name = rest.rsplit(":", 1)[0].strip()
            return name or fid, fid
        return rest.strip() or fid, fid
    if ":" in fid:
        name = fid.rsplit(":", 1)[0].strip()
        return name or fid, fid
    return fid, fid


def _parse_plain_functions_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """
    解析无 Kimi 特殊 token 的 ``functions.name:id{args}`` 串。

    @param text AIMessage 文本
    @return (tool_calls, 清理后正文)
    """
    if not text or "functions." not in text:
        return [], text
    if content_has_kimi_tool_tokens(text):
        return [], text
    calls: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        match = _PLAIN_FUNCTIONS_HEAD.search(text, cursor)
        if not match:
            break
        function_id = match.group(0)
        args_start = match.end()
        args_raw = ""
        if args_start < len(text) and text[args_start] == "{":
            args_end = _skip_json_object(text, args_start)
            args_raw = text[args_start:args_end]
            cursor = args_end
        else:
            cursor = args_start
        name, cid = _parse_kimi_function_id(function_id)
        args = parse_tool_args_json(args_raw) if args_raw else {}
        calls.append(
            {
                "name": name,
                "args": args,
                "id": cid,
                "type": "tool_call",
            }
        )
    if not calls:
        return [], text
    cleaned = strip_plain_functions_tool_calls(text)
    return calls, cleaned


def parse_kimi_native_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """
    从 Kimi K2 content 文本解析 tool_calls。

    @param text AIMessage 文本
    @return (tool_calls, 清理后正文)
    """
    if content_has_kimi_tool_tokens(text):
        prefix = text
        for marker in _KIMI_SECTION_BEGIN_MARKERS:
            idx = text.find(marker)
            if idx >= 0:
                prefix = text[:idx]
                break

        calls: list[dict[str, Any]] = []
        for match in _KIMI_TOOL_CALL_PATTERN.finditer(text):
            function_id = match.group("tool_call_id").strip()
            args_raw = match.group("function_arguments").strip()
            name, cid = _parse_kimi_function_id(function_id)
            args = parse_tool_args_json(args_raw) or {}
            calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": cid,
                    "type": "tool_call",
                }
            )

        tail = strip_kimi_tool_call_markup(text[len(prefix) :])
        normal_text = f"{prefix}{tail}".strip() if tail else prefix.strip()
        return calls, normal_text

    plain_calls, plain_text = _parse_plain_functions_tool_calls(text)
    if plain_calls:
        return plain_calls, plain_text
    return [], text


def _apply_cleaned_text_to_content(content: Any, cleaned_text: str) -> Any:
    if isinstance(content, str):
        return cleaned_text
    if isinstance(content, list):
        new_blocks: list[Any] = []
        text_replaced = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                if text_replaced:
                    continue
                if cleaned_text:
                    new_blocks.append({**block, "text": cleaned_text})
                text_replaced = True
            else:
                new_blocks.append(block)
        if not text_replaced and cleaned_text:
            new_blocks.insert(0, {"type": "text", "text": cleaned_text})
        return new_blocks if new_blocks else ""
    return cleaned_text


def normalize_plain_functions_tool_calls(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    入站：``functions.name:id{args}`` → tool_calls（全模型；网关泄漏兜底）。

    @param msg 原始 AIMessage
    @return (归一化消息, 是否改写)
    """
    text = ai_message_text_content(msg)
    if not text or "functions." not in text:
        return msg, False

    if content_has_kimi_tool_tokens(text):
        from llgraph.adapters.inbound.xml_tool_call import strip_inbound_tool_call_markup

        stripped = strip_inbound_tool_call_markup(text)
        if stripped != text.strip():
            new_content = _apply_cleaned_text_to_content(msg.content, stripped)
            return msg.model_copy(update={"content": new_content}), True
        return msg, False

    if msg.tool_calls:
        stripped = strip_plain_functions_tool_calls(text)
        if stripped != text.strip():
            new_content = _apply_cleaned_text_to_content(msg.content, stripped)
            return msg.model_copy(update={"content": new_content}), True
        return msg, False

    plain_calls, cleaned_text = _parse_plain_functions_tool_calls(text)
    if not plain_calls:
        stripped = strip_plain_functions_tool_calls(text)
        if stripped != text.strip():
            new_content = _apply_cleaned_text_to_content(msg.content, stripped)
            return msg.model_copy(update={"content": new_content}), True
        return msg, False

    updates: dict[str, Any] = {"tool_calls": plain_calls}
    new_content = _apply_cleaned_text_to_content(msg.content, cleaned_text)
    if new_content != msg.content:
        updates["content"] = new_content
    return msg.model_copy(update=updates), True


def normalize_kimi_native_tool_calls(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    Kimi 入站：content 内 native token → tool_calls，并剥离 markup。

    @param msg 原始 AIMessage
    @return (归一化消息, 是否改写)
    """
    text = ai_message_text_content(msg)
    kimi_calls, cleaned_text = parse_kimi_native_tool_calls(text)
    if not kimi_calls:
        stripped = strip_plain_functions_tool_calls(strip_kimi_tool_call_markup(text))
        if stripped != text.strip():
            new_content = _apply_cleaned_text_to_content(msg.content, stripped)
            return msg.model_copy(update={"content": new_content}), True
        return msg, False

    updates: dict[str, Any] = {}
    if not msg.tool_calls:
        updates["tool_calls"] = kimi_calls
    new_content = _apply_cleaned_text_to_content(msg.content, cleaned_text)
    if new_content != msg.content:
        updates["content"] = new_content
    if not updates:
        return msg, False
    return msg.model_copy(update=updates), True
