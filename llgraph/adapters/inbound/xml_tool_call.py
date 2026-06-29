"""XML ``<tool_call>`` 入站解析（Qwen / 网关未结构化时的兜底）。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from llgraph.adapters.inbound.kimi_native import (
    _apply_cleaned_text_to_content,
    ai_message_text_content,
)
from llgraph.adapters.inbound.streaming import parse_tool_args_json

_XML_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call(?:\s[^>]*)?>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_RE_TOOL_CALL_NAME = re.compile(
    r"<\s*tool_call_name\s*>\s*(.*?)\s*</\s*tool_call_name\s*>",
    re.DOTALL | re.IGNORECASE,
)
_RE_TOOL_CALL_ARGS = re.compile(
    r"<\s*tool_call_args\s*>\s*(.*?)\s*</\s*tool_call_args\s*>",
    re.DOTALL | re.IGNORECASE,
)
_RE_ARGUMENT_TAG = re.compile(
    r"<\s*argument\s+name\s*=\s*[\"']([^\"']+)[\"']"
    r"(?:\s+string\s*=\s*[\"']([^\"']*)[\"'])?\s*>\s*"
    r"(.*?)\s*</\s*argument\s*>",
    re.DOTALL | re.IGNORECASE,
)
_RE_TOOL_CALL_ID = re.compile(
    r"<\s*tool_call_id\s*>\s*(.*?)\s*</\s*tool_call_id\s*>",
    re.DOTALL | re.IGNORECASE,
)
_RE_FUNCTION_OPEN = re.compile(
    r"<function(?:=([^>\s]+)|[\s]+name=[\"']([^\"']+)[\"'])[\s]*>",
    re.IGNORECASE,
)
_RE_PARAMETER = re.compile(
    r"<\s*parameter\s*=\s*([^>]*?)>\s*"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=)|(?=<\s*/\s*function\s*>)|(?=<\s*/\s*tool_call\s*>)|$)",
    re.DOTALL | re.IGNORECASE,
)
_RE_ARG_KEY = re.compile(r"<\s*arg_key\s*>\s*(.*?)\s*</\s*arg_key\s*>", re.DOTALL | re.IGNORECASE)
_RE_ARG_VALUE = re.compile(
    r"<\s*arg_value\s*>\s*(.*?)\s*</\s*arg_value\s*>",
    re.DOTALL | re.IGNORECASE,
)
_RE_JSON_OBJECT = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def content_has_xml_tool_calls(text: str) -> bool:
    """content 是否含 XML ``<tool_call>`` 块。"""
    if not text:
        return False
    return bool(
        re.search(r"<\s*tool_call(?:\s[^>]*)?\s*>", text, re.IGNORECASE)
        or re.search(r"<\s*tool_calls\s*>", text, re.IGNORECASE)
    )


def strip_xml_tool_call_markup(text: str) -> str:
    """
    从文本中移除 XML ``<tool_call>`` 块。

    @param text 原始文本
    @return 清理后文本
    """
    if not text or not content_has_xml_tool_calls(text):
        return text
    cleaned = _XML_TOOL_CALL_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"<\s*/\s*tool_call\s*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*tool_calls\s*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*/\s*tool_calls\s*>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _parse_scalar_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    parsed = parse_tool_args_json(text)
    if parsed is not None:
        return parsed
    return text


def _coerce_tool_call(name: str, args: dict[str, Any], index: int) -> dict[str, Any]:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_") or "tool"
    return {
        "name": name.strip(),
        "args": args,
        "id": f"functions.{safe}:{index}",
        "type": "tool_call",
    }


def _parse_json_tool_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    candidate = inner.strip()
    if not candidate:
        return None
    for text in (candidate, *_RE_JSON_OBJECT.findall(candidate)):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name: str | None = None
        fn_field = obj.get("function")
        if isinstance(fn_field, str) and fn_field.strip():
            name = fn_field.strip()
        elif isinstance(fn_field, dict):
            raw_name = fn_field.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                name = raw_name.strip()
        if not name:
            raw = obj.get("name")
            if isinstance(raw, str) and raw.strip():
                name = raw.strip()
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters")
        if isinstance(args, str):
            parsed = parse_tool_args_json(args) or {}
            args = parsed
        if not isinstance(args, dict):
            args = {}
        if name:
            return name, dict(args)
    return None


def _parse_qwen_xml_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    fn_match = _RE_FUNCTION_OPEN.search(inner)
    if not fn_match:
        return None
    name = (fn_match.group(1) or fn_match.group(2) or "").strip()
    if not name:
        return None
    args: dict[str, Any] = {}
    for param_match in _RE_PARAMETER.finditer(inner):
        key = (param_match.group(1) or "").strip()
        if not key:
            continue
        args[key] = _parse_scalar_value(param_match.group(2) or "")
    return name, args


def _parse_arg_key_value_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    keys = [k.strip() for k in _RE_ARG_KEY.findall(inner) if k.strip()]
    vals = [v for v in _RE_ARG_VALUE.findall(inner)]
    if not keys or len(keys) != len(vals):
        return None
    fn_match = _RE_FUNCTION_OPEN.search(inner)
    name = ""
    if fn_match:
        name = (fn_match.group(1) or fn_match.group(2) or "").strip()
    if not name:
        for line in inner.splitlines():
            text = line.strip()
            if not text or text.startswith("<"):
                continue
            if text.startswith("{") or text.startswith("["):
                break
            name = text.split()[0]
            break
    if not name:
        return None
    args = {k: _parse_scalar_value(v) for k, v in zip(keys, vals)}
    return name, args


def _parse_deepseek_anthropic_call(
    inner: str,
    *,
    attr_name: str | None = None,
) -> tuple[str, dict[str, Any], str | None] | None:
    """
    DeepSeek Anthropic 兼容面：``tool_call_args`` JSON 或 ``<argument name=…>`` 标签。

    @param inner tool_call 块正文
    @param attr_name 开标签 ``name=`` 属性（若有）
    @return (工具名, args, tool_call_id) 或 None
    """
    name = (attr_name or "").strip()
    name_match = _RE_TOOL_CALL_NAME.search(inner)
    if name_match and name_match.group(1).strip():
        name = name_match.group(1).strip()
    if not name:
        return None

    args: dict[str, Any] | None = None
    args_match = _RE_TOOL_CALL_ARGS.search(inner)
    if args_match is not None:
        args_raw = (args_match.group(1) or "").strip()
        parsed = parse_tool_args_json(args_raw)
        if parsed is None:
            try:
                loaded = json.loads(args_raw)
                parsed = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                parsed = {}
        args = dict(parsed)

    if args is None:
        arg_tags: dict[str, Any] = {}
        for match in _RE_ARGUMENT_TAG.finditer(inner):
            key = (match.group(1) or "").strip()
            if not key:
                continue
            string_flag = (match.group(2) or "").strip().lower()
            if string_flag == "true":
                value: Any = (match.group(3) or "").strip()
            else:
                value = _parse_scalar_value(match.group(3) or "")
            arg_tags[key] = value
        if arg_tags:
            args = arg_tags

    if args is None:
        return None

    id_match = _RE_TOOL_CALL_ID.search(inner)
    tool_id = id_match.group(1).strip() if id_match and id_match.group(1).strip() else None
    return name, args, tool_id


def _parse_plain_name_json_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    lines = [ln.strip() for ln in inner.strip().splitlines() if ln.strip()]
    if not lines or lines[0].startswith("<"):
        return None
    first = lines[0]
    if first.startswith("{") or first.startswith("["):
        return None
    name = first.split()[0]
    if not name or "=" in name:
        return None
    tail = "\n".join(lines[1:]).strip()
    if not tail:
        return None
    parsed = _parse_json_tool_call(tail)
    if parsed is None:
        args = parse_tool_args_json(tail)
        if args is None:
            return None
        return name, args
    return parsed[0] if parsed[0] == name else name, parsed[1]


def _parse_single_tool_call_inner(
    inner: str,
    index: int,
    *,
    attr_name: str | None = None,
) -> dict[str, Any] | None:
    body = inner.strip()
    if not body:
        return None
    deepseek = _parse_deepseek_anthropic_call(body, attr_name=attr_name)
    if deepseek is not None:
        name, args, tool_id = deepseek
        if name:
            item = _coerce_tool_call(name, args, index)
            if tool_id:
                item["id"] = tool_id
            return item
    for parser in (
        _parse_json_tool_call,
        _parse_qwen_xml_call,
        _parse_arg_key_value_call,
        _parse_plain_name_json_call,
    ):
        parsed = parser(body)
        if parsed is not None:
            name, args = parsed
            if name:
                return _coerce_tool_call(name, args, index)
    return None


def parse_xml_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """
    从 content 文本解析 XML ``<tool_call>`` 块。

    支持 Qwen ``<function=…><parameter=…>``、``arg_key``/``arg_value``、
    首行工具名 + JSON 参数、块内 JSON 等常见变体。

    @param text AIMessage 文本
    @return (tool_calls, 清理后正文)
    """
    if not content_has_xml_tool_calls(text):
        return [], text

    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_XML_TOOL_CALL_BLOCK_RE.finditer(text)):
        open_tag = text[match.start() : match.start() + (match.group(0).find(">") + 1)]
        attr_name: str | None = None
        attr_match = re.search(
            r'name\s*=\s*["\']([^"\']+)["\']',
            open_tag,
            re.IGNORECASE,
        )
        if attr_match:
            attr_name = attr_match.group(1).strip()
        item = _parse_single_tool_call_inner(
            match.group(1) or "",
            index,
            attr_name=attr_name,
        )
        if item is not None:
            calls.append(item)

    first = _XML_TOOL_CALL_BLOCK_RE.search(text)
    if first is None:
        return calls, strip_xml_tool_call_markup(text)
    prefix = text[: first.start()]
    tail = strip_xml_tool_call_markup(text[first.start() :])
    normal_text = strip_xml_tool_call_markup(f"{prefix}{tail}".strip())
    return calls, normal_text


def normalize_xml_tool_calls(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    XML 入站：content 内 ``<tool_call>`` → tool_calls，并剥离 markup。

    @param msg 原始 AIMessage
    @return (归一化消息, 是否改写)
    """
    text = ai_message_text_content(msg)
    xml_calls, cleaned_text = parse_xml_tool_calls(text)
    if not xml_calls:
        stripped = strip_xml_tool_call_markup(text)
        if stripped != text.strip():
            new_content = _apply_cleaned_text_to_content(msg.content, stripped)
            return msg.model_copy(update={"content": new_content}), True
        return msg, False

    updates: dict[str, Any] = {}
    if not msg.tool_calls:
        updates["tool_calls"] = xml_calls
    new_content = _apply_cleaned_text_to_content(msg.content, cleaned_text)
    if new_content != msg.content:
        updates["content"] = new_content
    if not updates:
        return msg, False
    return msg.model_copy(update=updates), True


def strip_inbound_tool_call_markup(text: str) -> str:
    """剥离 Kimi token、plain ``functions.*`` 与 XML ``<tool_call>`` markup（展示/流式用）。"""
    from llgraph.adapters.inbound.kimi_native import (
        strip_kimi_tool_call_markup,
        strip_plain_functions_tool_calls,
    )

    return strip_plain_functions_tool_calls(
        strip_kimi_tool_call_markup(strip_xml_tool_call_markup(text)),
    )
