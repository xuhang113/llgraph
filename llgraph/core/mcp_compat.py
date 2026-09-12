"""MCP SDK 字段命名兼容：1.x 是 camelCase，2.x 全改成 snake_case。

`pyproject` 允许 `mcp>=1.0.0`，所以同一份代码会同时遇到两套命名：

| 含义 | 1.x | 2.x |
|---|---|---|
| 工具入参 schema | `Tool.inputSchema` | `Tool.input_schema` |
| 调用是否失败 | `CallToolResult.isError` | `CallToolResult.is_error` |
| 结构化返回 | `structuredContent` | `structured_content` |

这些都是 pydantic 模型，读不存在的字段抛 `AttributeError` 而不是给 None，
于是「取不到」会沿着 except 分支变成一句调用失败。其中 `isError` 最危险：
服务端**已经执行成功**的调用被报回模型说失败，模型就会重试一次已经生效的写操作。
本模块统一按两套命名取值，取不到才落到保守默认。
"""

from __future__ import annotations

import json
from typing import Any


def _first_attr(obj: object, names: tuple[str, ...], default: Any) -> Any:
    """
    按顺序取第一个真实存在的属性。

    @param obj 目标对象
    @param names 候选属性名（按优先级）
    @param default 全部取不到时的返回
    @return 属性值或 default
    """
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        if value is not None:
            return value
    return default


def tool_input_schema(mcp_tool: object) -> dict[str, Any]:
    """
    读取 MCP 工具的入参 JSON Schema。

    取不到时返回空字典：调用方会退回 `arguments_json` 文本入参，
    模型仍能调用，只是丢掉结构化字段。

    @param mcp_tool MCP `Tool` 对象
    @return JSON Schema 字典；无法解析时为空字典
    """
    schema = _first_attr(mcp_tool, ("input_schema", "inputSchema"), None)
    return schema if isinstance(schema, dict) else {}


def tool_description(mcp_tool: object) -> str:
    """
    读取 MCP 工具描述。

    @param mcp_tool MCP `Tool` 对象
    @return 描述文本；缺失时退回工具名
    """
    desc = _first_attr(mcp_tool, ("description",), "")
    if isinstance(desc, str) and desc.strip():
        return desc
    name = _first_attr(mcp_tool, ("name",), "")
    return str(name)


def _structured_text(result: object) -> str:
    structured = _first_attr(
        result, ("structured_content", "structuredContent"), None
    )
    if structured is None:
        return ""
    try:
        return json.dumps(structured, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(structured)


def render_call_result(result: object) -> tuple[str, bool]:
    """
    把 `CallToolResult` 渲染成文本，并给出「服务端是否报错」。

    内容块为空时退回结构化返回（2.x 的 `structured_content`），
    避免只给结构化输出的工具被渲染成空字符串。

    @param result MCP `CallToolResult` 对象
    @return (正文, 是否服务端报错)
    """
    is_error = bool(_first_attr(result, ("is_error", "isError"), False))
    blocks = _first_attr(result, ("content",), None) or []
    parts: list[str] = []
    for block in blocks:
        text = _first_attr(block, ("text",), None)
        parts.append(str(text) if text is not None else str(block))
    body = "".join(parts)
    if not body.strip():
        body = _structured_text(result) or body
    return body, is_error
