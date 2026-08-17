"""网关 HTTP 出站前校验与修复 formatted messages（Kimi 等 OpenAI 兼容层）。"""

from __future__ import annotations

import logging
from typing import Any

from llgraph.context.chat_history_repair import (
    THINKING_ONLY_DISPATCH_TEXT,
    TOOL_ASSISTANT_DISPATCH_TEXT,
)

_LOG = logging.getLogger(__name__)


def _block_type(block: dict[str, Any]) -> str:
    return str(block.get("type", "")).lower()


def _normalize_list_block(block: Any) -> dict[str, Any] | None:
    """
    规范化单条 content 块。

    @param block 原始块
    @return 合法 dict 或 None（丢弃）
    """
    if not isinstance(block, dict):
        return None
    kind = _block_type(block)
    if kind == "text":
        raw = block.get("text")
        if raw is None:
            text = TOOL_ASSISTANT_DISPATCH_TEXT
        elif isinstance(raw, str):
            text = raw if raw.strip() not in ("", " ") else TOOL_ASSISTANT_DISPATCH_TEXT
        else:
            text = str(raw)
        return {"type": "text", "text": text}
    if kind in ("thinking", "redacted_thinking"):
        thinking = (
            block.get("thinking")
            or block.get("reasoning")
            or block.get("text")
            or block.get("data")
        )
        if not thinking:
            return None
        out: dict[str, Any] = {"type": "thinking", "thinking": str(thinking)}
        signature = block.get("signature")
        if signature is not None:
            out["signature"] = signature
        return out
    if kind == "tool_use":
        return dict(block)
    if kind == "tool_result":
        inner = block.get("content")
        if inner is not None and not isinstance(inner, (str, list)):
            block = {**block, "content": str(inner)}
        return dict(block)
    return dict(block)


def _assistant_needs_text_block(blocks: list[dict[str, Any]]) -> bool:
    has_text = any(
        _block_type(b) == "text" and str(b.get("text", "")).strip()
        for b in blocks
    )
    if has_text:
        return False
    return any(
        _block_type(b) in ("thinking", "redacted_thinking", "tool_use")
        for b in blocks
    )


def _repair_assistant_content(content: Any, *, index: int, issues: list[str]) -> Any:
    """
    修复 assistant content 为网关可接受的 str 或 block 列表。

    @param content 原始 content
    @param index message 下标（日志用）
    @param issues 累积问题描述
    @return 修复后的 content
    """
    if content is None:
        issues.append(f"messages[{index}] assistant content was null")
        return [{"type": "text", "text": TOOL_ASSISTANT_DISPATCH_TEXT}]
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped or stripped == " ":
            issues.append(f"messages[{index}] assistant content empty string")
            return [{"type": "text", "text": TOOL_ASSISTANT_DISPATCH_TEXT}]
        return content
    if not isinstance(content, list):
        issues.append(f"messages[{index}] assistant content type {type(content).__name__}")
        return str(content)

    blocks: list[dict[str, Any]] = []
    for raw in content:
        fixed = _normalize_list_block(raw)
        if fixed is not None:
            blocks.append(fixed)
    if _assistant_needs_text_block(blocks):
        has_tool = any(_block_type(b) == "tool_use" for b in blocks)
        placeholder = TOOL_ASSISTANT_DISPATCH_TEXT if has_tool else THINKING_ONLY_DISPATCH_TEXT
        blocks.append({"type": "text", "text": placeholder})
        issues.append(f"messages[{index}] assistant missing text block")
    if not blocks:
        issues.append(f"messages[{index}] assistant content list empty")
        return [{"type": "text", "text": TOOL_ASSISTANT_DISPATCH_TEXT}]
    return blocks


def _repair_user_content(content: Any, *, index: int, issues: list[str]) -> Any:
    """修复 user / tool_result 聚合消息的 content。"""
    if content is None:
        issues.append(f"messages[{index}] content was null")
        return ""
    if isinstance(content, (str, list)):
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for raw in content:
                fixed = _normalize_list_block(raw)
                if fixed is not None:
                    blocks.append(fixed)
            return blocks or ""
        return content
    issues.append(f"messages[{index}] content type {type(content).__name__}")
    return str(content)


def validate_and_repair_formatted_messages(
    formatted_messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    扫描并修复 Anthropic formatted messages，避免 Kimi 网关 400。

    @param formatted_messages 即将提交的 messages
    @return (修复后列表, 问题描述)
    """
    issues: list[str] = []
    repaired: list[dict[str, Any]] = []
    for index, block in enumerate(formatted_messages):
        if not isinstance(block, dict):
            issues.append(f"messages[{index}] not a dict")
            continue
        role = str(block.get("role", ""))
        out = dict(block)
        content = block.get("content")
        if role == "assistant":
            out["content"] = _repair_assistant_content(content, index=index, issues=issues)
        else:
            out["content"] = _repair_user_content(content, index=index, issues=issues)
        repaired.append(out)

    if issues:
        preview = "; ".join(issues[:5])
        if len(issues) > 5:
            preview += f"; +{len(issues) - 5} more"
        _LOG.warning("dispatch payload repaired: %s", preview)
    return repaired, issues
