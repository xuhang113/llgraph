"""
跨模型会话历史 canonical v2 格式。

落盘目标（对齐多厂商最小公约）：
- OpenAI Chat Completions：assistant.tool_calls 后紧跟 role=tool，id 对齐；允许 1 AI + 多 Tool。
- Anthropic Messages：tool_use 与 tool_result 之间不得插入其它消息；tool_result 在 user 消息中（由 LangChain 转换）。
- 存储层：仅保留 LangGraph 原生 Human/AI/Tool + 置顶 System（manifest/anchor/archive），业务区不含 System。

参考：
- https://docs.anthropic.com/en/api/messages
- https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/handle-tool-calls
- https://developers.openai.com/api/docs/guides/function-calling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llgraph.context.conversation_anchor import is_conversation_anchor_message
from llgraph.context.message_normalize import _message_text
from llgraph.session.session_manifest import is_session_manifest_message

CANONICAL_FORMAT_VERSION = 2
_SESSION_ARCHIVE_TAG = "<session-system-archive>"
_LLGRAPH_META_KEY = "llgraph"


@dataclass
class CanonicalV2Report:
    """canonical v2 转换报告。"""

    flattened_ai_messages: int = 0
    archived_system_messages: int = 0
    removed_orphan_tools: int = 0
    normalized_ai_messages: int = 0
    patched_tool_results: int = 0

    @property
    def changed(self) -> bool:
        return (
            self.flattened_ai_messages > 0
            or self.archived_system_messages > 0
            or self.removed_orphan_tools > 0
            or self.normalized_ai_messages > 0
            or self.patched_tool_results > 0
        )


def is_session_archive_message(msg: BaseMessage) -> bool:
    """
    是否为合并后的中段 system 归档消息。

    @param msg LangChain 消息
    @return 是否归档 system
    """
    if not isinstance(msg, SystemMessage):
        return False
    return _SESSION_ARCHIVE_TAG in _message_text(getattr(msg, "content", ""))


def _extract_thinking_for_meta(msg: AIMessage) -> str:
    extra = getattr(msg, "additional_kwargs", None) or {}
    raw = extra.get("reasoning_content")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    content = getattr(msg, "content", "")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            kind = str(block.get("type", "")).lower()
            if kind in (
                "thinking",
                "reasoning",
                "reasoning_text",
                "redacted_thinking",
            ):
                text = (
                    block.get("thinking")
                    or block.get("reasoning")
                    or block.get("text")
                    or block.get("data")
                )
                if text:
                    parts.append(str(text))
    return "\n".join(parts).strip()


def persist_ai_thinking_in_message(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    将 content 内 thinking/reasoning 块落盘到 llgraph.thinking_text（出站前调用）。

    避免 ensure_nonempty 用占位符覆盖 content 时丢失 thinking。

    @param msg assistant 消息
    @return (消息, 是否改写)
    """
    return _flatten_ai_for_storage(msg)


def _flatten_ai_for_storage(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    将 AI 多段 content 收成纯文本；thinking 写入 additional_kwargs.llgraph。

    @param msg assistant 消息
    @return (消息, 是否改写)
    """
    thinking = _extract_thinking_for_meta(msg)
    content = getattr(msg, "content", "")
    texts: list[str] = []
    changed = False
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        changed = True
        for block in content:
            if isinstance(block, dict):
                kind = str(block.get("type", "")).lower()
                if kind in (
                    "thinking",
                    "reasoning",
                    "reasoning_text",
                    "tool_use",
                    "tool_calls",
                    "input_json_delta",
                ):
                    continue
                if kind == "text":
                    text = block.get("text")
                    if text:
                        texts.append(str(text))
            elif isinstance(block, str) and block.strip():
                texts.append(block)
    merged_text = "\n\n".join(texts).strip()
    extra = dict(getattr(msg, "additional_kwargs", None) or {})
    meta = dict(extra.get(_LLGRAPH_META_KEY) or {})
    if thinking:
        meta["thinking_text"] = thinking
        changed = True
    if meta:
        extra[_LLGRAPH_META_KEY] = meta
    extra.pop("reasoning_content", None)
    if not changed and not meta and isinstance(content, str):
        return msg, False
    return msg.model_copy(update={"content": merged_text, "additional_kwargs": extra}), True


def _build_archive_system(parts: list[str]) -> SystemMessage | None:
    """
    合并多条中段 system 为单条归档消息。

    @param parts 正文片段
    @return SystemMessage 或 None
    """
    merged = "\n\n---\n\n".join(p for p in parts if p.strip()).strip()
    if not merged:
        return None
    return SystemMessage(
        content=f"{_SESSION_ARCHIVE_TAG}\n\n{merged}",
    )


def to_canonical_v2_messages(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], CanonicalV2Report]:
    """
    转为 canonical v2：置顶 system + 无中段 system + 纯文本 AI + 合法 tool 链。

    @param messages 原始消息
    @return (canonical 消息列表, 报告)
    """
    from llgraph.context.chat_history_repair import rebuild_provider_safe_messages
    from llgraph.context.message_dispatch_profile import canonical_persist_profile

    report = CanonicalV2Report()
    if not messages:
        return messages, report

    safe, repair_report = rebuild_provider_safe_messages(
        messages,
        canonical_persist_profile(),
    )
    report.removed_orphan_tools = repair_report.removed_orphan_tools
    report.normalized_ai_messages = repair_report.normalized_ai_messages
    report.patched_tool_results = repair_report.patched_tool_results

    manifest: SystemMessage | None = None
    anchor: SystemMessage | None = None
    archive_parts: list[str] = []
    conversation: list[BaseMessage] = []

    for msg in safe:
        if isinstance(msg, SystemMessage):
            if is_session_manifest_message(msg):
                manifest = msg
                continue
            if is_conversation_anchor_message(msg):
                anchor = msg
                continue
            if is_session_archive_message(msg):
                text = _message_text(getattr(msg, "content", "")).strip()
                if text:
                    archive_parts.append(text)
                report.archived_system_messages += 1
                continue
            text = _message_text(getattr(msg, "content", "")).strip()
            if text:
                archive_parts.append(text)
                report.archived_system_messages += 1
            continue

        if isinstance(msg, AIMessage):
            flat, changed = _flatten_ai_for_storage(msg)
            if changed:
                report.flattened_ai_messages += 1
            conversation.append(flat)
            continue

        conversation.append(msg)

    ordered: list[BaseMessage] = []
    if manifest is not None:
        ordered.append(manifest)
    if anchor is not None:
        ordered.append(anchor)
    archive_msg = _build_archive_system(archive_parts)
    if archive_msg is not None:
        ordered.append(archive_msg)
    ordered.extend(conversation)
    return ordered if ordered else messages, report


def validate_canonical_v2_invariants(messages: list[BaseMessage]) -> list[str]:
    """
    校验 canonical v2 不变量。

    @param messages 消息列表
    @return 违规描述列表；空表示通过
    """
    from llgraph.context.chat_history_repair import ai_message_has_tool_calls

    issues: list[str] = []
    seen_conversation = False
    for idx, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            if seen_conversation:
                issues.append(f"index {idx}: 业务区出现 SystemMessage")
            if (
                not is_session_manifest_message(msg)
                and not is_conversation_anchor_message(msg)
                and not is_session_archive_message(msg)
            ):
                issues.append(f"index {idx}: 未标记的 SystemMessage")
            continue
        seen_conversation = True
        if isinstance(msg, ToolMessage):
            if idx == 0:
                issues.append("index 0: 首条为 ToolMessage")
                continue
            prev = messages[idx - 1]
            if not isinstance(prev, AIMessage) or not ai_message_has_tool_calls(prev):
                issues.append(
                    f"index {idx}: Tool 前一条不是带 tool_calls 的 AI",
                )
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                issues.append(f"index {idx}: AI content 仍为块数组")
            extra = getattr(msg, "additional_kwargs", None) or {}
            if extra.get("reasoning_content") is not None:
                issues.append(f"index {idx}: AI 仍含 reasoning_content 顶层字段")
    return issues
