"""
跨模型会话历史 canonical v2 格式。

落盘目标（对齐多厂商最小公约）：
- OpenAI Chat Completions：assistant.tool_calls 后紧跟 role=tool，id 对齐；允许 1 AI + 多 Tool。
- Anthropic Messages：tool_use 与 tool_result 之间不得插入其它消息；tool_result 在 user 消息中（由 LangChain 转换）。
- 存储层：仅 LangGraph 原生 Human/AI/Tool；manifest/anchor 为 Human 上下文消息，位于末条 user 前；不含 SystemMessage。

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
from llgraph.context.message_normalize import _message_text, migrate_legacy_pinned_message, reorder_pinned_session_messages
from llgraph.session.session_manifest import is_session_manifest_message

CANONICAL_FORMAT_VERSION = 2
_LLGRAPH_META_KEY = "llgraph"


@dataclass
class CanonicalV2Report:
    """canonical v2 转换报告。"""

    flattened_ai_messages: int = 0
    dropped_system_messages: int = 0
    removed_orphan_tools: int = 0
    normalized_ai_messages: int = 0
    patched_tool_results: int = 0

    @property
    def changed(self) -> bool:
        return (
            self.flattened_ai_messages > 0
            or self.dropped_system_messages > 0
            or self.removed_orphan_tools > 0
            or self.normalized_ai_messages > 0
            or self.patched_tool_results > 0
        )


_THINKING_BLOCK_KINDS = (
    "thinking",
    "reasoning",
    "reasoning_text",
    "redacted_thinking",
)


def _thinking_block_plaintext(block: dict[str, Any]) -> str:
    """从 thinking/reasoning 块提取可读明文（忽略 null）。"""
    for key in ("thinking", "reasoning", "text", "data"):
        raw = block.get(key)
        if isinstance(raw, str) and raw.strip():
            # redacted_thinking.data 多为密文，不当作展示明文
            if str(block.get("type", "")).lower() == "redacted_thinking" and key == "data":
                continue
            return raw.strip()
    return ""


def _extract_native_thinking_blocks(content: Any) -> list[dict[str, Any]]:
    """
    提取可回传的 native thinking 块（含 Opus/Bedrock 仅 signature 的空明文块）。

    @param content AIMessage.content
    @return 规范化块列表
    """
    if not isinstance(content, list):
        return []
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type", "")).lower()
        if kind not in _THINKING_BLOCK_KINDS:
            continue
        if kind == "redacted_thinking":
            kept: dict[str, Any] = {"type": "redacted_thinking"}
            if block.get("data") is not None:
                kept["data"] = block.get("data")
            if isinstance(block.get("signature"), str) and block["signature"]:
                kept["signature"] = block["signature"]
            out.append(kept)
            continue
        kept = {"type": "thinking"}
        text = _thinking_block_plaintext(block)
        # 明文可空：东买/Bedrock 上 Opus 常只回 signature
        kept["thinking"] = text
        sig = block.get("signature")
        if isinstance(sig, str) and sig.strip():
            kept["signature"] = sig.strip()
        # 无明文且无签名则无回传价值
        if not kept["thinking"] and "signature" not in kept:
            continue
        out.append(kept)
    return out


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
            if kind in _THINKING_BLOCK_KINDS:
                text = _thinking_block_plaintext(block)
                if text:
                    parts.append(text)
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

    同时保留 thinking_blocks（含 signature），供 Claude 多轮回传；
    东买/Bedrock 上 Opus 常无明文，仍须保住签名块。

    @param msg assistant 消息
    @return (消息, 是否改写)
    """
    thinking = _extract_thinking_for_meta(msg)
    content = getattr(msg, "content", "")
    native_blocks = _extract_native_thinking_blocks(content)
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
                    "redacted_thinking",
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
    if native_blocks:
        meta["thinking_blocks"] = native_blocks
        changed = True
        # 网关未回明文时仍标记曾开启 thinking，便于 UI/排查
        if not thinking and any(
            isinstance(b.get("signature"), str) and b.get("signature") for b in native_blocks
        ):
            meta["thinking_redacted"] = True
    if meta:
        extra[_LLGRAPH_META_KEY] = meta
    extra.pop("reasoning_content", None)
    if not changed and not meta and isinstance(content, str):
        return msg, False
    return msg.model_copy(update={"content": merged_text, "additional_kwargs": extra}), True


def to_canonical_v2_messages(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], CanonicalV2Report]:
    """
    转为 canonical v2：无 SystemMessage + 纯文本 AI + 合法 tool 链。

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

    manifest: BaseMessage | None = None
    anchor: BaseMessage | None = None
    conversation: list[BaseMessage] = []

    for msg in safe:
        if is_session_manifest_message(msg):
            manifest = migrate_legacy_pinned_message(msg)
            continue
        if is_conversation_anchor_message(msg):
            anchor = migrate_legacy_pinned_message(msg)
            continue
        if isinstance(msg, SystemMessage):
            report.dropped_system_messages += 1
            continue

        if isinstance(msg, AIMessage):
            flat, changed = _flatten_ai_for_storage(msg)
            if changed:
                report.flattened_ai_messages += 1
            conversation.append(flat)
            continue

        conversation.append(msg)

    ordered: list[BaseMessage] = list(conversation)
    pinned_tail: list[BaseMessage] = []
    if manifest is not None:
        pinned_tail.append(manifest)
    if anchor is not None:
        pinned_tail.append(anchor)
    if pinned_tail:
        ordered = reorder_pinned_session_messages([*ordered, *pinned_tail])
    return ordered if ordered else messages, report


def validate_canonical_v2_invariants(messages: list[BaseMessage]) -> list[str]:
    """
    校验 canonical v2 不变量。

    @param messages 消息列表
    @return 违规描述列表；空表示通过
    """
    from llgraph.context.chat_history_repair import ai_message_has_tool_calls

    issues: list[str] = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            issues.append(f"index {idx}: 落盘消息不得含 SystemMessage")
            continue
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
