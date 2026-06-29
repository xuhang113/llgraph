"""修复/清理 tool_calls 与 ToolMessage 链，满足 LangGraph 与网关 API 校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from llgraph.context.message_dispatch_profile import (
    MessageDispatchProfile,
    canonical_persist_profile,
)
from llgraph.context.message_normalize import _message_text
from llgraph.context.tool_call_id import (
    canonical_tool_call_id,
    gateway_safe_tool_call_id,
    normalize_tool_call_id_raw,
)

_REPAIR_TOOL_RESULT = "（上一轮工具调用未完成，已跳过；请继续当前问题。）"
_REASONING_PLACEHOLDER = "（历史思考过程未落盘，占位以满足 Kimi 等网关校验。）"
_EMPTY_ASSISTANT_PLACEHOLDER = " "


@dataclass
class ChatHistorySanitizeReport:
    """消息链清理报告。"""

    patched_tool_results: int = 0
    removed_orphan_tools: int = 0
    normalized_ai_messages: int = 0
    expanded_tool_rounds: int = 0
    patched_reasoning_content: int = 0
    stripped_thinking_blocks: int = 0
    sanitized_tool_call_ids: int = 0
    patched_empty_assistant: int = 0

    @property
    def changed(self) -> bool:
        return (
            self.patched_tool_results > 0
            or self.removed_orphan_tools > 0
            or self.normalized_ai_messages > 0
            or self.expanded_tool_rounds > 0
            or self.patched_reasoning_content > 0
            or self.stripped_thinking_blocks > 0
            or self.sanitized_tool_call_ids > 0
            or self.patched_empty_assistant > 0
        )


def _tool_call_id(call: object) -> str | None:
    """
    从 tool_call 条目取 id。

    @param call 工具调用 dict 或对象
    @return tool_call_id
    """
    if isinstance(call, dict):
        raw = call.get("id")
    else:
        raw = getattr(call, "id", None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _tool_call_name(call: object) -> str:
    """
    从 tool_call 条目取工具名。

    @param call 工具调用 dict 或对象
    @return 工具名
    """
    if isinstance(call, dict):
        raw = call.get("name")
        if raw is None and isinstance(call.get("function"), dict):
            raw = call["function"].get("name")
    else:
        raw = getattr(call, "name", None)
    if raw is None:
        return "tool"
    return str(raw).strip() or "tool"


def ai_message_tool_calls(msg: AIMessage) -> list[dict[str, Any]]:
    """
    读取 AIMessage 上的 tool_calls（含 additional_kwargs 兜底）。

    @param msg assistant 消息
    @return tool_calls 列表
    """
    calls = list(msg.tool_calls or [])
    if calls:
        return calls
    extra = getattr(msg, "additional_kwargs", None) or {}
    raw = extra.get("tool_calls")
    if isinstance(raw, list):
        return raw
    return []


def ai_message_has_tool_calls(msg: AIMessage) -> bool:
    """
    是否包含工具调用。

    @param msg assistant 消息
    @return 是否有 tool_calls
    """
    return bool(ai_message_tool_calls(msg))


def _normalize_tool_call_id(tool_call_id: object | None) -> str | None:
    return normalize_tool_call_id_raw(tool_call_id)


def _canonical_tool_call_id(tool_call_id: str) -> str:
    return canonical_tool_call_id(tool_call_id)


def _gateway_safe_tool_call_id(tool_call_id: str) -> str:
    return gateway_safe_tool_call_id(tool_call_id)


def _lookup_tool_message(
    tools_by_id: dict[str, ToolMessage],
    call_id: str,
) -> ToolMessage | None:
    """
    按原始或规范化 id 查找 ToolMessage。

    @param tools_by_id id -> ToolMessage（键可为原始或 canonical）
    @param call_id AI tool_call id
    @return 匹配的工具消息
    """
    direct = tools_by_id.get(call_id)
    if direct is not None:
        return direct
    return tools_by_id.get(_canonical_tool_call_id(call_id))


def sanitize_gateway_tool_call_ids(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """
    出站前规范化 tool_call_id（AI tool_calls 与 ToolMessage 对齐）。

    Kimi 等模型落盘 id 可能含 `grep_files:65`，Claude 网关会 400；仅 dispatch 改写。

    @param messages 消息列表
    @return (新列表, 改写条数)
    """
    changed = 0
    out: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            calls = ai_message_tool_calls(msg)
            if not calls:
                out.append(msg)
                continue
            new_calls: list[Any] = []
            remapped = False
            for tc in calls:
                if not isinstance(tc, dict):
                    new_calls.append(tc)
                    continue
                cid = _tool_call_id(tc)
                if not cid:
                    new_calls.append(tc)
                    continue
                safe = _gateway_safe_tool_call_id(cid)
                if safe == cid:
                    new_calls.append(tc)
                    continue
                new_tc = dict(tc)
                new_tc["id"] = safe
                new_calls.append(new_tc)
                remapped = True
            if remapped:
                changed += 1
                out.append(msg.model_copy(update={"tool_calls": new_calls}))
            else:
                out.append(msg)
            continue

        if isinstance(msg, ToolMessage):
            tid = getattr(msg, "tool_call_id", None)
            if tid is not None:
                text = str(tid)
                safe = _gateway_safe_tool_call_id(text)
                if safe != text:
                    changed += 1
                    out.append(msg.model_copy(update={"tool_call_id": safe}))
                    continue
            out.append(msg)
            continue

        out.append(msg)

    return out, changed


def normalize_ai_tool_calls_on_message(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    将 tool_calls 写入 AIMessage 主字段（jsonl 往返后可能仅在 kwargs 中）。

    @param msg assistant 消息
    @return (规范化消息, 是否改写)
    """
    calls = ai_message_tool_calls(msg)
    if not calls:
        return msg, False
    current = list(msg.tool_calls or [])
    if current == calls:
        return msg, False
    return msg.model_copy(update={"tool_calls": calls}), True


def _extract_reasoning_text(msg: AIMessage) -> str:
    """
    从 AIMessage 正文或 additional_kwargs 提取思考内容。

    @param msg assistant 消息
    @return 思考文本；无则空串
    """
    extra = getattr(msg, "additional_kwargs", None) or {}
    raw = extra.get("reasoning_content")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    content = getattr(msg, "content", "")
    if isinstance(content, str) and content.strip():
        return ""
    if isinstance(content, list):
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
            elif isinstance(block, str) and block.strip():
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


def _ai_has_persisted_thinking(msg: AIMessage) -> bool:
    """
    是否含可回传的 thinking（llgraph.thinking_text / content 块 / reasoning_content）。

    @param msg assistant 消息
    @return 是否有非占位 thinking
    """
    from llgraph.core.gateway_kimi_patch import resolve_kimi_reasoning_content

    reasoning = resolve_kimi_reasoning_content(msg)
    return bool(reasoning.strip()) and reasoning != _REASONING_PLACEHOLDER


def ensure_tool_ai_reasoning_content(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    为带 tool_calls 的 AI 补 reasoning_content（Kimi k2 等 thinking 模式必填）。

    @param msg assistant 消息
    @return (消息, 是否改写)
    """
    if not ai_message_has_tool_calls(msg):
        return msg, False

    extra = dict(getattr(msg, "additional_kwargs", None) or {})
    existing = extra.get("reasoning_content")
    if isinstance(existing, str) and existing.strip():
        return msg, False

    from llgraph.core.gateway_kimi_patch import resolve_kimi_reasoning_content

    reasoning = resolve_kimi_reasoning_content(msg)
    extra["reasoning_content"] = reasoning
    return msg.model_copy(update={"additional_kwargs": extra}), True


def ensure_thinking_only_ai_reasoning_content(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    为 thinking-only（无 tool、无 visible text）的 AI 补 reasoning_content。

    Think step 续跑时 Kimi 需将 reasoning 写回历史。

    @param msg assistant 消息
    @return (消息, 是否改写)
    """
    if ai_message_has_tool_calls(msg):
        return msg, False
    if not _ai_has_persisted_thinking(msg):
        return msg, False

    extra = dict(getattr(msg, "additional_kwargs", None) or {})
    existing = extra.get("reasoning_content")
    if isinstance(existing, str) and existing.strip():
        return msg, False

    from llgraph.core.gateway_kimi_patch import resolve_kimi_reasoning_content

    reasoning = resolve_kimi_reasoning_content(msg)
    extra["reasoning_content"] = reasoning
    return msg.model_copy(update={"additional_kwargs": extra}), True


def flatten_assistant_thinking_blocks(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    将 assistant 多段 content 规范为纯文本（剥离 thinking / tool_use 等块）。

    Kimi 等落盘的 thinking、Anthropic 原生 tool_use 块在 Claude/GPT 回传时会 400。

    @param msg assistant 消息
    @return (消息, 是否改写)
    """
    content = getattr(msg, "content", "")
    if not isinstance(content, list):
        return msg, False
    texts: list[str] = []
    stripped_block = False
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
                stripped_block = True
                continue
            if kind == "text":
                text = block.get("text")
                if text:
                    texts.append(str(text))
                continue
            stripped_block = True
        elif isinstance(block, str) and block.strip():
            texts.append(block)
    if not stripped_block:
        return msg, False
    merged = "\n\n".join(texts).strip()
    return msg.model_copy(update={"content": merged}), True


def strip_assistant_thinking_blocks(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """
    批量剥离 assistant thinking 内容块。

    @param messages 消息列表
    @return (新列表, 改写条数)
    """
    changed = 0
    out: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            out.append(msg)
            continue
        fixed, did = flatten_assistant_thinking_blocks(msg)
        if did:
            changed += 1
        out.append(fixed)
    return out, changed


def strip_tool_ai_reasoning_contents(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """
    移除带 tool_calls 的 AI 上的 reasoning_content（非 Kimi 出站时避免网关拒收）。

    @param messages 消息列表
    @return (新列表, 剥离条数)
    """
    stripped = 0
    out: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage) or not ai_message_has_tool_calls(msg):
            out.append(msg)
            continue
        extra = dict(getattr(msg, "additional_kwargs", None) or {})
        if "reasoning_content" not in extra:
            out.append(msg)
            continue
        extra.pop("reasoning_content", None)
        stripped += 1
        update: dict[str, Any] = {"additional_kwargs": extra}
        out.append(msg.model_copy(update=update))
    return out, stripped


def patch_tool_ai_reasoning_contents(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """
    批量补全 tool 轮与 thinking-only AI 的 reasoning_content。

    @param messages 消息列表
    @return (新列表, 补全条数)
    """
    patched = 0
    out: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            fixed, changed = ensure_tool_ai_reasoning_content(msg)
            if not changed:
                fixed, changed = ensure_thinking_only_ai_reasoning_content(fixed)
            if changed:
                patched += 1
            out.append(fixed)
            continue
        out.append(msg)
    return out, patched


def _strict_gateway_tool_violation_count(messages: list[BaseMessage]) -> int:
    """
    统计「tool 前一条不是带 tool_calls 的 AI」条数（火山等网关严格模式）。

    @param messages 消息列表
    @return 违规条数
    """
    count = 0
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        if idx == 0:
            count += 1
            continue
        prev = messages[idx - 1]
        if not isinstance(prev, AIMessage) or not ai_message_has_tool_calls(prev):
            count += 1
    return count


def _maybe_patch_reasoning(
    msg: AIMessage,
    profile: MessageDispatchProfile,
) -> AIMessage:
    if not profile.patch_tool_ai_reasoning:
        return msg
    fixed, _ = ensure_tool_ai_reasoning_content(msg)
    return fixed


def _append_tool_results(
    safe: list[BaseMessage],
    tool_calls: list[dict[str, Any]],
    tools_by_id: dict[str, ToolMessage],
    report: ChatHistorySanitizeReport,
) -> None:
    for tc in tool_calls:
        cid = _tool_call_id(tc)
        if not cid:
            continue
        tool_msg = _lookup_tool_message(tools_by_id, cid)
        if tool_msg is not None:
            safe.append(tool_msg)
            continue
        safe.append(
            ToolMessage(
                content=_REPAIR_TOOL_RESULT,
                tool_call_id=cid,
                name=_tool_call_name(tc),
            ),
        )
        report.patched_tool_results += 1


def _append_single_tool_round(
    safe: list[BaseMessage],
    norm: AIMessage,
    tool_calls: list[dict[str, Any]],
    tools_by_id: dict[str, ToolMessage],
    report: ChatHistorySanitizeReport,
    profile: MessageDispatchProfile,
) -> None:
    """
    写入单轮工具链；严格网关下多 tool 拆成「AI(单 call)+Tool」交替。

    @param safe 输出列表
    @param norm 已规范化的 AI 消息
    @param tool_calls 工具调用列表
    @param tools_by_id tool_call_id -> ToolMessage
    @param report 清理报告
    @param profile 出站/落盘策略
    """
    if not tool_calls:
        safe.append(_maybe_patch_reasoning(norm, profile))
        return

    if len(tool_calls) <= 1 or not profile.expand_parallel_tool_rounds:
        safe.append(_maybe_patch_reasoning(norm, profile))
        _append_tool_results(safe, tool_calls, tools_by_id, report)
        return

    # 严格网关：禁止 Tool 紧跟 Tool，按每个 call 拆成独立的 AI+Tool
    report.expanded_tool_rounds += 1
    base_content = norm.content if isinstance(norm.content, str) else ""
    base_extra = dict(getattr(norm, "additional_kwargs", None) or {})
    if profile.patch_tool_ai_reasoning and base_extra.get("reasoning_content") is None:
        from llgraph.core.gateway_kimi_patch import resolve_kimi_reasoning_content

        base_extra["reasoning_content"] = resolve_kimi_reasoning_content(norm)
    for idx, tc in enumerate(tool_calls):
        cid = _tool_call_id(tc)
        if not cid:
            continue
        ai_piece = norm.model_copy(
            update={
                "tool_calls": [tc],
                "content": base_content if idx == 0 else _EMPTY_ASSISTANT_PLACEHOLDER,
                "additional_kwargs": dict(base_extra),
            },
        )
        safe.append(ai_piece)
        tool_msg = _lookup_tool_message(tools_by_id, cid)
        if tool_msg is not None:
            safe.append(tool_msg)
        else:
            safe.append(
                ToolMessage(
                    content=_REPAIR_TOOL_RESULT,
                    tool_call_id=cid,
                    name=_tool_call_name(tc),
                ),
            )
            report.patched_tool_results += 1


def _assistant_dispatch_text(msg: AIMessage) -> str:
    """@param msg assistant 消息 @return 出站 content 文本（含 thinking 块）"""
    return _message_text(getattr(msg, "content", "")).strip()


def rehydrate_native_thinking_block(msg: AIMessage) -> tuple[AIMessage, bool]:
    """
    将 llgraph.thinking_text 还原为 content 内 thinking 块（Claude native 协议）。

    @param msg assistant 消息
    @return (消息, 是否改写)
    """
    extra = dict(getattr(msg, "additional_kwargs", None) or {})
    meta = extra.get("llgraph")
    if not isinstance(meta, dict):
        return msg, False
    thinking = meta.get("thinking_text")
    if not isinstance(thinking, str) or not thinking.strip():
        return msg, False

    content = getattr(msg, "content", "")
    text_parts: list[str] = []
    if isinstance(content, list):
        has_thinking = False
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type", "")).lower()
            if kind in ("thinking", "reasoning", "reasoning_text", "redacted_thinking"):
                has_thinking = True
                continue
            if kind == "text":
                text = block.get("text")
                if text:
                    text_parts.append(str(text))
        if has_thinking:
            return msg, False
    elif isinstance(content, str):
        stripped = content.strip()
        if stripped and stripped != _EMPTY_ASSISTANT_PLACEHOLDER.strip():
            text_parts.append(stripped)
    else:
        return msg, False

    blocks: list[dict[str, str]] = [
        {"type": "thinking", "thinking": thinking.strip()},
    ]
    merged_text = "\n\n".join(text_parts).strip()
    if merged_text:
        blocks.append({"type": "text", "text": merged_text})
    return msg.model_copy(update={"content": blocks}), True


def rehydrate_all_native_thinking_blocks(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """
    批量还原 native thinking 块。

    @param messages 消息列表
    @return (新列表, 改写条数)
    """
    changed = 0
    out: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            out.append(msg)
            continue
        fixed, did = rehydrate_native_thinking_block(msg)
        if did:
            changed += 1
        out.append(fixed)
    return out, changed


def persist_all_ai_thinking_to_meta(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """
    批量将 AI content 内 thinking 块写入 llgraph.thinking_text（canonical 落盘）。

    所有模型均执行，避免 ensure_nonempty 覆盖 content 时丢失 thinking；
    **是否回传到网关** 由 dispatch profile 决定（见 dispatch_preserves_thinking_on_outbound）。

    @param messages 消息列表
    @return (新列表, 改写条数)
    """
    from llgraph.context.message_canonical import persist_ai_thinking_in_message

    changed = 0
    out: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            out.append(msg)
            continue
        fixed, did = persist_ai_thinking_in_message(msg)
        if did:
            changed += 1
        out.append(fixed)
    return out, changed


def _assistant_has_thinking_blocks(msg: AIMessage) -> bool:
    """content 列表内是否含 thinking/reasoning 块。"""
    content = getattr(msg, "content", "")
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).lower() in (
            "thinking",
            "reasoning",
            "reasoning_text",
            "redacted_thinking",
        ):
            return True
    return False


def ensure_nonempty_assistant_messages(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """
    补齐 content 为空的 assistant（火山等网关 400：assistant must not be empty）。

    已 rehydrate 的 thinking 块不覆盖。

    @param messages 消息列表
    @return (新列表, 补齐条数)
    """
    patched = 0
    out: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            out.append(msg)
            continue
        text = _assistant_dispatch_text(msg)
        if text:
            out.append(msg)
            continue
        if _assistant_has_thinking_blocks(msg):
            out.append(msg)
            continue
        if ai_message_has_tool_calls(msg):
            patched += 1
            out.append(msg.model_copy(update={"content": _EMPTY_ASSISTANT_PLACEHOLDER}))
            continue
        patched += 1
        out.append(msg.model_copy(update={"content": _EMPTY_ASSISTANT_PLACEHOLDER}))
    return out, patched


def rebuild_provider_safe_messages(
    messages: list[BaseMessage],
    profile: MessageDispatchProfile | None = None,
) -> tuple[list[BaseMessage], ChatHistorySanitizeReport]:
    """
    按网关规则重建消息链：每条 tool 前一条必须是仅含对应 call 的 AI。

    说明：OpenAI 允许「1 个 AI + 连续多个 Tool」；火山等要求每条 Tool 紧跟一个 AI。
    Anthropic/Claude 要求并行 tool 的全部 tool_result 同处一条 user 消息，故 expand 保持 false。

    @param messages 原始消息
    @param profile 修链策略；None 时为落盘 canonical（不展开、不补 reasoning）
    @return (安全消息列表, 报告)
    """
    effective = profile if profile is not None else canonical_persist_profile()
    report = ChatHistorySanitizeReport()
    if not messages:
        return messages, report

    safe: list[BaseMessage] = []
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]
        if isinstance(msg, AIMessage):
            norm, changed = normalize_ai_tool_calls_on_message(msg)
            if changed:
                report.normalized_ai_messages += 1
            tool_calls = ai_message_tool_calls(norm)
            i += 1
            if not tool_calls:
                safe.append(norm)
                continue

            expected_canonical = {
                _canonical_tool_call_id(x)
                for x in (_tool_call_id(tc) for tc in tool_calls)
                if x
            }
            tools_by_id: dict[str, ToolMessage] = {}
            while i < n and isinstance(messages[i], ToolMessage):
                tool_msg = messages[i]
                tid = _normalize_tool_call_id(getattr(tool_msg, "tool_call_id", None))
                if tid and _canonical_tool_call_id(tid) in expected_canonical:
                    tools_by_id[tid] = tool_msg
                    tools_by_id[_canonical_tool_call_id(tid)] = tool_msg
                else:
                    report.removed_orphan_tools += 1
                i += 1

            _append_single_tool_round(
                safe,
                norm,
                tool_calls,
                tools_by_id,
                report,
                effective,
            )
            continue

        if isinstance(msg, ToolMessage):
            report.removed_orphan_tools += 1
            i += 1
            continue

        safe.append(msg)
        i += 1

    safe, _persisted_thinking = persist_all_ai_thinking_to_meta(safe)
    from llgraph.core.model_thinking_profile import dispatch_rehydrates_native_thinking_blocks

    if dispatch_rehydrates_native_thinking_blocks(effective):
        safe, _rehydrated = rehydrate_all_native_thinking_blocks(safe)
    if effective.patch_tool_ai_reasoning:
        # Kimi k2 等：补 reasoning_content，thinking-only 续跑可回灌网关
        safe, reasoning_patched = patch_tool_ai_reasoning_contents(safe)
        report.patched_reasoning_content = reasoning_patched
    else:
        safe, _stripped = strip_tool_ai_reasoning_contents(safe)
    if effective.strip_assistant_thinking_blocks:
        # Claude/GPT 等：出站剥离 content 内 thinking 块（meta 仍保留 thinking_text）
        safe, stripped = strip_assistant_thinking_blocks(safe)
        report.stripped_thinking_blocks = stripped
    # 出站前始终规范化 tool_call_id（Kimi 落盘 id 常含 ':'，Claude/Bedrock 会 400）
    safe, sanitized = sanitize_gateway_tool_call_ids(safe)
    report.sanitized_tool_call_ids = sanitized
    safe, nonempty = ensure_nonempty_assistant_messages(safe)
    report.patched_empty_assistant = nonempty
    return safe, report


def repair_incomplete_tool_rounds(
    messages: list[BaseMessage],
    profile: MessageDispatchProfile | None = None,
) -> tuple[list[BaseMessage], int]:
    """
    为缺少结果的 tool_calls 补占位 ToolMessage；丢弃不属于本轮 AI 的 ToolMessage。

    @param messages 原始消息列表
    @return (修复后列表, 补全的 ToolMessage 条数)
    """
    safe, report = rebuild_provider_safe_messages(messages, profile)
    return safe, report.patched_tool_results


def remove_orphan_tool_messages(
    messages: list[BaseMessage],
    profile: MessageDispatchProfile | None = None,
) -> tuple[list[BaseMessage], int]:
    """
    移除前序无匹配 tool_calls 的 ToolMessage（网关 400 常见原因）。

    @param messages 消息列表
    @return (新列表, 移除条数)
    """
    safe, report = rebuild_provider_safe_messages(messages, profile)
    return safe, report.removed_orphan_tools


def sanitize_chat_history(
    messages: list[BaseMessage],
    profile: MessageDispatchProfile | None = None,
) -> tuple[list[BaseMessage], ChatHistorySanitizeReport]:
    """
    完整清理：规范化 AI tool_calls + 重建合法 tool 链。

    @param messages 原始消息
    @param profile 修链策略；None 为 canonical 落盘策略
    @return (清理后消息, 报告)
    """
    return rebuild_provider_safe_messages(messages, profile)


def sanitize_chat_history_for_dispatch(
    messages: list[BaseMessage],
    workspace: Any,
    model_id: str | None = None,
) -> tuple[list[BaseMessage], ChatHistorySanitizeReport]:
    """
    按当前模型出站策略清理（/model 切换后下条消息生效）。

    @param messages 原始消息
    @param workspace 工作区根
    @param model_id 模型 id；None 时用当前 effective model
    @return (清理后消息, 报告)
    """
    from pathlib import Path

    from llgraph.context.message_dispatch_profile import resolve_dispatch_profile

    ws = Path(workspace).expanduser().resolve() if workspace is not None else None
    dispatch = resolve_dispatch_profile(ws, model_id)
    return rebuild_provider_safe_messages(messages, dispatch)


def _messages_changed(before: list[BaseMessage], after: list[BaseMessage]) -> bool:
    from langchain_core.messages import messages_to_dict

    return messages_to_dict(before) != messages_to_dict(after)


def ensure_agent_chat_history_sanitized(
    agent: Any,
    workspace: Any,
    thread_id: str,
) -> ChatHistorySanitizeReport:
    """
    读取 agent 状态、转为 canonical v2 并写回。

    @param agent LangGraph agent
    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 清理报告
    """
    from pathlib import Path

    from llgraph.context.message_canonical import to_canonical_v2_messages

    empty = ChatHistorySanitizeReport()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception:
        return empty
    messages = list((state.values or {}).get("messages") or [])
    if not messages:
        return empty

    new_messages, canon_report = to_canonical_v2_messages(messages)
    report = ChatHistorySanitizeReport(
        patched_tool_results=canon_report.patched_tool_results,
        removed_orphan_tools=canon_report.removed_orphan_tools,
        normalized_ai_messages=canon_report.normalized_ai_messages,
    )
    if not _messages_changed(messages, new_messages):
        return report

    try:
        agent.update_state(config, {"messages": new_messages})
        from llgraph.session.session_file_store import save_session_messages

        save_session_messages(Path(workspace), thread_id, new_messages)
    except Exception:
        return empty

    parts: list[str] = ["canonical v2"]
    if report.removed_orphan_tools > 0:
        parts.append(f"移除 {report.removed_orphan_tools} 条无效 tool 结果")
    if report.patched_tool_results > 0:
        parts.append(f"补齐 {report.patched_tool_results} 条中断占位")
    if report.normalized_ai_messages > 0:
        parts.append(f"规范化 {report.normalized_ai_messages} 条 AI tool_calls")
    if report.expanded_tool_rounds > 0:
        parts.append(f"展开 {report.expanded_tool_rounds} 轮并行工具为 AI+Tool 交替")
    if report.patched_reasoning_content > 0:
        parts.append(f"补齐 {report.patched_reasoning_content} 条 AI reasoning_content")
    if report.stripped_thinking_blocks > 0:
        parts.append(f"剥离 {report.stripped_thinking_blocks} 条 AI thinking 块")
    if canon_report.flattened_ai_messages > 0:
        parts.append(f"扁平化 {canon_report.flattened_ai_messages} 条 AI 内容")
    if canon_report.archived_system_messages > 0:
        parts.append(f"归档 {canon_report.archived_system_messages} 条中段 system")
    if parts:
        from llgraph.terminal.ops_notice import ops_notice

        ops_notice(f"历史消息已修复: {'；'.join(parts)}。")
    return report


def ensure_agent_chat_history_dispatch_safe(
    agent: Any,
    workspace: Any,
    thread_id: str,
) -> ChatHistorySanitizeReport:
    """
    按当前模型出站规则清理 checkpoint 消息并写回（修复空 assistant 等 400）。

    @param agent LangGraph agent
    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 清理报告
    """
    from pathlib import Path

    from llgraph.core.llm_settings import resolve_effective_model

    empty = ChatHistorySanitizeReport()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception:
        return empty
    messages = list((state.values or {}).get("messages") or [])
    if not messages:
        return empty

    ws = Path(workspace).expanduser().resolve() if workspace is not None else None
    model_id = resolve_effective_model(ws)
    safe, report = sanitize_chat_history_for_dispatch(messages, ws, model_id)
    if not _messages_changed(messages, safe):
        return report

    try:
        agent.update_state(config, {"messages": safe})
        if ws is not None:
            from llgraph.session.session_file_store import save_session_messages

            save_session_messages(ws, thread_id, safe)
    except Exception:
        return empty
    return report


def ensure_agent_chat_history_repaired(
    agent: Any,
    workspace: Any,
    thread_id: str,
) -> int:
    """
    兼容旧调用：返回补全的 ToolMessage 条数。

    @param agent LangGraph agent
    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 补全条数
    """
    report = ensure_agent_chat_history_sanitized(agent, workspace, thread_id)
    return report.patched_tool_results
