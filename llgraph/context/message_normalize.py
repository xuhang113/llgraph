"""发往 LLM 前合并 system 消息（兼容 Anthropic 仅允许首段 system）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

_SESSION_MANIFEST_TAG = "<session-manifest>"
_CONVERSATION_ANCHOR_TAG = "<conversation-anchor>"


def _message_text(content: Any) -> str:
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


def _is_session_manifest_message(msg: BaseMessage) -> bool:
    from llgraph.session.session_manifest import is_session_manifest_message

    return is_session_manifest_message(msg)


def _is_conversation_anchor_message(msg: BaseMessage) -> bool:
    from llgraph.context.conversation_anchor import is_conversation_anchor_message

    return is_conversation_anchor_message(msg)


def _is_conversation_summary_message(msg: BaseMessage) -> bool:
    from llgraph.context.conversation_anchor import is_conversation_summary_message

    return is_conversation_summary_message(msg)


def _is_business_user_message(msg: BaseMessage) -> bool:
    from llgraph.context.conversation_anchor import is_pinned_session_context_message

    return isinstance(msg, HumanMessage) and not is_pinned_session_context_message(msg)


def migrate_legacy_pinned_message(msg: BaseMessage) -> BaseMessage:
    """
    将旧版 SystemMessage 形式的 manifest/anchor 转为 HumanMessage。

    @param msg 原始消息
    @return 迁移后消息
    """
    if isinstance(msg, HumanMessage):
        return msg
    if isinstance(msg, SystemMessage) and (
        _is_session_manifest_message(msg) or _is_conversation_anchor_message(msg)
    ):
        return HumanMessage(content=_message_text(getattr(msg, "content", "")))
    return msg


def format_agent_chat_display_text(text: str) -> str:
    """
    Web 聊天区助手正文：剥离 tool markup 与【规划】行。保留空行，避免 GFM 表格吞掉后续段落。

    @param text 原始助手正文
    @return 用户可见文本
    """
    from llgraph.adapters.inbound.xml_tool_call import strip_inbound_tool_call_markup
    from llgraph.survey.survey_prompt import strip_survey_for_display

    cleaned = strip_inbound_tool_call_markup(text or "")
    cleaned = strip_survey_for_display(cleaned)
    lines = [
        ln
        for ln in cleaned.splitlines()
        if not ln.strip().startswith("【规划】")
    ]
    collapsed = "\n".join(lines)
    while "\n\n\n" in collapsed:
        collapsed = collapsed.replace("\n\n\n", "\n\n")
    return collapsed.strip()


def _state_messages(state: Any) -> list[BaseMessage]:
    if isinstance(state, dict):
        raw = state.get("messages") or []
    else:
        raw = getattr(state, "messages", None) or []
    return list(raw)


def prepare_messages_for_llm_dispatch(
    messages: list[BaseMessage],
    *,
    agent_system_content: str | None = None,
    workspace: Path | None = None,
    model_id: str | None = None,
) -> list[BaseMessage]:
    """
    发往模型前的完整规范化（对齐 Cursor：落盘 canonical，调用前按模型 adapter 修链）。

    出站顺序：按模型 profile 清理 tool 链 → 出站窗口裁剪 → manifest/anchor 注入历史尾段
    → 仅稳定 Agent 规范合并为 system → 本轮 user 在最后。

    @param messages 图状态中的消息
    @param agent_system_content build_system_prompt 正文
    @param workspace 工作区根（解析 /model 与 agent.json dispatch）
    @param model_id 模型 id；None 时用当前 effective model
    @return 可安全提交网关的消息列表
    """
    from llgraph.context.chat_history_repair import sanitize_chat_history_for_dispatch
    from llgraph.context.message_canonical import to_canonical_v2_messages

    canonical, _canon = to_canonical_v2_messages(messages)
    cleaned, _report = sanitize_chat_history_for_dispatch(
        canonical,
        workspace,
        model_id,
    )
    from llgraph.core.user_message_content import prepare_messages_for_multimodal_dispatch

    cleaned = prepare_messages_for_multimodal_dispatch(cleaned)
    if workspace is not None:
        from llgraph.context.runtime_context import get_active_thread_id

        thread_id = get_active_thread_id()
        if thread_id:
            from llgraph.context.conversation_anchor import (
                ensure_messages_include_conversation_anchor,
            )

            cleaned = ensure_messages_include_conversation_anchor(
                workspace,
                thread_id,
                cleaned,
            )
    if workspace is not None:
        from llgraph.context.context_settings import resolve_context_settings
        from llgraph.context.incremental_context import (
            dedupe_read_tool_messages_for_dispatch,
            prune_tool_messages_for_dispatch,
        )

        ctx_settings = resolve_context_settings(workspace)
        cleaned = prune_tool_messages_for_dispatch(cleaned, workspace, ctx_settings)
        cleaned = dedupe_read_tool_messages_for_dispatch(cleaned, ctx_settings)
    ordered = reorder_pinned_session_messages(cleaned)
    normalized = normalize_messages_for_llm(
        ordered,
        agent_system_content=agent_system_content,
        workspace=workspace,
        model_id=model_id,
    )
    from llgraph.context.user_correction_nudge import append_user_correction_nudge_for_dispatch

    with_correction = append_user_correction_nudge_for_dispatch(normalized)
    from llgraph.context.react_step_reminder import append_react_step_reminder_for_dispatch
    from llgraph.context.investigate_harness import append_soft_close_for_dispatch

    with_reminder = append_react_step_reminder_for_dispatch(
        with_correction,
        workspace=workspace,
    )
    with_soft = append_soft_close_for_dispatch(with_reminder, workspace=workspace)

    if workspace is not None:
        from llgraph.context.outbound_redact import (
            redact_messages_for_dispatch,
            resolve_outbound_redact_settings,
        )

        redact_settings = resolve_outbound_redact_settings(workspace)
        return redact_messages_for_dispatch(with_soft, redact_settings)
    return with_soft


def normalize_messages_for_llm(
    messages: list[BaseMessage],
    *,
    agent_system_content: str | None = None,
    workspace: Path | None = None,
    model_id: str | None = None,
) -> list[BaseMessage]:
    """
    将稳定 Agent 系统规范合并为单条 SystemMessage；manifest/anchor 保留在 messages 尾段。

    启用 prompt_cache 时：仅 Agent 规范带 cache_control（最长稳定前缀）。

    @param messages 当前状态消息
    @param agent_system_content build_system_prompt 正文
    @param workspace 工作区根（prompt_cache 配置）
    @param model_id 当前模型 id
    @return 规范化后的消息列表
    """
    from llgraph.core.prompt_cache import prepare_system_message_for_dispatch

    merged_system, non_system = prepare_system_message_for_dispatch(
        messages,
        agent_system_content=agent_system_content,
        workspace=workspace,
        model_id=model_id,
    )
    if merged_system is None:
        return messages
    return [merged_system, *non_system]


def reorder_pinned_session_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    将会话 manifest / anchor 放到业务历史尾段（末条真实 user 之前），并迁移为 HumanMessage。

    落盘/出站不含 SystemMessage；动态会话态不进 system，利于 Prompt Cache 前缀稳定。

    @param messages 原始消息
    @return 重排后的消息
    """
    manifest: list[BaseMessage] = []
    anchor: list[BaseMessage] = []
    rest: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if _is_session_manifest_message(msg):
            manifest.append(migrate_legacy_pinned_message(msg))
            continue
        if _is_conversation_anchor_message(msg):
            anchor.append(migrate_legacy_pinned_message(msg))
            continue
        if _is_conversation_summary_message(msg):
            continue
        rest.append(msg)

    pinned: list[BaseMessage] = []
    if manifest:
        pinned.append(manifest[-1])
    if anchor:
        pinned.append(anchor[-1])

    if not pinned:
        return messages

    last_user_idx: int | None = None
    for i in range(len(rest) - 1, -1, -1):
        if _is_business_user_message(rest[i]):
            last_user_idx = i
            break

    if not pinned:
        body = rest
    elif last_user_idx is None:
        body = [*rest, *pinned]
    else:
        body = [*rest[:last_user_idx], *pinned, *rest[last_user_idx:]]

    return body


def reorder_pinned_system_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """兼容旧名。"""
    return reorder_pinned_session_messages(messages)


def make_prompt_normalizer(
    agent_system_content: str,
    workspace: Path | None = None,
):
    """
    供 build_react_graph(prompt=...) 使用的可调用对象。

    @param agent_system_content 系统提示词
    @param workspace 工作区根（按当前 /model 解析出站 profile）
    @return 接收 graph state、返回 messages 的函数
    """
    ws = workspace.expanduser().resolve() if workspace is not None else None

    def _prepare(state: Any) -> list[BaseMessage]:
        from llgraph.core.llm_settings import resolve_effective_model

        model_id = resolve_effective_model(ws)
        return prepare_messages_for_llm_dispatch(
            _state_messages(state),
            agent_system_content=agent_system_content,
            workspace=ws,
            model_id=model_id,
        )

    return _prepare
