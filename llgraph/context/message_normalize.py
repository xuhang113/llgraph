"""发往 LLM 前合并 system 消息（兼容 Anthropic 仅允许首段 system）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

_SESSION_MANIFEST_TAG = "<session-manifest>"
_CONVERSATION_ANCHOR_TAG = "<conversation-anchor>"


def _is_session_manifest_message(msg: BaseMessage) -> bool:
    if not isinstance(msg, SystemMessage):
        return False
    return _SESSION_MANIFEST_TAG in _message_text(getattr(msg, "content", ""))


def _is_conversation_anchor_message(msg: BaseMessage) -> bool:
    if not isinstance(msg, SystemMessage):
        return False
    return _CONVERSATION_ANCHOR_TAG in _message_text(getattr(msg, "content", ""))


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

    顺序：按模型 profile 清理/展开 tool 链 → manifest/anchor 置顶 → 合并 system。

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
    ordered = reorder_pinned_system_messages(cleaned)
    if workspace is not None:
        from llgraph.context.context_dispatch_window import trim_messages_for_dispatch_window
        from llgraph.context.context_settings import resolve_context_settings

        ctx_settings = resolve_context_settings(workspace)
        if ctx_settings.dispatch_keep_user_turns > 0:
            ordered = trim_messages_for_dispatch_window(
                ordered,
                keep_user_turns=ctx_settings.dispatch_keep_user_turns,
            )
    return normalize_messages_for_llm(
        ordered,
        agent_system_content=agent_system_content,
        workspace=workspace,
        model_id=model_id,
    )


def normalize_messages_for_llm(
    messages: list[BaseMessage],
    *,
    agent_system_content: str | None = None,
    workspace: Path | None = None,
    model_id: str | None = None,
) -> list[BaseMessage]:
    """
    将 agent system prompt 与会话内置顶 SystemMessage 合并为一条，其余保持顺序。

    启用 prompt_cache 时：稳定 Agent 规范带 cache_control，manifest/anchor 为可变块不打断点。
    仍为单条 SystemMessage（多 content block），满足 Anthropic 连续 system 要求。

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


def reorder_pinned_system_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    将会话 manifest / anchor 置顶到业务消息之前（落盘状态整理）。

    @param messages 原始消息
    @return 重排后的消息
    """
    manifest = [m for m in messages if _is_session_manifest_message(m)]
    anchor = [m for m in messages if _is_conversation_anchor_message(m)]
    rest = [
        m
        for m in messages
        if not _is_session_manifest_message(m)
        and not _is_conversation_anchor_message(m)
    ]
    ordered: list[BaseMessage] = []
    if manifest:
        ordered.append(manifest[-1])
    if anchor:
        ordered.append(anchor[-1])
    ordered.extend(rest)
    return ordered if ordered else messages


def make_prompt_normalizer(
    agent_system_content: str,
    workspace: Path | None = None,
):
    """
    供 create_react_agent(prompt=...) 使用的可调用对象。

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
