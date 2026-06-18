"""对话上下文压缩（P3）：Tier1 切分 + Tier2 锚点 + Tier3 检索。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llgraph.context.context_message_split import split_messages_for_compress_strategy
from llgraph.context.context_settings import is_auto_compress_strategy, resolve_context_settings
from llgraph.context.context_spill import compact_tool_messages_for_compress
from llgraph.context.conversation_anchor import (
    build_conversation_anchor_system_message,
    is_conversation_anchor_message,
    is_conversation_summary_message,
    load_session_from_manifest,
    run_anchor_update,
)
from llgraph.session.session_manifest import (
    build_session_manifest_system_message,
    is_session_manifest_message,
)


@dataclass
class CompressReport:
    """压缩结果报告。"""

    before_count: int
    after_count: int
    before_tokens: int
    after_tokens: int
    archive_path: str | None = None
    anchor_path: str | None = None

    @property
    def saved_ratio(self) -> float:
        if self.before_tokens <= 0:
            return 0.0
        return max(0.0, 1.0 - self.after_tokens / self.before_tokens)


def estimate_tokens(messages: list[Any]) -> int:
    """
    启发式 token 估算（字符数 / 3）。

    @param messages 消息列表
    @return 估算 token
    """
    total = 0
    for msg in messages:
        content = getattr(msg, "content", msg)
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += len(str(content))
        else:
            total += len(str(content))
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            total += len(str(tool_calls))
    return max(1, total // 3)


def _message_to_dict(msg: BaseMessage) -> dict[str, Any]:
    role = "unknown"
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    return {
        "role": role,
        "content": getattr(msg, "content", ""),
        "tool_calls": getattr(msg, "tool_calls", None),
    }


def _export_session_archive(
    workspace: Path,
    session_id: str,
    messages: list[BaseMessage],
) -> str | None:
    """
    导出完整对话到 jsonl。

    @param workspace 工作区根
    @param session_id 会话 ID
    @param messages 消息列表
    @return 归档路径
    """
    from llgraph.session.session_manifest import session_archive_jsonl_path

    path = session_archive_jsonl_path(workspace, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as handle:
            for msg in messages:
                handle.write(json.dumps(_message_to_dict(msg), ensure_ascii=False) + "\n")
        return str(path)
    except OSError:
        return None


def _strip_ephemeral_system_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """移除旧 anchor / 旧 summary，保留业务消息。"""
    return [
        m
        for m in messages
        if not is_conversation_anchor_message(m)
        and not is_conversation_summary_message(m)
    ]


class ContextCompressor:
    """上下文压缩器。"""

    def __init__(self, workspace: Path, session_id: str) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self.settings = resolve_context_settings(workspace)

    def should_auto_compress(self, messages: list[BaseMessage]) -> bool:
        """
        是否应自动压缩。

        @param messages 当前消息
        @return 是否超过阈值
        """
        from llgraph.context.incremental_context import resolve_auto_compress_threshold

        tokens = estimate_tokens(messages)
        threshold = resolve_auto_compress_threshold(self.settings)
        return tokens >= threshold

    def compress(
        self,
        messages: list[BaseMessage],
        *,
        force: bool = False,
        preserve_current_turn: bool | None = None,
    ) -> tuple[list[BaseMessage], CompressReport | None]:
        """
        压缩消息列表（Tier1~3）。

        @param messages 原始消息
        @param force 强制压缩（忽略阈值）
        @param preserve_current_turn cursor 策略：True 保留当前 user 轮；False 换窗仅 manifest+anchor；None 按策略默认
        @return (新消息列表, 报告)；无需压缩时返回原列表与 None
        """
        before_tokens = estimate_tokens(messages)
        if not force and not self.should_auto_compress(messages):
            return messages, None

        manifest_msgs = [m for m in messages if is_session_manifest_message(m)]
        pinned_manifest = manifest_msgs[-1] if manifest_msgs else None

        unpinned = _strip_ephemeral_system_messages(messages)
        unpinned = [m for m in unpinned if not is_session_manifest_message(m)]

        if preserve_current_turn is None:
            preserve_current_turn = not is_auto_compress_strategy(self.settings.compress_strategy)

        token_budget = int(
            self.settings.max_tokens_estimate * self.settings.keep_recent_token_ratio
        )
        to_compress, to_keep = split_messages_for_compress_strategy(
            unpinned,
            strategy=self.settings.compress_strategy,
            preserve_current_turn=preserve_current_turn,
            token_budget=token_budget,
            min_user_turns=self.settings.keep_recent_turns,
            estimate_tokens=estimate_tokens,
        )
        if not to_compress:
            return messages, None

        # Tier1：待压缩段内超长 tool 输出掩码
        to_compress = compact_tool_messages_for_compress(
            to_compress,
            self.workspace,
            max_chars=self.settings.compress_tool_mask_max_chars,
        )

        archive_path = None
        if self.settings.session_archive_on_compress:
            archive_path = _export_session_archive(
                self.workspace, self.session_id, messages
            )

        spill_dir = self.settings.spill_dir
        merged_sections, anchor_saved = run_anchor_update(
            self.workspace,
            self.session_id,
            to_compress,
            archive_path=archive_path,
            spill_dir=spill_dir,
            compress_model=self.settings.compress_model,
            retrieval_enabled=self.settings.compress_retrieval_enabled,
            retrieval_top_k=self.settings.compress_retrieval_top_k,
            summary_chunk_chars=self.settings.compress_summary_chunk_chars,
        )
        anchor_msg = build_conversation_anchor_system_message(
            self.workspace,
            self.session_id,
            merged_sections,
        )

        session = load_session_from_manifest(self.workspace, self.session_id)
        if pinned_manifest is None:
            pinned_manifest = build_session_manifest_system_message(
                self.workspace,
                self.session_id,
                session,
                "",
                archive_path=archive_path,
                spill_dir=spill_dir,
                anchor_path=anchor_saved or None,
            )

        new_messages: list[BaseMessage] = [
            pinned_manifest,
            anchor_msg,
            *to_keep,
        ]
        after_tokens = estimate_tokens(new_messages)
        report = CompressReport(
            before_count=len(messages),
            after_count=len(new_messages),
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            archive_path=archive_path,
            anchor_path=anchor_saved,
        )
        return new_messages, report


def apply_compress_to_agent_state(
    agent,
    *,
    thread_id: str,
    workspace: Path,
    force: bool = False,
    preserve_current_turn: bool | None = None,
) -> CompressReport | None:
    """
    从 agent 状态读取、压缩并写回 messages.jsonl。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区
    @param force 是否强制压缩
    @param preserve_current_turn cursor 策略切分参数；None 为 invoke 前换窗（False）
    @return 压缩报告
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception:
        return None
    messages = list((state.values or {}).get("messages") or [])
    if not messages:
        return None

    compressor = ContextCompressor(workspace, session_id=thread_id)
    if preserve_current_turn is None:
        preserve_current_turn = not is_auto_compress_strategy(compressor.settings.compress_strategy)

    new_messages, report = compressor.compress(
        messages,
        force=force,
        preserve_current_turn=preserve_current_turn,
    )
    if report is None:
        return None

    agent.update_state(config, {"messages": new_messages})
    from llgraph.session.session_file_store import save_session_messages

    save_session_messages(workspace, thread_id, new_messages)

    from llgraph.session.session_manifest import sync_session_manifest_after_compress

    session = load_session_from_manifest(workspace, thread_id)
    sync_session_manifest_after_compress(
        agent,
        thread_id=thread_id,
        workspace=workspace,
        session=session,
        archive_path=report.archive_path,
        anchor_path=report.anchor_path,
    )
    return report


def maybe_compress_during_react(
    agent: Any,
    *,
    thread_id: str,
    workspace: Path,
) -> CompressReport | None:
    """
    ReAct 循环中途接近上下文上限时压缩（cursor 策略：保留当前 user 轮，远早段 LLM 摘要）。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区根
    @return 压缩报告；未触发时 None
    """
    settings = resolve_context_settings(workspace)
    if not settings.compress_during_react:
        return None
    return apply_compress_to_agent_state(
        agent,
        thread_id=thread_id,
        workspace=workspace,
        force=False,
        preserve_current_turn=True,
    )


def format_compress_report(report: CompressReport) -> str:
    """
    格式化压缩报告。

    @param report 压缩报告
    @return 多行摘要
    """
    pct = int(report.saved_ratio * 100)
    msg = (
        f"已压缩: 消息 {report.before_count}→{report.after_count}, "
        f"估算 token {report.before_tokens}→{report.after_tokens}（约释放 {pct}%）"
    )
    if report.archive_path:
        msg += f"\n归档: {report.archive_path}"
    if report.anchor_path:
        msg += f"\n锚点: {report.anchor_path}"
    return msg
