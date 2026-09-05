"""对话上下文压缩：Tier1 切分 + Tier2 锚点摘要。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from llgraph.context.context_message_split import split_messages_for_compress_strategy
from llgraph.context.context_settings import is_auto_compress_strategy, resolve_context_settings
from llgraph.context.conversation_anchor import (
    build_conversation_anchor_message,
    is_conversation_anchor_message,
    is_conversation_summary_message,
    load_session_from_manifest,
    run_anchor_update,
)
from llgraph.session.session_manifest import (
    build_session_manifest_message,
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
    elapsed_sec: float = 0.0
    llm_sec: float = 0.0
    trigger: str = "auto"
    skipped_reason: str | None = None

    @property
    def saved_ratio(self) -> float:
        if self.before_tokens <= 0:
            return 0.0
        return max(0.0, 1.0 - self.after_tokens / self.before_tokens)


def estimate_text_tokens(text: str) -> int:
    """
    单段文本的 token 估算（与 estimate_tokens 同一启发式）。

    estimate_tokens 接的是**消息列表**：误传字符串会逐字符走一遍 isinstance 分支，
    30 轮工具链能把出站组装从 4ms 拖到 366ms。需要估单段正文时用本函数。

    @param text 文本
    @return 估算 token
    """
    return max(1, len(text) // 3)


def estimate_tokens(messages: list[Any]) -> int:
    """
    启发式 token 估算（字符数 / 3）。

    HumanMessage 含 image_ref 时仅计文字；内联 image 仅计文字 + 粗算视觉占位。

    @param messages 消息列表
    @return 估算 token
    """
    from langchain_core.messages import HumanMessage

    from llgraph.core.user_message_content import (
        human_content_has_image_refs,
        human_content_has_inline_images,
        human_content_text_for_llm,
    )

    total = 0
    for msg in messages:
        content = getattr(msg, "content", msg)
        if isinstance(msg, HumanMessage) and isinstance(content, list):
            text = human_content_text_for_llm(content)
            total += len(text)
            if human_content_has_inline_images(content):
                n = sum(
                    1
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "image"
                )
                total += n * 1500
            continue
        if isinstance(msg, HumanMessage) and human_content_has_image_refs(content):
            total += len(human_content_text_for_llm(content))
            continue
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


# 待压缩段低于此估算 token 时不调 LLM 摘要（避免 68K→63K 反复压、收益极低）
_MIN_COMPRESS_SPAN_TOKENS = 20_000
# 刚压完后的 token 回升在此比例内不再触发（防抖）
_COMPRESS_RETRIGGER_RATIO = 1.05


def _anchor_recently_updated(workspace: Path, thread_id: str) -> bool:
    """@return 锚点文件是否存在（本会话曾成功压缩过）"""
    from llgraph.session.session_manifest import conversation_anchor_json_path

    return conversation_anchor_json_path(workspace, thread_id).is_file()


def _compress_span_too_small(to_compress: list[BaseMessage]) -> bool:
    return estimate_tokens(to_compress) < _MIN_COMPRESS_SPAN_TOKENS


def replace_agent_messages(agent: Any, config: dict[str, Any], messages: list[BaseMessage]) -> None:
    """
    用新列表整体替换 agent 内存中的 messages（add_messages 默认会追加）。

    @param agent LangGraph agent
    @param config configurable thread 配置
    @param messages 替换后的完整消息链
    """
    agent.update_state(
        config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]},
    )


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
        是否应自动压缩（整窗占用达到 auto_compress_ratio）。

        @param messages 当前消息
        @return 是否超过阈值
        """
        from llgraph.context.incremental_context import resolve_auto_compress_threshold

        tokens = estimate_tokens(messages)
        threshold = resolve_auto_compress_threshold(self.settings)
        if tokens < threshold:
            return False
        if _anchor_recently_updated(self.workspace, self.session_id):
            # 防抖：刚压完略有回升时不立刻再压
            if tokens < int(threshold * _COMPRESS_RETRIGGER_RATIO):
                return False
        return True

    def resolve_compress_trigger(
        self,
        messages: list[BaseMessage],
        *,
        force: bool = False,
        trigger: str = "auto",
    ) -> str | None:
        """
        决定本次是否压缩及有效触发名（仅满窗阈值 / force）。

        @param messages 当前消息
        @param force 强制压缩
        @param trigger 调用方传入的触发名（invoke/react/auto/…）
        @return 有效 trigger；不压缩返回 None
        """
        if force:
            return trigger or "force"
        if self.should_auto_compress(messages):
            return trigger if trigger and trigger != "auto" else "threshold"
        return None

    def compress(
        self,
        messages: list[BaseMessage],
        *,
        force: bool = False,
        preserve_current_turn: bool | None = None,
        trigger: str = "auto",
    ) -> tuple[list[BaseMessage], CompressReport | None]:
        """
        压缩消息列表（Tier1 + Tier2）。

        触发：force 或整窗占用达到 auto_compress_ratio（默认约 85%）。

        @param messages 原始消息
        @param force 强制压缩（忽略阈值）
        @param preserve_current_turn cursor 策略：True 保留当前 user 轮；False 换窗仅 manifest+anchor；None 按策略默认
        @return (新消息列表, 报告)；无需压缩时返回原列表与 None
        """
        wall_start = time.perf_counter()
        before_tokens = estimate_tokens(messages)
        effective_trigger = self.resolve_compress_trigger(
            messages,
            force=force,
            trigger=trigger,
        )
        if effective_trigger is None:
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

        if not force and estimate_tokens(to_compress) < _MIN_COMPRESS_SPAN_TOKENS:
            return messages, None

        archive_path = None
        if self.settings.session_archive_on_compress:
            archive_path = _export_session_archive(
                self.workspace, self.session_id, messages
            )

        spill_dir = self.settings.spill_dir
        llm_start = time.perf_counter()
        merged_sections, anchor_saved = run_anchor_update(
            self.workspace,
            self.session_id,
            to_compress,
            archive_path=archive_path,
            spill_dir=spill_dir,
            compress_model=self.settings.compress_model,
            summary_chunk_chars=self.settings.compress_summary_chunk_chars,
        )
        llm_sec = time.perf_counter() - llm_start
        anchor_msg = build_conversation_anchor_message(
            self.workspace,
            self.session_id,
            merged_sections,
        )

        session = load_session_from_manifest(self.workspace, self.session_id)
        if pinned_manifest is None:
            pinned_manifest = build_session_manifest_message(
                self.workspace,
                self.session_id,
                session,
                "",
                archive_path=archive_path,
                spill_dir=spill_dir,
                anchor_path=anchor_saved or None,
            )

        from llgraph.context.message_normalize import reorder_pinned_session_messages

        new_messages = reorder_pinned_session_messages(
            [*to_keep, pinned_manifest, anchor_msg]
        )
        after_tokens = estimate_tokens(new_messages)
        report = CompressReport(
            before_count=len(messages),
            after_count=len(new_messages),
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            archive_path=archive_path,
            anchor_path=anchor_saved,
            elapsed_sec=time.perf_counter() - wall_start,
            llm_sec=llm_sec,
            trigger=effective_trigger,
        )
        return new_messages, report


def apply_compress_to_agent_state(
    agent,
    *,
    thread_id: str,
    workspace: Path,
    force: bool = False,
    preserve_current_turn: bool | None = None,
    trigger: str = "auto",
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
    except Exception as exc:
        from llgraph.session.session_run_log import log_react_phase

        log_react_phase(
            workspace,
            thread_id,
            phase="compress_get_state_error",
            detail={"trigger": trigger},
            error=exc,
        )
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
        trigger=trigger,
    )
    if report is None:
        return None

    from llgraph.session.session_run_log import log_react_phase

    log_react_phase(
        workspace,
        thread_id,
        phase="compress_applied",
        detail={
            "trigger": trigger,
            "tokens_before": report.before_tokens,
            "tokens_after": report.after_tokens,
            "llm_sec": round(report.llm_sec, 3),
            "elapsed_sec": round(report.elapsed_sec, 3),
        },
    )

    replace_agent_messages(agent, config, new_messages)
    from llgraph.session.session_file_store import save_agent_session_messages

    save_agent_session_messages(workspace, thread_id, new_messages, sync_pool=True)

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


def peek_react_compress_needed(
    agent: Any,
    *,
    thread_id: str,
    workspace: Path,
) -> bool:
    """
    预判 ReAct 中途是否将触发压缩（不调 LLM、不写状态）。

    用于 UI：仅在真正需要摘要时展示「压缩摘要 LLM 调用中」。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区根
    @return 是否将压缩
    """
    settings = resolve_context_settings(workspace)
    if not settings.compress_during_react:
        return False
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception:
        return False
    messages = list((state.values or {}).get("messages") or [])
    if not messages:
        return False
    compressor = ContextCompressor(workspace, session_id=thread_id)
    if compressor.resolve_compress_trigger(messages, trigger="react") is None:
        return False
    # 阈值路径：确认切分后有足够可压段
    if compressor.should_auto_compress(messages):
        preserve_current_turn = True
        unpinned = _strip_ephemeral_system_messages(messages)
        unpinned = [m for m in unpinned if not is_session_manifest_message(m)]
        token_budget = int(
            compressor.settings.max_tokens_estimate * compressor.settings.keep_recent_token_ratio
        )
        to_compress, _to_keep = split_messages_for_compress_strategy(
            unpinned,
            strategy=compressor.settings.compress_strategy,
            preserve_current_turn=preserve_current_turn,
            token_budget=token_budget,
            min_user_turns=compressor.settings.keep_recent_turns,
            estimate_tokens=estimate_tokens,
        )
        if not to_compress or _compress_span_too_small(to_compress):
            return False
    return True


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
        trigger="react",
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
    if report.elapsed_sec > 0:
        msg += f"\n耗时: {_format_compress_duration(report.elapsed_sec)}"
        if report.llm_sec > 0:
            msg += f"（LLM 摘要 {_format_compress_duration(report.llm_sec)}）"
    if report.trigger:
        msg += f"\n触发: {report.trigger}"
    if report.archive_path:
        msg += f"\n归档: {report.archive_path}"
    if report.anchor_path:
        msg += f"\n锚点: {report.anchor_path}"
    return msg


def _format_compress_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"
