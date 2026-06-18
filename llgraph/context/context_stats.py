"""会话上下文占用估算与 /context 展示。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from llgraph.core.agent import build_system_prompt
from llgraph.core.agent_session import AgentSessionContext
from llgraph.context.context_session import ContextSession
from llgraph.context.context_settings import resolve_context_settings
from llgraph.context.context_builder import build_workspace_context_block
from llgraph.core.tools import get_agent_tools

_SUMMARY_TAGS = ("<conversation-anchor>", "<conversation-summary>")


@dataclass(frozen=True)
class ContextUsageBreakdown:
    """上下文各分项 token 估算。"""

    system_prompt: int
    tool_definitions: int
    rules: int
    skills: int
    mcp: int
    markdowns_index: int
    summarized_conversation: int
    conversation: int
    message_count: int
    tool_count: int
    mcp_tool_count: int

    @property
    def total(self) -> int:
        return (
            self.system_prompt
            + self.tool_definitions
            + self.rules
            + self.skills
            + self.mcp
            + self.markdowns_index
            + self.summarized_conversation
            + self.conversation
        )


def chars_to_tokens(chars: int) -> int:
    """
    启发式 token 估算（与 context_compressor 一致：字符数 / 3）。

    @param chars 字符数
    @return 估算 token
    """
    return max(0, chars // 3)


def estimate_text_tokens(text: str) -> int:
    """
    估算文本 token。

    @param text 文本
    @return 估算 token
    """
    if not text:
        return 0
    return chars_to_tokens(len(text))


def _format_token_count(tokens: int) -> str:
    """格式化为 ~1.2K 风格。"""
    if tokens >= 10_000:
        return f"~{tokens / 1000:.1f}K"
    if tokens >= 1000:
        return f"~{tokens / 1000:.1f}K"
    return f"~{tokens}"


def _mcp_tool_names(mcp_tools: list | None) -> set[str]:
    names: set[str] = set()
    for tool in mcp_tools or []:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _estimate_tool_schema_tokens(tools: list, mcp_names: set[str]) -> tuple[int, int, int]:
    """
    估算工具 schema token，拆分为内置与 MCP。

    @param tools 工具列表
    @param mcp_names MCP 工具名集合
    @return (builtin_tokens, mcp_tokens, tool_count_total)
    """
    builtin = 0
    mcp = 0
    mcp_count = 0
    for tool in tools:
        try:
            schema_text = json.dumps(convert_to_openai_tool(tool), ensure_ascii=False)
            size = len(schema_text)
        except Exception:
            name = str(getattr(tool, "name", "") or "")
            desc = str(getattr(tool, "description", "") or "")
            size = len(name) + len(desc)
        tokens = chars_to_tokens(size)
        name = getattr(tool, "name", "") or ""
        if name in mcp_names:
            mcp += tokens
            mcp_count += 1
        else:
            builtin += tokens
    return builtin, mcp, mcp_count


def _measure_catalog_tokens(
    workspace: Path,
    session: ContextSession,
    user_message: str,
) -> tuple[int, int]:
    """
    估算下轮 workspace-context 中 ephemeral 提示 token（Skills/Rules 目录在 manifest）。

    @return (rules_tokens, skills_tokens) 粗分；现多为 /skill 指针或写失败 hint
    """
    block = build_workspace_context_block(workspace, session, user_message)
    if not block:
        return 0, 0
    total = estimate_text_tokens(block)
    if "## 本会话已启用技能" in block:
        return 0, total
    return 0, total


def _measure_markdowns_tokens(workspace: Path, session: ContextSession) -> int:
    """Markdowns 不内联索引，仅 manifest 指针。"""
    _ = workspace, session
    return 0


def _message_content_chars(msg: BaseMessage) -> int:
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    extra = getattr(msg, "tool_calls", None)
    extra_len = len(str(extra)) if extra else 0
    return len(content) + extra_len


def _split_message_tokens(messages: list[BaseMessage]) -> tuple[int, int]:
    """
    将消息历史拆为摘要段与普通对话段 token。

    @param messages LangGraph 消息列表
    @return (summarized_tokens, conversation_tokens)
    """
    summarized = 0
    conversation = 0
    for msg in messages:
        chars = _message_content_chars(msg)
        tokens = chars_to_tokens(chars)
        content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        if isinstance(msg, SystemMessage) and any(tag in content for tag in _SUMMARY_TAGS):
            summarized += tokens
        else:
            conversation += tokens
    return summarized, conversation


def collect_context_usage(
    workspace: Path,
    *,
    context_session: ContextSession,
    last_user_message: str = "",
    allow_write: bool = False,
    web_search_enabled: bool = False,
    agent_session: AgentSessionContext | None = None,
) -> ContextUsageBreakdown:
    """
    汇总当前会话上下文各分项占用（启发式 token）。

    @param workspace 工作区根
    @param context_session Rule/Skill 会话状态
    @param last_user_message 用于 glob/技能匹配与下轮注入估算
    @param allow_write 是否包含写工具
    @param web_search_enabled 是否包含 web_search
    @param agent_session Agent 会话（读取消息历史）
    @return 分项报告
    """
    system_prompt = estimate_text_tokens(
        build_system_prompt(
            workspace,
            allow_write=allow_write,
            web_search_enabled=web_search_enabled,
        )
    )

    mcp_tools = agent_session.mcp_tools if agent_session else None
    spill = agent_session.context_spill if agent_session else None
    tools = get_agent_tools(
        workspace_root=workspace,
        allow_write=allow_write,
        mcp_tools=mcp_tools,
        context_spill=spill,
        web_search_enabled=web_search_enabled,
    )
    mcp_names = _mcp_tool_names(mcp_tools)
    tool_definitions, mcp_tokens, mcp_tool_count = _estimate_tool_schema_tokens(
        tools, mcp_names
    )

    rules, skills = _measure_catalog_tokens(workspace, context_session, last_user_message)
    markdowns_index = _measure_markdowns_tokens(workspace, context_session)

    messages: list[BaseMessage] = []
    if agent_session is not None and agent_session.with_memory:
        config = {"configurable": {"thread_id": agent_session.thread_id}}
        try:
            state = agent_session.agent.get_state(config)
            messages = list((state.values or {}).get("messages") or [])
        except Exception:
            messages = []

    summarized, conversation = _split_message_tokens(messages)

    return ContextUsageBreakdown(
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
        rules=rules,
        skills=skills,
        mcp=mcp_tokens,
        markdowns_index=markdowns_index,
        summarized_conversation=summarized,
        conversation=conversation,
        message_count=len(messages),
        tool_count=len(tools),
        mcp_tool_count=mcp_tool_count,
    )


def _render_bar(ratio: float, width: int = 28) -> str:
    """简易 ASCII 进度条。"""
    from llgraph.terminal.style import color_enabled, sty

    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    if not color_enabled():
        return "[" + "#" * filled + "-" * (width - filled) + "]"
    return (
        sty("[", "dim")
        + sty("#" * filled, "ok")
        + sty("-" * (width - filled), "dim")
        + sty("]", "dim")
    )


def format_context_usage_report(
    breakdown: ContextUsageBreakdown,
    *,
    workspace: Path,
    limit_tokens: int | None = None,
) -> str:
    """
    格式化为 /context 终端输出。

    @param breakdown 分项数据
    @param workspace 工作区根（读取压缩阈值）
    @param limit_tokens 展示用上下文上限
    @return 多行文本
    """
    from llgraph.terminal.style import sty

    settings = resolve_context_settings(workspace)
    limit = limit_tokens or settings.max_tokens_estimate
    total = breakdown.total
    ratio = total / limit if limit > 0 else 0.0
    pct = min(100, int(ratio * 100))
    full_label = f"{pct}% Full" if pct >= 85 else f"{pct}%"
    pct_style = "warn" if pct >= 85 else "ok"

    lines = [
        sty("Context", "title"),
        sty("=======", "dim"),
        f"{sty(full_label, pct_style)}    "
        f"{sty(_format_token_count(total), 'number')} / "
        f"{sty(_format_token_count(limit), 'number')} "
        f"{sty('Tokens', 'hint')}",
        "",
        _render_bar(ratio),
        "",
    ]

    rows: list[tuple[str, int]] = [
        ("System prompt", breakdown.system_prompt),
        ("Tool definitions", breakdown.tool_definitions),
        ("Rules", breakdown.rules),
        ("Skills", breakdown.skills),
        ("MCP", breakdown.mcp),
        ("Markdowns index", breakdown.markdowns_index),
        ("Summarized conversation", breakdown.summarized_conversation),
        ("Conversation", breakdown.conversation),
    ]
    name_width = max(len(name) for name, _ in rows)
    for name, tokens in rows:
        if tokens <= 0 and name == "Markdowns index":
            continue
        lines.append(
            f"{sty(name.ljust(name_width), 'label')}  "
            f"{sty(_format_token_count(tokens), 'number')}"
        )

    from llgraph.core.model_context_window import format_context_budget_note

    budget_note = format_context_budget_note(
        workspace,
        max_tokens=settings.max_tokens_estimate,
        source=settings.budget_source,
        model_id=settings.context_model_id,
        ratio=settings.auto_compress_ratio,
    )
    lines.extend(
        [
            "",
            sty(
                f"消息数: {breakdown.message_count}  |  工具: {breakdown.tool_count}"
                f"（MCP {breakdown.mcp_tool_count}）",
                "value",
            ),
            sty(budget_note, "accent"),
            sty(
                "说明: token 为字符÷3 粗算；Skills/Rules 目录在 manifest；"
                "下轮 workspace-context 仅 /skill 指针等 ephemeral 提示；",
                "hint",
            ),
            sty(
                "      历史 human 中旧版 workspace-context 仍计入 Conversation。",
                "hint",
            ),
            sty(
                "      上下文预算默认按当前模型上限（/model 切换后下条消息起更新）。",
                "hint",
            ),
        ]
    )
    from llgraph.context.context_settings import format_context_config_help

    lines.extend(["", format_context_config_help(workspace)])
    return "\n".join(lines)


def print_context_usage(
    workspace: Path,
    *,
    context_session: ContextSession,
    last_user_message: str = "",
    allow_write: bool = False,
    agent_session: AgentSessionContext | None = None,
) -> None:
    """
    打印 /context 报告。

    @param workspace 工作区根
    @param context_session Rule/Skill 会话
    @param last_user_message 最近用户消息
    @param allow_write 是否可写
    @param agent_session Agent 会话
    """
    web_enabled = (
        agent_session.web_search_enabled if agent_session is not None else False
    )
    breakdown = collect_context_usage(
        workspace,
        context_session=context_session,
        last_user_message=last_user_message,
        allow_write=allow_write,
        web_search_enabled=web_enabled,
        agent_session=agent_session,
    )
    from llgraph.terminal.output import emit_report

    emit_report(format_context_usage_report(breakdown, workspace=workspace))
