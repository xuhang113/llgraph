"""Web / 排查用：上下文分项内容与消息预览（只读）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llgraph.context.context_builder import build_workspace_context_block
from llgraph.context.context_compressor import estimate_tokens
from llgraph.context.context_session import ContextSession
from llgraph.context.context_settings import resolve_context_settings
from llgraph.context.context_stats import collect_context_usage
from llgraph.context.conversation_anchor import (
    is_conversation_anchor_message,
    is_conversation_summary_message,
)
from llgraph.context.incremental_context import resolve_auto_compress_threshold
from llgraph.core.agent import build_system_prompt
from llgraph.core.tools import get_agent_tools
from llgraph.session.session_manifest import is_session_manifest_message


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


def _message_kind(msg: BaseMessage) -> str:
    if is_session_manifest_message(msg):
        return "manifest"
    if is_conversation_anchor_message(msg):
        return "anchor"
    if is_conversation_summary_message(msg):
        return "summary"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, ToolMessage):
        return "tool"
    return "other"


def _preview_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n…（还有 {len(text) - max_chars} 字符）", True


def _inspect_message(msg: BaseMessage, *, index: int, max_preview_chars: int) -> dict[str, Any]:
    from llgraph.context.context_stats import _message_content_chars

    role = "system"
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, ToolMessage):
        role = "tool"

    text = _message_text(getattr(msg, "content", ""))
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        tc_text = json.dumps(tool_calls, ensure_ascii=False, default=str)
        text = (text + "\n\n[tool_calls]\n" + tc_text).strip()

    chars = _message_content_chars(msg)
    tokens = max(0, chars // 3)
    preview, truncated = _preview_text(text, max_chars=max_preview_chars)

    item: dict[str, Any] = {
        "index": index,
        "role": role,
        "kind": _message_kind(msg),
        "tokens": tokens,
        "chars": chars,
        "preview": preview,
        "truncated": truncated,
    }
    if isinstance(msg, ToolMessage):
        name = getattr(msg, "name", None)
        if name:
            item["tool_name"] = str(name)
    return item


_BREAKDOWN_TITLES: dict[str, str] = {
    "system_prompt": "System prompt",
    "tool_definitions": "Tool definitions",
    "mcp": "MCP",
    "rules": "Rules",
    "skills": "Skills",
    "markdowns_index": "Markdowns index",
    "summarized_conversation": "Summarized conversation",
    "conversation": "Conversation",
}

_BREAKDOWN_ORDER = (
    "system_prompt",
    "tool_definitions",
    "mcp",
    "rules",
    "skills",
    "markdowns_index",
    "summarized_conversation",
    "conversation",
)


def _split_stored_messages_for_preview(
    stored: list[BaseMessage],
) -> tuple[list[tuple[int, BaseMessage]], list[tuple[int, BaseMessage]]]:
    """@return (压缩摘要消息, 普通对话消息)"""
    summarized: list[tuple[int, BaseMessage]] = []
    conversation: list[tuple[int, BaseMessage]] = []
    for idx, msg in enumerate(stored):
        kind = _message_kind(msg)
        if kind in ("anchor", "summary"):
            summarized.append((idx, msg))
        else:
            conversation.append((idx, msg))
    return summarized, conversation


def _anchor_preview_from_disk(
    workspace: Path,
    thread_id: str,
    *,
    preview_limit: int,
) -> dict[str, Any] | None:
    """
    从 conversation_anchor.json 读取压缩摘要预览（落盘但尚未注入 messages 时）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param preview_limit 预览字符上限
    @return preview / tokens；无锚点返回 None
    """
    from llgraph.context.conversation_anchor import (
        conversation_anchor_json_path,
        format_anchor_system_message,
        load_anchor_sections,
    )
    from llgraph.context.context_stats import estimate_text_tokens

    tid = thread_id.strip()
    if not tid:
        return None
    sections = load_anchor_sections(workspace, tid)
    if not any(str(v or "").strip() for v in sections.values()):
        return None
    path = conversation_anchor_json_path(workspace, tid)
    from llgraph.session.session_manifest import _rel_workspace_path

    rel = _rel_workspace_path(workspace, path)
    text = format_anchor_system_message(sections, anchor_path=rel)
    preview, truncated = _preview_text(text, max_chars=preview_limit)
    return {
        "preview": preview,
        "truncated": truncated,
        "tokens": estimate_text_tokens(text),
    }


def _build_breakdown_sections(
    *,
    workspace: Path,
    thread_id: str = "",
    context_session: ContextSession,
    stored: list[BaseMessage],
    breakdown: Any,
    tools: list,
    mcp_names: set[str],
    preview_limit: int,
    allow_write: bool,
    web_search_enabled: bool,
    last_user_message: str = "",
) -> list[dict[str, Any]]:
    """构建 Token 分项可展开预览（对齐 Web 上下文面板）。"""
    from llgraph.context.context_stats import ContextUsageBreakdown

    assert isinstance(breakdown, ContextUsageBreakdown)

    token_by_key: dict[str, int] = {
        "system_prompt": breakdown.system_prompt,
        "tool_definitions": breakdown.tool_definitions,
        "mcp": breakdown.mcp,
        "rules": breakdown.rules,
        "skills": breakdown.skills,
        "markdowns_index": breakdown.markdowns_index,
        "summarized_conversation": breakdown.summarized_conversation,
        "conversation": breakdown.conversation,
    }

    previews: dict[str, dict[str, Any]] = {}

    system_text = build_system_prompt(
        workspace,
        allow_write=allow_write,
        web_search_enabled=web_search_enabled,
    )
    sys_preview, sys_trunc = _preview_text(system_text, max_chars=preview_limit)
    previews["system_prompt"] = {
        "preview": sys_preview,
        "truncated": sys_trunc,
    }

    builtin_lines: list[str] = []
    mcp_lines: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", "?")
        desc = str(getattr(tool, "description", "") or "")[:200]
        line = f"- {name}: {desc}"
        if name in mcp_names:
            mcp_lines.append(line)
        else:
            builtin_lines.append(line)
    tools_preview, tools_trunc = _preview_text(
        "\n".join(builtin_lines) if builtin_lines else "(无内置工具)",
        max_chars=preview_limit,
    )
    previews["tool_definitions"] = {
        "preview": tools_preview,
        "truncated": tools_trunc,
    }
    if mcp_lines:
        mcp_preview, mcp_trunc = _preview_text(
            "\n".join(mcp_lines),
            max_chars=preview_limit,
        )
        previews["mcp"] = {"preview": mcp_preview, "truncated": mcp_trunc}

    catalog_block = build_workspace_context_block(
        workspace, context_session, last_user_message
    )
    if catalog_block.strip():
        if breakdown.skills > 0:
            skills_preview, skills_trunc = _preview_text(
                catalog_block,
                max_chars=preview_limit,
            )
            previews["skills"] = {
                "preview": skills_preview,
                "truncated": skills_trunc,
            }
        elif breakdown.rules > 0:
            rules_preview, rules_trunc = _preview_text(
                catalog_block,
                max_chars=preview_limit,
            )
            previews["rules"] = {
                "preview": rules_preview,
                "truncated": rules_trunc,
            }

    summarized_msgs, conversation_msgs = _split_stored_messages_for_preview(stored)
    disk_anchor = _anchor_preview_from_disk(
        workspace,
        thread_id,
        preview_limit=preview_limit,
    )
    if summarized_msgs:
        previews["summarized_conversation"] = {
            "preview": "",
            "truncated": False,
            "messages": [
                _inspect_message(msg, index=idx + 1, max_preview_chars=preview_limit)
                for idx, msg in summarized_msgs
            ],
        }
    elif disk_anchor:
        previews["summarized_conversation"] = {
            "preview": disk_anchor["preview"],
            "truncated": bool(disk_anchor.get("truncated", False)),
            "from_disk": True,
        }
    if conversation_msgs:
        previews["conversation"] = {
            "preview": "",
            "truncated": False,
            "messages": [
                _inspect_message(msg, index=idx + 1, max_preview_chars=preview_limit)
                for idx, msg in conversation_msgs
            ],
        }

    sections: list[dict[str, Any]] = []
    for key in _BREAKDOWN_ORDER:
        tokens = token_by_key.get(key, 0)
        meta = previews.get(key, {"preview": "(暂无预览)", "truncated": False})
        if key == "summarized_conversation":
            if not thread_id.strip() and not summarized_msgs and tokens <= 0 and not disk_anchor:
                continue
        elif key == "conversation":
            if not conversation_msgs and tokens <= 0:
                continue
        elif tokens <= 0:
            continue
        title = _BREAKDOWN_TITLES.get(key, key)
        if key == "tool_definitions" and breakdown.tool_count:
            title = f"Tool definitions（{breakdown.tool_count - breakdown.mcp_tool_count} 个）"
        elif key == "mcp" and breakdown.mcp_tool_count:
            title = f"MCP（{breakdown.mcp_tool_count} 个）"
        if key == "summarized_conversation" and meta.get("messages"):
            msg_tokens = sum(
                int(m.get("tokens") or 0)
                for m in meta["messages"]
                if m.get("kind") in ("anchor", "summary")
            )
            if msg_tokens > tokens:
                tokens = msg_tokens
        elif key == "summarized_conversation" and disk_anchor:
            disk_tokens = int(disk_anchor.get("tokens") or 0)
            if disk_tokens > tokens:
                tokens = disk_tokens
        section: dict[str, Any] = {
            "key": key,
            "title": title,
            "tokens": tokens,
            "preview": meta.get("preview", ""),
            "truncated": bool(meta.get("truncated", False)),
        }
        if meta.get("from_disk"):
            section["from_disk"] = True
        if key == "summarized_conversation" and not summarized_msgs and not disk_anchor and tokens <= 0:
            section["empty_hint"] = (
                "尚未压缩。上下文接近阈值（约 85%）时会自动压缩；"
                "也可点击下方「压缩历史」手动生成摘要。"
            )
        if meta.get("messages"):
            section["messages"] = meta["messages"]
        sections.append(section)
    return sections


@dataclass
class ContextDetailPayload:
    """结构化上下文详情（供 Web JSON）。"""

    usage: dict[str, Any]
    settings: dict[str, Any]
    compress_threshold: int
    config_help: str
    breakdown_sections: list[dict[str, Any]] = field(default_factory=list)
    fixed_sections: list[dict[str, Any]] = field(default_factory=list)
    stored_messages: list[dict[str, Any]] = field(default_factory=list)
    dispatch_messages: list[dict[str, Any]] = field(default_factory=list)
    dispatch_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_context_detail(
    workspace: Path,
    *,
    context_session: ContextSession,
    thread_id: str = "",
    allow_write: bool = False,
    web_search_enabled: bool = False,
    mcp_tools: list | None = None,
    max_preview_chars: int = 6000,
) -> ContextDetailPayload:
    """
    汇总 token 用量 + 落盘消息 + 下轮出站预览。

    @param workspace 工作区根
    @param context_session Rule/Skill 会话
    @param thread_id Agent 会话 ID
    @param allow_write 是否含写工具
    @param web_search_enabled 是否含 web_search
    @param mcp_tools MCP 工具列表
    @param max_preview_chars 单条 preview 最大字符
    @return 结构化详情
    """
    preview_limit = max(500, min(50_000, int(max_preview_chars)))
    tid = thread_id.strip()

    breakdown = collect_context_usage(
        workspace,
        context_session=context_session,
        allow_write=allow_write,
        web_search_enabled=web_search_enabled,
        thread_id=tid,
        mcp_tools=mcp_tools,
    )
    settings = resolve_context_settings(workspace)
    limit = settings.max_tokens_estimate
    total = breakdown.total
    ratio = total / limit if limit > 0 else 0.0

    usage = {
        "total": total,
        "limit": limit,
        "ratio": ratio,
        "pct": min(100, int(ratio * 100)),
        "message_count": breakdown.message_count,
        "tool_count": breakdown.tool_count,
        "mcp_tool_count": breakdown.mcp_tool_count,
        "breakdown": {
            "system_prompt": breakdown.system_prompt,
            "tool_definitions": breakdown.tool_definitions,
            "rules": breakdown.rules,
            "skills": breakdown.skills,
            "mcp": breakdown.mcp,
            "markdowns_index": breakdown.markdowns_index,
            "summarized_conversation": breakdown.summarized_conversation,
            "conversation": breakdown.conversation,
        },
        "has_session": breakdown.message_count > 0,
    }

    from llgraph.core.model_context_window import format_context_budget_note

    usage["budget_note"] = format_context_budget_note(
        workspace,
        max_tokens=settings.max_tokens_estimate,
        source=settings.budget_source,
        model_id=settings.context_model_id,
        ratio=settings.auto_compress_ratio,
    )

    settings_snapshot = {
        "compress_strategy": settings.compress_strategy,
        "auto_compress_ratio": settings.auto_compress_ratio,
        "compress_during_react": settings.compress_during_react,
        "incremental_tool_prune": settings.incremental_tool_prune,
        "keep_recent_tool_messages": settings.keep_recent_tool_messages,
        "dispatch_tool_chain_compress": settings.dispatch_tool_chain_compress,
        "dispatch_keep_full_tool_messages": settings.dispatch_keep_full_tool_messages,
        "tool_prune_token_ratio": settings.tool_prune_token_ratio,
        "grep_max_inline_chars": settings.grep_max_inline_chars,
        "tool_result_max_chars": settings.tool_result_max_chars,
    }

    from llgraph.context.context_settings import format_context_config_help

    config_help = format_context_config_help(workspace)

    system_text = build_system_prompt(
        workspace,
        allow_write=allow_write,
        web_search_enabled=web_search_enabled,
    )
    sys_preview, sys_trunc = _preview_text(system_text, max_chars=preview_limit)
    fixed_sections: list[dict[str, Any]] = [
        {
            "key": "system_prompt",
            "title": "System prompt",
            "tokens": breakdown.system_prompt,
            "preview": sys_preview,
            "truncated": sys_trunc,
        },
    ]

    tools = get_agent_tools(
        workspace_root=workspace,
        allow_write=allow_write,
        mcp_tools=mcp_tools,
        context_spill=None,
        web_search_enabled=web_search_enabled,
    )
    tool_lines: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", "?")
        desc = str(getattr(tool, "description", "") or "")[:200]
        tool_lines.append(f"- {name}: {desc}")
    tools_preview, tools_trunc = _preview_text(
        "\n".join(tool_lines) if tool_lines else "(无工具)",
        max_chars=preview_limit,
    )
    fixed_sections.append(
        {
            "key": "tool_definitions",
            "title": f"Tool definitions（{len(tools)} 个）",
            "tokens": breakdown.tool_definitions + breakdown.mcp,
            "preview": tools_preview,
            "truncated": tools_trunc,
        }
    )

    stored: list[BaseMessage] = []
    if tid:
        from llgraph.session.session_file_store import load_session_messages

        try:
            stored = load_session_messages(workspace, tid)
        except Exception:
            stored = []

    stored_items = [
        _inspect_message(msg, index=i + 1, max_preview_chars=preview_limit)
        for i, msg in enumerate(stored)
    ]

    dispatch_items: list[dict[str, Any]] = []
    dispatch_note = "无会话消息"
    if stored:
        from llgraph.context.incremental_context import prune_tool_messages_for_dispatch

        trimmed = prune_tool_messages_for_dispatch(list(stored), workspace, settings)
        if trimmed != list(stored):
            dispatch_note = (
                f"出站不裁 user 轮；工具链指针压缩：落盘 {len(stored)} 条 → "
                f"预览态约 {len(trimmed)} 条结构不变"
            )
        else:
            dispatch_note = f"出站不裁 user 轮（落盘 {len(stored)} 条；工具结果按阈值指针化）"
        dispatch_items = [
            _inspect_message(msg, index=i + 1, max_preview_chars=preview_limit)
            for i, msg in enumerate(trimmed)
        ]

    from llgraph.context.context_stats import _mcp_tool_names

    mcp_names = _mcp_tool_names(mcp_tools)
    breakdown_sections = _build_breakdown_sections(
        workspace=workspace,
        thread_id=tid,
        context_session=context_session,
        stored=stored,
        breakdown=breakdown,
        tools=tools,
        mcp_names=mcp_names,
        preview_limit=preview_limit,
        allow_write=allow_write,
        web_search_enabled=web_search_enabled,
    )

    return ContextDetailPayload(
        usage=usage,
        settings=settings_snapshot,
        compress_threshold=resolve_auto_compress_threshold(settings),
        config_help=config_help,
        breakdown_sections=breakdown_sections,
        fixed_sections=fixed_sections,
        stored_messages=stored_items,
        dispatch_messages=dispatch_items,
        dispatch_note=dispatch_note,
    )
