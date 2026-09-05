"""Prompt Cache：system/tools 断点 + 对话前缀自动缓存。"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from llgraph.core.prompt_cache_settings import (
    PromptCacheSettings,
    prompt_cache_enabled_for_model,
    resolve_prompt_cache_settings,
)


def build_cache_control(settings: PromptCacheSettings) -> dict[str, str]:
    """
    构建 cache_control 字典。

    @param settings Prompt cache 配置
    @return Anthropic cache_control
    """
    return {"type": "ephemeral", "ttl": settings.ttl}


def tag_tools_for_prompt_cache(
    tools: list[Any],
    cache_control: dict[str, str],
) -> list[Any]:
    """
    在最后一个 BaseTool 的 extras 上打 cache_control（缓存整段 tool 定义）。

    @param tools 工具列表
    @param cache_control cache_control 字典
    @return 新列表（不修改原 tool 时可变对象）
    """
    if not tools or not cache_control:
        return tools

    last_idx = -1
    last_tool: BaseTool | None = None
    for idx, item in enumerate(tools):
        if isinstance(item, BaseTool):
            last_idx = idx
            last_tool = item

    if last_tool is None or last_idx < 0:
        return tools

    new_extras = {**(last_tool.extras or {}), "cache_control": dict(cache_control)}
    try:
        cloned = last_tool.model_copy(update={"extras": new_extras})
    except AttributeError:
        cloned = copy(last_tool)
        cloned.extras = new_extras

    out = list(tools)
    out[last_idx] = cloned
    return out


def build_cached_system_message(
    *,
    stable_text: str,
    cache_control: dict[str, str],
) -> SystemMessage:
    """
    构建带断点的 SystemMessage。

    动静边界之前为可缓存静态前缀；之后为动态段（不打 cache_control，
    避免环境/路由变化击穿静态缓存）。

    @param stable_text Agent 系统规范（build_system_prompt，可含边界标记）
    @param cache_control 断点参数
    @return SystemMessage
    """
    from llgraph.core.prompt_boundary import split_system_prompt_at_boundary

    static, dynamic = split_system_prompt_at_boundary(stable_text)
    if not static and not dynamic:
        return SystemMessage(content="")
    blocks: list[dict[str, Any]] = []
    if static:
        blocks.append(
            {
                "type": "text",
                "text": static,
                "cache_control": dict(cache_control),
            }
        )
    if dynamic:
        blocks.append({"type": "text", "text": dynamic})
    return SystemMessage(content=blocks)


# Anthropic 单请求最多 4 个 cache_control 断点；system 静态前缀与 tools 定义各占一个
MAX_CACHE_BREAKPOINTS = 4
_RESERVED_BREAKPOINTS = 2
MAX_MESSAGE_BREAKPOINTS = MAX_CACHE_BREAKPOINTS - _RESERVED_BREAKPOINTS


def _last_text_block_index(blocks: list[Any]) -> int | None:
    for i in range(len(blocks) - 1, -1, -1):
        block = blocks[i]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and str(block.get("text", "")).strip()
        ):
            return i
    return None


def _replace_content(msg: BaseMessage, blocks: list[Any]) -> BaseMessage:
    try:
        return msg.model_copy(update={"content": blocks})
    except AttributeError:
        cloned = copy(msg)
        cloned.content = blocks
        return cloned


def _as_text_blocks(msg: BaseMessage) -> BaseMessage:
    """
    把纯字符串正文改成单 text 块。

    断点只能挂在块上：若一条消息「本步带断点＝块形式、下步不带＝字符串形式」，
    序列化就会变，恰好在想缓存的那一块上把前缀打断。故凡可能成为断点的消息，
    无论本步是否带断点都统一成块形式。

    @param msg 出站消息
    @return 块形式消息（已是块或空正文时原样返回）
    """
    content = getattr(msg, "content", None)
    if not isinstance(content, str) or not content.strip():
        return msg
    return _replace_content(msg, [{"type": "text", "text": content}])


def _tag_message_for_cache(
    msg: BaseMessage,
    cache_control: dict[str, str],
) -> BaseMessage | None:
    """
    在消息最后一个非空 text 块上打 cache_control。

    ToolMessage 由 langchain-anthropic 把块级 cache_control 提升到 tool_result 层，
    因此 tool 结果同样可以当断点。

    @param msg 出站消息（正文须已是块形式）
    @param cache_control 断点参数
    @return 新消息；无可打断点的块时 None
    """
    if isinstance(msg, SystemMessage):
        return None
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return None
    idx = _last_text_block_index(content)
    if idx is None:
        return None
    blocks = list(content)
    if "cache_control" in blocks[idx]:
        return msg
    blocks[idx] = {**blocks[idx], "cache_control": dict(cache_control)}
    return _replace_content(msg, blocks)


def _is_ephemeral_tail(msg: BaseMessage) -> bool:
    from llgraph.context.investigate_harness import is_ephemeral_harness_human

    return is_ephemeral_harness_human(msg)


def last_stable_message_index(messages: list[BaseMessage]) -> int | None:
    """
    最后一条「下一步仍会原样存在」的消息下标。

    出站尾部的 <system-reminder> / 预算提醒等是 ephemeral：本步有、下步没有。
    断点若落在它上面，写进缓存的前缀下一步永远匹配不上，等于白付 1.25x 写入费
    且每步全价重算——这正是原 top-level cache_control 的问题。

    @param messages 出站消息（含合并后的 system）
    @return 下标；无可用消息时 None
    """
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, SystemMessage) or _is_ephemeral_tail(msg):
            continue
        return idx
    return None


def _turn_anchor_index(messages: list[BaseMessage], *, before: int) -> int | None:
    """
    本轮 user 消息下标：它在本轮工具链之前，压缩纪元推进也动不到它。

    @param messages 出站消息
    @param before 只找该下标之前的
    @return 下标或 None
    """
    from llgraph.context.conversation_anchor import is_pinned_session_context_message

    for idx in range(min(before, len(messages)) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, HumanMessage):
            continue
        if _is_ephemeral_tail(msg) or is_pinned_session_context_message(msg):
            continue
        return idx
    return None


def apply_message_cache_breakpoints(
    messages: list[BaseMessage],
    *,
    cache_control: dict[str, str],
    max_breakpoints: int = MAX_MESSAGE_BREAKPOINTS,
) -> list[BaseMessage]:
    """
    把对话断点打在**最后一条稳定消息**上，而不是出站列表的最后一块。

    这是 Anthropic 多轮缓存的标准用法：断点随对话前移，写入的前缀正好是下一步
    请求的真前缀。第二个断点放在本轮 user 消息上：压缩纪元推进的那一步会在本轮
    工具链中途分叉，这个断点至少兜住此前所有轮次的历史。

    @param messages 出站消息
    @param cache_control 断点参数
    @param max_breakpoints 消息级断点上限（system/tools 已各占一个）
    @return 打好断点的新列表（原列表不变）
    """
    if not messages or max_breakpoints <= 0 or not cache_control:
        return messages
    stable = last_stable_message_index(messages)
    if stable is None:
        return messages

    targets: list[int] = [stable]
    anchor: int | None = None
    if max_breakpoints > 1:
        anchor = _turn_anchor_index(messages, before=stable)
        if anchor is not None:
            targets.append(anchor)

    out = list(messages)
    # 本轮之内的 ToolMessage 轮流当「最后一条稳定消息」，形状必须一致
    for idx in range((anchor or 0), len(out)):
        if isinstance(out[idx], ToolMessage):
            out[idx] = _as_text_blocks(out[idx])
    for idx in targets:
        out[idx] = _as_text_blocks(out[idx])

    applied = 0
    for idx in targets:
        if applied >= max_breakpoints:
            break
        tagged = _tag_message_for_cache(out[idx], cache_control)
        if tagged is None:
            continue
        out[idx] = tagged
        applied += 1
    return out if applied else messages


def apply_cache_breakpoints_for_dispatch(
    messages: list[BaseMessage],
    *,
    workspace: Path | None,
    model_id: str | None,
) -> list[BaseMessage]:
    """
    出站最后一步：按配置打对话缓存断点。

    @param messages 已 redact 完成的出站消息
    @param workspace 工作区根
    @param model_id 当前模型
    @return 打好断点的消息
    """
    settings = resolve_prompt_cache_settings(workspace)
    if not settings.enabled or not settings.tag_conversation_tail:
        return messages
    if not prompt_cache_enabled_for_model(workspace, model_id):
        return messages
    if settings.min_messages and len(messages) < settings.min_messages:
        return messages
    return apply_message_cache_breakpoints(
        messages,
        cache_control=build_cache_control(settings),
    )


def apply_prompt_cache_to_llm(llm: Any, workspace: Path | None) -> Any:
    """
    旧式 top-level cache_control 绑定（诊断脚本仍在用）。

    Agent 出站路径已改为 apply_cache_breakpoints_for_dispatch：top-level 形式会把
    断点落在出站最后一块，也就是 ephemeral 提醒上，缓存永远读不回来。

    @param llm LangChain 聊天模型
    @param workspace 工作区根
    @return 绑定后的模型
    """
    settings = resolve_prompt_cache_settings(workspace)
    if not settings.enabled or not settings.tag_conversation_tail:
        return llm
    cache_control = build_cache_control(settings)
    bind = getattr(llm, "bind", None)
    if not callable(bind):
        return llm
    try:
        return bind(cache_control=cache_control)
    except Exception:
        return llm


def prepare_system_message_for_dispatch(
    messages: list[BaseMessage],
    *,
    agent_system_content: str | None,
    workspace: Path | None,
    model_id: str | None,
) -> tuple[SystemMessage | None, list[BaseMessage]]:
    """
    合并 system：仅稳定 Agent 规范；manifest/anchor 留在 messages 中。

    @param messages 已整理顺序的消息列表
    @param agent_system_content Agent 系统提示
    @param workspace 工作区根
    @param model_id 当前模型
    @return (合并后的 SystemMessage 或 None, 非 agent-system 消息)
    """
    non_system: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        non_system.append(msg)

    stable = str(agent_system_content or "").strip()
    if not stable:
        return None, non_system

    from llgraph.core.prompt_boundary import split_system_prompt_at_boundary

    static, dynamic = split_system_prompt_at_boundary(stable)
    settings = resolve_prompt_cache_settings(workspace)
    use_cache = prompt_cache_enabled_for_model(workspace, model_id) and settings.enabled
    if not use_cache:
        # 不向模型暴露边界标记本身
        merged = "\n\n".join(p for p in (static, dynamic) if p)
        return SystemMessage(content=merged), non_system

    cache_control = build_cache_control(settings)
    return (
        build_cached_system_message(
            stable_text=stable,
            cache_control=cache_control,
        ),
        non_system,
    )
