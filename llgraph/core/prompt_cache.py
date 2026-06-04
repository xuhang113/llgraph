"""Prompt Cache：system/tools 断点 + 对话前缀自动缓存。"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
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
    dynamic_text: str,
    cache_control: dict[str, str],
) -> SystemMessage:
    """
    构建带断点的单条 SystemMessage：稳定前缀可缓存，会话指针不缓存。

    @param stable_text Agent 系统规范（build_system_prompt）
    @param dynamic_text manifest + anchor 等每轮可变内容
    @param cache_control 断点参数
    @return SystemMessage
    """
    blocks: list[dict[str, Any]] = []
    stable = stable_text.strip()
    if stable:
        blocks.append(
            {
                "type": "text",
                "text": stable,
                "cache_control": dict(cache_control),
            }
        )
    dynamic = dynamic_text.strip()
    if dynamic:
        blocks.append({"type": "text", "text": dynamic})

    if not blocks:
        return SystemMessage(content="")
    if len(blocks) == 1 and "cache_control" not in blocks[0]:
        return SystemMessage(content=blocks[0]["text"])
    return SystemMessage(content=blocks)


def apply_prompt_cache_to_llm(llm: Any, workspace: Path | None) -> Any:
    """
    在 ChatAnthropic 上绑定 cache_control，用于对话尾部可缓存块（近期轮次重叠）。

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
    合并 system：启用 cache 时稳定/可变分块；否则保持字符串拼接。

    @param messages 已置顶 manifest/anchor 的消息列表
    @param agent_system_content Agent 系统提示
    @param workspace 工作区根
    @param model_id 当前模型
    @return (合并后的 SystemMessage 或 None, 非 system 业务消息)
    """
    session_parts: list[str] = []
    non_system: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            from llgraph.context.message_normalize import _message_text

            text = _message_text(getattr(msg, "content", "")).strip()
            if text:
                session_parts.append(text)
        else:
            non_system.append(msg)

    stable = str(agent_system_content or "").strip()
    dynamic = "\n\n".join(session_parts).strip()

    if not stable and not dynamic:
        return None, non_system

    settings = resolve_prompt_cache_settings(workspace)
    use_cache = prompt_cache_enabled_for_model(workspace, model_id) and settings.enabled
    if not use_cache:
        merged = "\n\n".join(part for part in (stable, dynamic) if part)
        return SystemMessage(content=merged), non_system

    cache_control = build_cache_control(settings)
    return (
        build_cached_system_message(
            stable_text=stable,
            dynamic_text=dynamic,
            cache_control=cache_control,
        ),
        non_system,
    )
