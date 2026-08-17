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
