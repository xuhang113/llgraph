"""ReAct 图 recursion_limit（agent / plan 子图共用）与批量工具 nudge。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import BaseMessage

from llgraph.core.agent_config import load_agent_config

DEFAULT_REACT_MAX_TURNS = 100
REACT_MAX_TURNS_CAP = 500
# 连续 N 轮每次仅 1 个 tool_call 后，在 ToolMessage 末尾注入并行提示；0=关闭。
DEFAULT_BATCH_TOOLS_NUDGE_AFTER = 3
BATCH_TOOLS_NUDGE_AFTER_CAP = 20
# 本问内相同参数的 read/grep/失败写 是否在 ToolNode 短路径拦截；默认开。
DEFAULT_IDENTICAL_TOOL_GUARD = True


def parse_react_max_turns(raw: object, *, default: int = DEFAULT_REACT_MAX_TURNS) -> int:
    """
    解析 max_turns 配置项。

    @param raw agent.json 原始值
    @param default 缺省时的默认值
    @return 1～REACT_MAX_TURNS_CAP 之间的整数
    """
    if raw is None:
        return default
    try:
        return max(1, min(REACT_MAX_TURNS_CAP, int(raw)))
    except (TypeError, ValueError):
        return default


def resolve_agent_max_turns(workspace: Path | None) -> int:
    """
    主 Chat Agent 工具往返 / 图步上限（写入 recursion_limit）。

    @param workspace 工作区根；None 时用默认
    @return 上限
    """
    if workspace is None:
        return DEFAULT_REACT_MAX_TURNS
    cfg = load_agent_config(workspace)
    agent = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    return parse_react_max_turns(agent.get("max_turns"))


def format_tool_round_budget_line(
    messages: list[BaseMessage],
    *,
    workspace: Path | None = None,
) -> str:
    """
    告知模型本问工具往返已用/剩余（与 max_turns 对齐）。

    @param messages 当前消息
    @param workspace 工作区根
    @return 单行预算文案
    """
    from llgraph.context.investigate_harness import count_tool_rounds_since_user

    limit = resolve_agent_max_turns(workspace)
    used = count_tool_rounds_since_user(messages)
    left = max(0, limit - used)
    line = f"工具往返预算：上限 {limit}；已用 {used}；还能调用约 {left} 次。"
    if left <= 0:
        return line + " 已触及上限，本轮禁止再调工具，立即写终答。"
    if left <= 10:
        return line + " 剩余很少，优先终答，少开新检索线。"
    return line


def parse_batch_tools_nudge_after(
    raw: object,
    *,
    default: int = DEFAULT_BATCH_TOOLS_NUDGE_AFTER,
) -> int:
    """
    解析 batch_tools_nudge_after。

    @param raw agent.json 原始值
    @param default 缺省
    @return 0～CAP；0 表示关闭
    """
    if raw is None:
        return default
    try:
        return max(0, min(BATCH_TOOLS_NUDGE_AFTER_CAP, int(raw)))
    except (TypeError, ValueError):
        return default


def resolve_batch_tools_nudge_after(workspace: Path | None) -> int:
    """
    连续单工具轮次达到阈值后，在 ToolMessage 注入并行提示。

    @param workspace 工作区根；None 时用默认
    @return 阈值；0=关闭
    """
    if workspace is None:
        return DEFAULT_BATCH_TOOLS_NUDGE_AFTER
    cfg = load_agent_config(workspace)
    agent = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    return parse_batch_tools_nudge_after(agent.get("batch_tools_nudge_after"))


def parse_identical_tool_guard(
    raw: object,
    *,
    default: bool = DEFAULT_IDENTICAL_TOOL_GUARD,
) -> bool:
    """
    解析 identical_tool_guard。

    @param raw agent.json 原始值
    @param default 缺省
    @return 是否启用短路径拦截
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return default


def resolve_identical_tool_guard(workspace: Path | None) -> bool:
    """
    是否拦截本问内相同参数的重复工具（对齐 Cursor / Claude Code 工具循环治理）。

    @param workspace 工作区根；None 时用默认
    @return 是否启用
    """
    if workspace is None:
        return DEFAULT_IDENTICAL_TOOL_GUARD
    cfg = load_agent_config(workspace)
    agent = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    return parse_identical_tool_guard(agent.get("identical_tool_guard"))
