"""llgraph 交互与会话内命令帮助文案。"""

from __future__ import annotations

from pathlib import Path

from llgraph.context.context_session import ContextSession
from llgraph.display.trace_display import TRACE_MODE_LABELS, TraceSession
from llgraph.loaders.rules_loader import discover_rules
from llgraph.loaders.skills_loader import discover_skills
from llgraph.terminal.keys import HELP_SHORTCUT_LINES
from llgraph.terminal.install_extras import format_install_extras_report, suggest_pip_install
from llgraph.terminal.output import emit_report


def _print_compact_help(
    *,
    allow_write: bool,
    web_search_enabled: bool,
    trace_session: TraceSession,
    context_session: ContextSession | None,
    workspace: Path | None,
    mcp_summary: str,
) -> None:
    """打印精简版 /help（默认）。"""
    write_state = "可写" if allow_write else "只读"
    web_state = "开" if web_search_enabled else "关"
    trace_state = TRACE_MODE_LABELS[trace_session.mode]
    ctx = context_session or ContextSession()
    skills_on = ", ".join(ctx.active_skills) if ctx.active_skills else "无"
    rule_count = len(discover_rules(workspace)) if workspace else 0
    skill_count = len(discover_skills(workspace)) if workspace else 0
    mcp_line = mcp_summary if mcp_summary else "MCP: 未加载"
    deps_hint = suggest_pip_install()
    shortcut_lines = "\n".join(HELP_SHORTCUT_LINES)

    text = f"""
llgraph 交互帮助
================

【常用】
  /help              本帮助（完整说明: /help full）
  /help deps         可选 pip 依赖：已装/缺失与安装命令
  /paste, /p         多行粘贴（--- 或连按两次回车结束）
  /plan run [补充]   Agent 聊好后进 Plan（附带会话摘录）；/plan list | /plan switch <id>
  /session           Agent 会话（/session use|new|title|delete|current）
  /sessions          列出 Agent 会话
  exit               退出

【上下文】
  /context           上下文占用
  /tools             内置工具与 MCP
  /compress          压缩上下文
  /index             索引状态与构建

【调试】（子命令详见各命令本身）
  /trace             过程展示（all|steps|reply|none|step|token|stats）
  /log               向量检索日志（debug|info|tail|purge|file on|off）

【配置】
  /write on|off      只读/可写（-w 启动默认可写）
  /web on|off        Web 搜索
  /watch on|off      索引文件监听
  /model [名|reset|refresh]
  /config            agent.json 路径
  /survey off|on|status

【规则与技能】
  /rule [on|off <id>]
  /skill [<name>|clear]    输入 / 斜杠补全
  Thought 规范       .llgraph/thought/*.md

【变更】（需 -w）
  /review [主题]     /changes  /undo [path|all]

【其它】
  /commands          自定义 .llgraph/commands
{shortcut_lines}

【当前会话】
  写入 {write_state} | Web {web_state} | 展示 {trace_state}
  技能: {skills_on} | 规则 {rule_count} | 技能定义 {skill_count}
  {mcp_line}
  依赖: {deps_hint}

详情: /plan  /session  /trace  /log  |  完整帮助: /help full
"""
    emit_report(text.strip())


def _print_full_help(
    *,
    allow_write: bool,
    web_search_enabled: bool,
    trace_session: TraceSession,
    context_session: ContextSession | None,
    workspace: Path | None,
    mcp_summary: str,
) -> None:
    """打印完整版 /help full。"""
    write_state = "已开启" if allow_write else "未开启（只读）"
    web_state = "已启用" if web_search_enabled else "未启用（/web on）"
    trace_state = TRACE_MODE_LABELS[trace_session.mode]
    ctx = context_session or ContextSession()
    skills_on = ", ".join(ctx.active_skills) if ctx.active_skills else "（无，可用 /skill <name>）"
    rule_count = len(discover_rules(workspace)) if workspace else 0
    skill_count = len(discover_skills(workspace)) if workspace else 0
    mcp_line = mcp_summary if mcp_summary else "MCP: 未加载"
    deps_hint = suggest_pip_install()
    shortcut_lines = "\n".join(HELP_SHORTCUT_LINES)

    text = f"""
llgraph 完整帮助
================

【会话内命令】（不会发给模型）

  /help, help, ?     精简帮助；/help full 本页；/help deps 可选依赖
  /paste, /p         多行粘贴模式
  /plan              Plan 规划（/plan help）；Agent 内 /plan [目标] 进入
  /trace             过程展示说明（/trace all|steps|reply|none|step|token|stats）
  /log               向量检索日志与执行日志
  /context           上下文分项占用
  /tools             内置工具与 MCP
  /config            agent.json 配置路径
  /survey            followup 开关
  /sessions          列出 Agent 会话
  /session           会话切换与子命令（/session 查看详情）
  /model             模型列表与切换
  /write             只读/可写
  /watch             索引监听
  /web               Web 搜索（Tavily）
  /rule              规则列表与开关
  /skill             技能列表与启用
  /index             索引构建
  /compress          压缩上下文
  /review            代码评审
  /changes, /undo    会话内文件变更
  /commands          自定义命令
{shortcut_lines}

【当前会话】
  文件写入: {write_state}
  Web 搜索: {web_state}
  过程展示: {trace_state}
  已启用技能: {skills_on}
  工作区规则: {rule_count} 条 | 技能定义: {skill_count} 个
  {mcp_line}
  依赖补装: {deps_hint}（详情 /help deps）

【启动参数】
  llgraph -C <目录> -w --model <名> --trace all|steps|reply|none
  llgraph --list-sessions -C .   列出 Agent 会话
  llgraph plan --list-plans      列出 Plan 会话
  llgraph index / search         索引与检索调试

【Rule / Skill 目录】
  项目: .llgraph/rules/*.mdc 、.llgraph/skills/<name>/SKILL.md
  个人: ~/.llgraph/rules/ 、~/.llgraph/skills/
  llgraph --init-config -C .  |  llgraph --init-user-config

【会话恢复】
  ~/.llgraph/context/<工作区>/sessions/<thread_id>/messages.jsonl
  llgraph -C <工作区> --thread-id cli-xxx
  会话内: /session use <id>  |  /session new
  Plan: /plan switch plan-xxx  |  llgraph plan --thread-id plan-xxx
"""
    emit_report(text.strip())


def print_install_extras_help(*, missing_only: bool = False) -> None:
    """
    打印 /help deps：optional extras 状态与 pip 建议。

    @param missing_only 仅显示未安装项
    """
    emit_report(format_install_extras_report(missing_only=missing_only))


def print_interactive_help(
    *,
    allow_write: bool,
    web_search_enabled: bool = False,
    trace_session: TraceSession,
    context_session: ContextSession | None = None,
    workspace: Path | None = None,
    mcp_summary: str = "",
    full: bool = False,
) -> None:
    """
    打印交互模式下的 /help 说明。

    @param allow_write 当前会话是否允许写文件
    @param trace_session 过程展示配置
    @param full 是否输出完整帮助
    """
    kwargs = dict(
        allow_write=allow_write,
        web_search_enabled=web_search_enabled,
        trace_session=trace_session,
        context_session=context_session,
        workspace=workspace,
        mcp_summary=mcp_summary,
    )
    if full:
        _print_full_help(**kwargs)
    else:
        _print_compact_help(**kwargs)
