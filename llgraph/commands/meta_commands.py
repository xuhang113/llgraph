"""交互命令：/rule、/skill、/compress、/review、自定义 commands。"""

import re
import shlex
from pathlib import Path
from llgraph.core.agent_session import AgentSessionContext
from llgraph.loaders.commands_loader import format_commands_help, resolve_command
from llgraph.context.context_builder import format_rules_list
from llgraph.context.context_compressor import apply_compress_to_agent_state, format_compress_report
from llgraph.context.context_session import ContextSession
from llgraph.commands.review_command import run_review
from llgraph.loaders.rules_loader import discover_rules
from llgraph.session.session_edits import SessionEditTracker
from llgraph.loaders.skills_loader import discover_skills
from llgraph.display.trace_display import (
    TraceSession,
    _print_trace_usage,
    parse_trace_mode,
    print_trace_step_detail,
    print_trace_step_list,
    set_trace_step_tokens,
)
from llgraph.terminal.output import emit, emit_error, emit_ok, emit_report, emit_warn


def _print_trace_stats(
    workspace: Path,
    agent_session: AgentSessionContext | None,
) -> None:
    """
    打印 token 与落盘统计（P6）。

    @param workspace 工作区根
    @param agent_session 会话上下文
    """
    from llgraph.context.context_compressor import estimate_tokens
    from llgraph.context.context_spill import format_spill_stats

    messages_tokens = 0
    if agent_session is not None:
        config = {"configurable": {"thread_id": agent_session.thread_id}}
        try:
            state = agent_session.agent.get_state(config)
            messages = list((state.values or {}).get("messages") or [])
            messages_tokens = estimate_tokens(messages)
        except Exception:
            pass

    spill = agent_session.context_spill if agent_session else None
    spilled_bytes = spill.spilled_bytes_on_disk() if spill else 0
    spill_count = spill.spill_count() if spill else 0
    # 启发式：system + 首轮 user 约可缓存前缀
    cacheable = min(messages_tokens, int(messages_tokens * 0.15)) if messages_tokens else 0

    emit_report(
        format_spill_stats(
            messages_tokens=messages_tokens,
            spilled_bytes=spilled_bytes,
            spill_count=spill_count,
            cacheable_prefix_estimate=cacheable,
        )
    )
    spill_dir = workspace / ".llgraph" / "context" / "tool-results"
    emit(f"落盘目录: {spill_dir}", colorize=True)
    try:
        from llgraph.display.execution_log import (
            execution_log_path,
            extract_usage_from_messages,
            read_execution_tail,
        )
        from llgraph.core.llm_settings import resolve_effective_model

        emit(f"执行日志: {execution_log_path(workspace)}", colorize=True)
        emit(f"当前模型: {resolve_effective_model(workspace)}", colorize=True)
        if agent_session is not None:
            config = {"configurable": {"thread_id": agent_session.thread_id}}
            try:
                state = agent_session.agent.get_state(config)
                messages = list((state.values or {}).get("messages") or [])
                usage = extract_usage_from_messages(messages)
                totals = usage.get("totals") or {}
                if totals:
                    emit(
                        "网关 usage 累计: "
                        f"in={totals.get('input_tokens', 0)} "
                        f"out={totals.get('output_tokens', 0)} "
                        f"cache_read={totals.get('cache_read_input_tokens', 0)}",
                        colorize=True,
                    )
            except Exception:
                pass
        last = read_execution_tail(workspace, limit=1)
        if last:
            from llgraph.display.execution_log import format_execution_record

            emit(f"最近一轮: {format_execution_record(last[0])}", colorize=True)
    except Exception:
        pass
    emit("更多: /log tail | /log purge", colorize=True)
# 内置元命令首 token（不含 /）；新增命令须同步更新 handle_meta_command 与此集合
_BUILTIN_META_COMMAND_NAMES = frozenset({
    "index",
    "survey",
    "paste",
    "p",
    "watch",
    "write",
    "web",
    "model",
    "config",
    "session",
    "sessionid",
    "session-id",
    "sessions",
    "plan",
    "help",
    "h",
    "compress",
    "context",
    "tools",
    "review",
    "commands",
    "trace",
    "log",
    "rule",
    "skill",
    "changes",
    "undo",
    "diff",
})


def _meta_command_token(text: str) -> str | None:
    """
    解析首 token 元命令名（不含 /）。

    @param text 用户输入
    @return 命令名小写；非 /command 形式则 None
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return None
    if not parts or not parts[0].startswith("/"):
        return None
    name = parts[0][1:]
    if not name or "/" in name:
        return None
    return name.lower()


def is_registered_meta_command(text: str, workspace: Path) -> bool:
    """
    输入是否为已注册元命令（仅此类不发给 Agent）。

    @param text 用户输入
    @param workspace 工作区根（查 .llgraph/commands 自定义命令）
    @return 是否走 handle_meta_command
    """
    token = _meta_command_token(text)
    if token is None:
        return False
    if token in _BUILTIN_META_COMMAND_NAMES:
        return True
    return resolve_command(workspace, token) is not None


def resolve_meta_display_mode(text: str, workspace: Path) -> str:
    """
    Web UI 元命令结果展示方式。

    @param text 用户输入
    @param workspace 工作区根
    @return modal：弹窗展示，不写入对话区；agent：已注入 Agent 会话，应刷新历史
    """
    token = _meta_command_token(text)
    if token is None:
        return "modal"
    if token in _BUILTIN_META_COMMAND_NAMES:
        return "modal"
    cmd = resolve_command(workspace, token)
    if cmd is not None and cmd.handler == "prompt":
        return "agent"
    return "modal"


def _handle_index_meta_command(line: str, workspace: Path) -> bool:
    """
    处理 /index 及子命令（在交互会话内构建索引）。

    @param line 用户输入
    @param workspace 当前会话工作区
    @return 恒为 True（已消费命令）
    """
    from llgraph.code_index.index_dispatch import (
        dispatch_index,
        print_index_interactive_help,
        print_index_status,
    )

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        emit(f"无法解析 /index 参数: {exc}", colorize=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/index", "index"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens:
        print_index_status(workspace)
        emit("", colorize=True)
        print_index_interactive_help()
        return True

    if tokens[0].lower() in ("help", "?"):
        print_index_interactive_help()
        return True

    normalized: list[str] = []
    for token in tokens:
        low = token.lower()
        if low in ("inc",):
            normalized.append("incremental")
        elif low in ("run",):
            normalized.append("full")
        else:
            normalized.append(token)

    dispatch_index(workspace, normalized, prog="/index", bare_means_status=False)
    return True


def _effective_allow_write(
    agent_session: AgentSessionContext | None,
    allow_write: bool,
) -> bool:
    """
    解析当前是否可写（优先 agent_session）。

    @param agent_session Agent 会话
    @param allow_write 调用方传入的默认值
    @return 是否可写
    """
    if agent_session is not None:
        return agent_session.allow_write
    return allow_write


def _effective_edit_tracker(
    agent_session: AgentSessionContext | None,
    edit_tracker: SessionEditTracker | None,
) -> SessionEditTracker | None:
    """
    解析当前编辑账本（优先 agent_session）。

    @param agent_session Agent 会话
    @param edit_tracker 调用方传入的账本
    @return 编辑账本
    """
    if agent_session is not None and agent_session.edit_tracker is not None:
        return agent_session.edit_tracker
    return edit_tracker


def _handle_survey_command(
    line: str,
    workspace: Path,
    agent_session: AgentSessionContext | None,
    context_session: ContextSession | None = None,
) -> bool:
    """
    处理 /survey：off | on | status（无固定问卷向导）。

    @param line 用户输入
    @param workspace 工作区根
    @param agent_session Agent 会话
    @param context_session Rule/Skill 会话状态
    @return 恒为 True
    """
    from llgraph.config.survey_settings import (
        format_survey_status,
        survey_interactive_enabled,
    )

    stripped = line.strip()
    parts = stripped.split(maxsplit=1)
    sub = parts[1].strip().lower() if len(parts) > 1 else ""

    if sub in ("off", "disable", "0"):
        if context_session is not None:
            context_session.survey_enabled = False
        if agent_session is not None:
            from llgraph.core.agent import rebuild_agent_preserving_memory

            rebuild_agent_preserving_memory(
                agent_session,
                allow_write=agent_session.allow_write,
            )
        emit("Survey 交互已关闭（本会话）。", colorize=True)
        return True
    if sub in ("on", "enable", "1"):
        if context_session is not None:
            context_session.survey_enabled = True
        if agent_session is not None:
            from llgraph.core.agent import rebuild_agent_preserving_memory

            rebuild_agent_preserving_memory(
                agent_session,
                allow_write=agent_session.allow_write,
            )
        emit("Survey 交互已开启（本会话）。", colorize=True)
        return True
    if sub in ("status", "?"):
        emit_report(format_survey_status(workspace, context_session))
        return True

    emit_warn(
        "无固定问卷。确认须由 Agent 在回复末尾输出 <<<llgraph-survey>>> JSON；"
        "可用 /survey on|off|status 控制 followup 渲染。"
    )
    emit_report(format_survey_status(workspace, context_session))
    return True


def _handle_watch_command(
    line: str,
    workspace: Path,
    agent_session: AgentSessionContext | None,
) -> bool:
    """
    处理 /watch on|off|status。

    @param line 用户输入
    @param workspace 工作区根
    @param agent_session Agent 会话
    @return 恒为 True
    """
    from llgraph.code_index.index_watch import ensure_index_watch, format_watch_status

    if agent_session is None:
        emit("仅交互模式支持 /watch。", colorize=True)
        return True

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        emit(f"无法解析 /watch 参数: {exc}", colorize=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/watch", "watch"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens or tokens[0].lower() in ("status", "?"):
        emit_report(format_watch_status(agent_session.watch_service, workspace))
        return True

    sub = tokens[0].lower()
    if sub in ("on", "true", "1", "start", "enable"):
        was_active = (
            agent_session.watch_service is not None
            and agent_session.watch_service.active
        )
        service, err = ensure_index_watch(workspace, agent_session.watch_service)
        agent_session.watch_service = service
        if err:
            emit(err, colorize=True)
        elif was_active:
            emit("index watch 已在运行。", colorize=True)
        else:
            emit("已启动 index watch（保存文件后自动增量索引）。", colorize=True)
        return True

    if sub in ("off", "false", "0", "stop", "disable"):
        service = agent_session.watch_service
        if service is None or not service.active:
            emit("index watch 未运行。", colorize=True)
            return True
        service.stop()
        emit("已停止 index watch。", colorize=True)
        return True

    emit(
        "用法: /watch  |  /watch status  |  /watch on  |  /watch off",
        colorize=True,
    )
    return True


def _handle_write_command(
    line: str,
    agent_session: AgentSessionContext | None,
    context_session: ContextSession,
) -> bool:
    """
    处理 /write on|off。

    @param line 用户输入
    @param agent_session Agent 会话
    @param context_session Rule/Skill 状态
    @return 恒为 True
    """
    from llgraph.session.session_write_mode import format_write_mode_status, set_session_write_mode

    if agent_session is None:
        emit("仅交互模式支持 /write。", colorize=True)
        return True

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        emit(f"无法解析 /write 参数: {exc}", colorize=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/write", "write"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens:
        emit_report(format_write_mode_status(agent_session))
        return True

    sub = tokens[0].lower()
    if sub in ("on", "true", "1", "enable", "w"):
        if set_session_write_mode(agent_session, enabled=True, context_session=context_session):
            emit("已切换为可写模式（写工具与受限 shell 已启用，会话历史已保留）。", colorize=True)
        else:
            emit("当前已是可写模式。", colorize=True)
        return True

    if sub in ("off", "false", "0", "disable", "ro", "readonly"):
        if set_session_write_mode(agent_session, enabled=False, context_session=context_session):
            emit("已切换为只读模式（禁止 Agent 写文件；/changes · /undo 仍可用）。", colorize=True)
        else:
            emit("当前已是只读模式。", colorize=True)
        return True

    emit(
        "用法: /write  |  /write on  |  /write off\n"
        "说明: 切换后重建 Agent，对话历史保留；等价于启动时 -w 开关。",
        colorize=True,
    )
    return True


def _handle_web_command(
    line: str,
    agent_session: AgentSessionContext | None,
) -> bool:
    """
    处理 /web on|off|status。

    @param line 用户输入
    @param agent_session Agent 会话
    @return 恒为 True
    """
    from llgraph.session.session_web_search import (
        format_web_search_status,
        set_session_web_search_mode,
    )

    if agent_session is None:
        emit("仅交互模式支持 /web。", colorize=True)
        return True

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        emit(f"无法解析 /web 参数: {exc}", colorize=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/web", "web"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens or tokens[0].lower() in ("status", "?"):
        emit_report(format_web_search_status(agent_session))
        return True

    sub = tokens[0].lower()
    if sub in ("on", "true", "1", "start", "enable"):
        changed, msg = set_session_web_search_mode(agent_session, enabled=True)
        emit(msg, colorize=True)
        if not changed and "未配置" in msg:
            pass
        return True

    if sub in ("off", "false", "0", "stop", "disable"):
        changed, msg = set_session_web_search_mode(agent_session, enabled=False)
        emit(msg, colorize=True)
        return True

    emit(
        "用法: /web  |  /web status  |  /web on  |  /web off\n"
        "说明: 启用后注册 web_search（Tavily）；切换后重建 Agent，对话历史保留。\n"
        "Skill/Rule: 仅注入描述+路径，正文请 read_file；锚点见 <session-manifest>。",
        colorize=True,
    )
    return True


def _handle_model_command(
    line: str,
    workspace: Path,
    agent_session: AgentSessionContext | None,
    allow_write: bool,
) -> bool:
    """
    处理 /model 列表与切换。

    @param line 用户输入
    @param workspace 工作区根
    @param agent_session Agent 会话
    @param allow_write 是否可写
    @return 恒为 True
    """
    from llgraph.core.agent import rebuild_agent_preserving_memory
    from llgraph.core.gateway_models import format_models_list, is_catalog_model
    from llgraph.core.llm_settings import (
        format_model_status,
        resolve_effective_model,
        set_runtime_model,
    )

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        emit(f"无法解析 /model 参数: {exc}", colorize=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/model", "model"):
        tokens = parts[1:]
    else:
        tokens = parts

    current = resolve_effective_model(workspace)
    write_mode = _effective_allow_write(agent_session, allow_write)

    if not tokens or tokens[0].lower() in ("list", "ls", "?"):
        emit_report(format_model_status(workspace))
        emit("", colorize=True)
        try:
            emit_report(format_models_list(workspace, current=current))
        except RuntimeError as exc:
            emit(f"拉取网关模型列表失败: {exc}", colorize=True)
            emit("仍可用 /model <模型名> 手动切换。", colorize=True)
        return True

    if tokens[0].lower() == "refresh":
        try:
            emit_report(
                format_models_list(workspace, current=current, force_refresh=True)
            )
        except RuntimeError as exc:
            emit(f"刷新失败: {exc}", colorize=True)
        return True

    if tokens[0].lower() in ("reset", "default", "env"):
        set_runtime_model(None)
        if agent_session is not None:
            rebuild_agent_preserving_memory(
                agent_session,
                allow_write=write_mode,
                mcp_tools=agent_session.mcp_tools,
                on_file_changed=agent_session.on_file_changed if write_mode else None,
            )
        restored = resolve_effective_model(workspace)
        emit(f"已恢复默认模型: {restored}", colorize=True)
        return True

    new_model = tokens[0].strip()
    if not new_model:
        emit("用法: /model <模型名>  |  /model list  |  /model reset", colorize=True)
        return True

    set_runtime_model(new_model)
    if not is_catalog_model(workspace, new_model):
        emit_warn(
            f"提示: {new_model!r} 不在 agent.json 模型目录（llm.models）内，"
            "若调用失败请 /model list 换模型。"
        )
    if agent_session is not None:
        rebuild_agent_preserving_memory(
            agent_session,
            allow_write=write_mode,
            mcp_tools=agent_session.mcp_tools,
            on_file_changed=agent_session.on_file_changed if write_mode else None,
        )
    emit(f"已切换模型: {new_model}（下一条消息起生效，会话历史已保留）", colorize=True)
    return True


def _persist_agent_and_enter_plan(
    agent_session: AgentSessionContext,
    workspace: Path,
    *,
    opening_goal: str,
) -> None:
    """
    落盘当前 Agent 对话并请求切换到 Plan（附带 source_agent_thread_id）。

    @param agent_session Agent 会话
    @param workspace 工作区根
    @param opening_goal 计划目标（可简短；细节在 Agent 摘录中）
    """
    from llgraph.session.session_file_store import persist_agent_session
    from llgraph.session.mode_switch import SessionModeTransition

    persist_agent_session(agent_session.agent, workspace, agent_session.thread_id)
    agent_session.mode_switch = SessionModeTransition(
        mode="plan",
        thread_id=None,
        opening_goal=opening_goal,
        from_thread_id=agent_session.thread_id,
    )


def _handle_plan_command(
    stripped: str,
    workspace: Path,
    agent_session: AgentSessionContext | None,
) -> bool:
    """
    Agent 模式 /plan 入口：进入 Plan、列举、切换；Plan 内子命令提示先进入 Plan。

    @param stripped 原始输入
    @param workspace 工作区根
    @param agent_session Agent 会话
    @return 是否已处理
    """
    if agent_session is None:
        emit("需要交互会话才能使用 /plan。", colorize=True)
        return True

    from llgraph.plan.meta_commands import _PLAN_SUBCOMMANDS, _parse_plan_command
    from llgraph.plan.help_text import print_plan_help
    from llgraph.plan.plan_registry import format_plans_list
    from llgraph.session.mode_switch import SessionModeTransition

    sub, arg = _parse_plan_command(stripped)
    if sub is None:
        return False

    if sub == "help":
        print_plan_help(in_plan_mode=False)
        return True

    if sub == "list":
        emit_report(format_plans_list(workspace, current_thread_id=None))
        return True

    if sub == "switch":
        tid = arg.strip()
        if not tid:
            emit_warn("用法: /plan switch plan-xxxxxxxx")
            return True
        if not tid.startswith("plan-"):
            emit_warn("Plan 会话 thread_id 须以 plan- 开头")
            return True
        agent_session.mode_switch = SessionModeTransition(
            mode="plan",
            thread_id=tid,
            from_thread_id=agent_session.thread_id,
        )
        emit_ok(f"正在切换到 Plan 模式（{tid}）…")
        return True

    if sub == "run":
        goal = arg.strip() or "根据上文 Agent 对话已讨论的内容制定并执行计划。"
        _persist_agent_and_enter_plan(agent_session, workspace, opening_goal=goal)
        emit_ok("正在进入 Plan 模式（已附带当前 Agent 会话摘录）…")
        return True

    if sub == "results":
        from llgraph.plan.plan_results import format_plan_results

        parts = arg.split() if arg else []
        query = ""
        task_id = ""
        for part in parts:
            if part.startswith("plan-") or (len(part) == 8 and all(c in "0123456789abcdef" for c in part.lower())):
                query = part
            elif part.startswith("w"):
                task_id = part
        emit_report(
            format_plan_results(
                workspace,
                query=query,
                task_id=task_id,
            )
        )
        return True

    plan_only = frozenset({
        "graph",
        "status",
        "confirm",
        "revise",
        "cancel",
        "handoff",
    })
    if sub in plan_only:
        emit_warn(
            f"/plan {sub} 需在 Plan 模式内使用。"
            " 输入 /plan [目标] 或 /plan switch <id> 进入 Plan。"
        )
        return True

    if sub not in _PLAN_SUBCOMMANDS:
        goal = stripped.split(None, 1)[1].strip() if len(stripped.split(None, 1)) > 1 else ""
        if goal:
            _persist_agent_and_enter_plan(agent_session, workspace, opening_goal=goal)
            emit_ok("正在进入 Plan 模式（已附带当前 Agent 会话摘录）…")
            return True
        print_plan_help(in_plan_mode=False)
        return True

    print_plan_help(in_plan_mode=False)
    return True


def _handle_session_command(
    stripped: str,
    workspace: Path,
    agent_session: AgentSessionContext | None,
) -> bool:
    """
    /session 子命令：列举、切换、新建 thread_id。

    @param stripped 原始输入
    @param workspace 工作区根
    @param agent_session Agent 会话
    @return 是否已处理
    """
    if agent_session is None:
        emit("需要交互会话才能切换 thread_id。", colorize=True)
        return True

    from llgraph.session.session_switch import (
        create_new_thread_id,
        format_session_command_help,
        print_current_session_info,
        switch_agent_thread,
    )

    parts = stripped.split(maxsplit=2)
    sub = parts[1].lower() if len(parts) >= 2 else ""

    def _no_spill() -> bool:
        spill = agent_session.context_spill
        return spill is not None and bool(getattr(spill, "disabled", False))

    if not sub or sub in ("list", "ls"):
        emit_report(
            format_session_command_help(workspace, agent_session.thread_id)
        )
        return True

    if sub in ("current", "id", "session-id", "sessionid"):
        print_current_session_info(workspace, agent_session.thread_id)
        return True

    if sub == "new":
        from llgraph.session.session_meta import default_session_title, set_session_title

        new_id = create_new_thread_id()
        _, msg = switch_agent_thread(
            agent_session, new_id, no_spill=_no_spill()
        )
        emit(msg, colorize=True)
        return True

    if sub in ("use", "switch", "resume") and len(parts) >= 3:
        tid = parts[2].strip()
        _, msg = switch_agent_thread(
            agent_session, tid, no_spill=_no_spill()
        )
        emit(msg, colorize=True)
        return True

    if sub == "plan":
        from llgraph.session.mode_switch import SessionModeTransition, parse_session_mode_command

        mode, plan_tid, goal = parse_session_mode_command(stripped)
        if mode == "plan_switch_removed":
            emit_warn(
                "切换 Plan 请用 /plan switch <plan-id>；"
                "/session plan 仅用于进入 Plan（/session plan [目标]）。"
            )
            return True
        if mode != "plan":
            emit("用法: /session plan [目标说明]  |  切换 Plan: /plan switch plan-xxx", colorize=True)
            return True
        if agent_session is not None:
            from llgraph.session.session_file_store import persist_agent_session

            persist_agent_session(
                agent_session.agent, workspace, agent_session.thread_id
            )
        agent_session.mode_switch = SessionModeTransition(
            mode="plan",
            thread_id=None,
            opening_goal=goal,
            from_thread_id=agent_session.thread_id,
        )
        if goal:
            emit("正在切换到 Plan 模式（已附带 Agent 会话摘录）…", colorize=True)
        else:
            emit("正在切换到 Plan 模式（新建 Plan 会话）…", colorize=True)
        return True

    if sub == "agent":
        emit("当前已在 Agent 模式。", colorize=True)
        return True

    if sub in ("title", "rename", "name"):
        from llgraph.session.session_meta import get_session_title, set_session_title

        m = re.match(
            r"(?i)^(?:/session|session)\s+(?:title|rename|name)\s+"
            r"(?:(cli-[0-9a-f]{8})\s+)?(.+)$",
            stripped.strip(),
        )
        if not m:
            emit(
                "用法: /session title <新标题>\n"
                "      /session title <thread_id> <新标题>",
                colorize=True,
            )
            return True
        tid = m.group(1) or agent_session.thread_id
        title_text = m.group(2).strip()
        ok, msg = set_session_title(workspace, tid, title_text, source="manual")
        emit(msg, colorize=True)
        if ok and tid == agent_session.thread_id:
            from llgraph.session.session_meta import resolve_session_display_title

            emit(
                f"当前展示: {resolve_session_display_title(workspace, tid)}",
                colorize=True,
            )
        return True

    if sub in ("delete", "del", "rm"):
        from llgraph.session.session_delete import (
            delete_sessions,
            delete_workspace_session,
            format_delete_report,
            is_plan_main_thread,
            validate_thread_id,
        )
        from llgraph.session.session_meta import default_session_title, set_session_title
        from llgraph.session.session_registry import (
            list_empty_session_ids,
            list_workspace_session_ids,
        )

        tokens = stripped.split()
        flags = {t.lower() for t in tokens if t.startswith("--")}
        if len(tokens) < 3:
            emit(
                "用法:\n"
                "  /session delete <thread_id>           删除指定会话\n"
                "  /session delete <thread_id> --force   删除当前会话\n"
                "  /session delete empty                 删除空壳会话\n"
                "  /session delete all                   删除除当前外全部\n"
                "  /session delete all --including-current  全量删除并切到新会话",
                colorize=True,
            )
            return True

        target = tokens[2]
        if target.lower() == "empty":
            empty_ids = list_empty_session_ids(workspace)
            if not empty_ids:
                emit("（无空壳会话）", colorize=True)
                return True
            report = delete_sessions(workspace, empty_ids)
            emit_report(format_delete_report(report))
            return True

        if target.lower() == "all":
            including_current = (
                "--including-current" in flags or "--all" in flags
            )
            ids = list_workspace_session_ids(workspace)
            if not ids:
                emit("（无可删除会话）", colorize=True)
                return True
            if not including_current:
                ids = [i for i in ids if i != agent_session.thread_id]
            if not ids:
                emit_warn(
                    "除当前外无其它会话；全量删除请加: /session delete all --including-current"
                )
                return True
            deleted_current = agent_session.thread_id in ids
            report = delete_sessions(workspace, ids)
            emit_report(format_delete_report(report))
            if deleted_current and report.success_count > 0:
                new_id = create_new_thread_id()
                set_session_title(
                    workspace, new_id, default_session_title(new_id), source="fallback"
                )
                _, msg = switch_agent_thread(
                    agent_session, new_id, no_spill=_no_spill()
                )
                emit(msg, colorize=True)
            return True

        try:
            tid = validate_thread_id(target)
        except ValueError as exc:
            emit_error(str(exc))
            return True

        if tid == agent_session.thread_id and "--force" not in flags:
            emit_warn(
                "不能删除当前会话。请先 /session new，或:\n"
                "  /session delete "
                + tid
                + " --force"
            )
            return True

        result = delete_workspace_session(workspace, tid)
        if result.ok:
            label = "Plan 及子节点" if is_plan_main_thread(tid) else "会话"
            emit(f"已删除{label} {tid}。", colorize=True)
            for path in result.removed_paths:
                emit(f"  - {path}", colorize=True)
            for path in result.related_removed:
                emit(f"  - {path}", colorize=True)
            if not result.removed_paths and not result.related_removed:
                emit("  （该会话无落盘文件或已不存在）", colorize=True)
        else:
            emit(f"删除失败 {tid}: {result.error}", colorize=True)
            return True

        if tid == agent_session.thread_id:
            new_id = create_new_thread_id()
            set_session_title(
                workspace, new_id, default_session_title(new_id), source="fallback"
            )
            _, msg = switch_agent_thread(
                agent_session, new_id, no_spill=_no_spill()
            )
            emit(msg, colorize=True)
        return True

    emit_report(
        format_session_command_help(workspace, agent_session.thread_id)
    )
    return True


def _print_skills_usage(workspace: Path, session: ContextSession) -> None:
    from llgraph.config.catalog_paths import scope_label

    skills = discover_skills(workspace)
    emit(
        "技能（Skills）— 项目: .llgraph/skills/<name>/SKILL.md  |  个人: ~/.llgraph/skills/",
        colorize=True,
    )
    if not skills:
        emit(
            "  （未找到技能；llgraph --init-config / --init-user-config）",
            colorize=True,
        )
    else:
        for skill in skills:
            active = "✓" if skill.name.lower() in [s.lower() for s in session.active_skills] else " "
            origin = scope_label(skill.scope)
            emit(f"  [{active}] {skill.name} [{origin}] — {skill.description}", colorize=True)
    active = ", ".join(session.active_skills) if session.active_skills else "（无）"
    emit(f"当前 /skill 置顶: {active}", colorize=True)
    emit("", colorize=True)
    emit("命令:", colorize=True)
    emit("  /skill              列出技能", colorize=True)
    emit("  /skill <name>       置顶技能（manifest ⭐，正文 read_file）", colorize=True)
    emit("  /skill off <name>   关闭指定技能", colorize=True)
    emit("  /skill clear        清空已置顶技能", colorize=True)
    emit("  说明: 全量目录在 <session-manifest>；不自动匹配、不注入 SKILL 正文", colorize=True)


def handle_meta_command(
    line: str,
    *,
    workspace: Path,
    trace_session: TraceSession,
    context_session: ContextSession,
    allow_write: bool = False,
    last_user_message: str = "",
    edit_tracker: SessionEditTracker | None = None,
    agent_session: AgentSessionContext | None = None,
    mcp_summary: str = "",
) -> bool:
    """
    处理 /help、/trace、/rule、/skill 等元命令。

    @param line 用户输入
    @param workspace 工作区根
    @param trace_session 过程展示配置
    @param context_session Rule/Skill 会话状态
    @param edit_tracker 会话编辑账本
    @param agent_session Agent 会话（自定义命令 / prompt 用）
    @param mcp_summary MCP 加载摘要（/help 用）
    @return True 表示已消费，不发给 Agent
    """
    from llgraph.commands.help_text import print_interactive_help

    allow_write = _effective_allow_write(agent_session, allow_write)
    edit_tracker = _effective_edit_tracker(agent_session, edit_tracker)

    stripped = line.strip()
    lower = stripped.lower()

    if lower == "/index" or lower.startswith("/index ") or lower == "index" or lower.startswith(
        "index "
    ):
        return _handle_index_meta_command(stripped, workspace)

    if lower == "/survey" or lower == "survey" or lower.startswith("/survey "):
        return _handle_survey_command(
            stripped, workspace, agent_session, context_session
        )

    if lower in ("/paste", "/p", "paste"):
        emit(
            "请直接输入 /paste 后回车进入粘贴模式，或:\n"
            "  1. 输入 /paste 回车\n"
            "  2. 粘贴完整报错/日志\n"
            "  3. 单独一行输入 --- 结束",
            colorize=True,
        )
        return True

    if lower == "/watch" or lower == "watch":
        return _handle_watch_command(stripped, workspace, agent_session)

    if lower.startswith("/watch ") or lower.startswith("watch "):
        return _handle_watch_command(stripped, workspace, agent_session)

    if lower == "/write" or lower == "write":
        return _handle_write_command(stripped, agent_session, context_session)

    if lower.startswith("/write ") or lower.startswith("write "):
        return _handle_write_command(stripped, agent_session, context_session)

    if lower == "/web" or lower == "web":
        return _handle_web_command(stripped, agent_session)

    if lower.startswith("/web ") or lower.startswith("web "):
        return _handle_web_command(stripped, agent_session)

    if lower == "/model" or lower == "model":
        return _handle_model_command(stripped, workspace, agent_session, allow_write)

    if lower.startswith("/model ") or lower.startswith("model "):
        return _handle_model_command(stripped, workspace, agent_session, allow_write)

    if lower == "/config" or lower == "config":
        from llgraph.core.agent_config import format_agent_config_sources

        emit_report(format_agent_config_sources(workspace))
        return True

    if lower == "/session" or lower == "session":
        return _handle_session_command(stripped, workspace, agent_session)

    if lower.startswith("/session ") or lower.startswith("session "):
        return _handle_session_command(stripped, workspace, agent_session)

    if lower == "/plan" or lower == "plan":
        return _handle_plan_command(stripped, workspace, agent_session)

    if lower.startswith("/plan ") or lower.startswith("plan "):
        return _handle_plan_command(stripped, workspace, agent_session)

    if lower in ("/sessionid", "sessionid", "/session-id", "session-id"):
        if agent_session is None:
            emit("需要交互会话才能查看 thread_id。", colorize=True)
            return True
        from llgraph.session.session_switch import print_current_session_info

        print_current_session_info(workspace, agent_session.thread_id)
        return True

    if lower in ("/sessions", "sessions"):
        from llgraph.session.session_registry import format_sessions_list

        current = agent_session.thread_id if agent_session is not None else None
        emit_report(format_sessions_list(workspace, current_thread_id=current))
        return True

    if lower in ("/help", "help", "?", "/h", "h"):
        web_enabled = (
            agent_session.web_search_enabled if agent_session is not None else False
        )
        print_interactive_help(
            allow_write=allow_write,
            web_search_enabled=web_enabled,
            trace_session=trace_session,
            context_session=context_session,
            workspace=workspace,
            mcp_summary=mcp_summary,
        )
        return True

    if lower in ("/help full", "help full"):
        web_enabled = (
            agent_session.web_search_enabled if agent_session is not None else False
        )
        print_interactive_help(
            allow_write=allow_write,
            web_search_enabled=web_enabled,
            trace_session=trace_session,
            context_session=context_session,
            workspace=workspace,
            mcp_summary=mcp_summary,
            full=True,
        )
        return True

    if lower in (
        "/help deps",
        "help deps",
        "/help install",
        "help install",
        "/deps",
        "deps",
    ):
        from llgraph.commands.help_text import print_install_extras_help

        missing_only = "missing" in lower or "缺" in stripped
        print_install_extras_help(missing_only=missing_only)
        return True

    if lower in ("/help plan", "help plan", "/plan help"):
        from llgraph.plan.help_text import print_plan_help

        print_plan_help(in_plan_mode=False)
        return True

    if lower in ("/compress", "compress") or lower.startswith("/compress "):
        return _handle_compress(workspace, agent_session)

    if lower in ("/context", "context"):
        from llgraph.context.context_stats import print_context_usage

        print_context_usage(
            workspace,
            context_session=context_session,
            last_user_message=last_user_message,
            allow_write=allow_write,
            agent_session=agent_session,
        )
        return True

    if lower in ("/tools", "tools"):
        from llgraph.core.tool_list import format_agent_tools_report

        web_enabled = (
            agent_session.web_search_enabled if agent_session is not None else False
        )
        mcp = agent_session.mcp_tools if agent_session is not None else None
        emit_report(
            format_agent_tools_report(
                workspace,
                allow_write=allow_write,
                web_search_enabled=web_enabled,
                mcp_tools=mcp,
            )
        )
        return True

    if lower == "/review" or lower.startswith("/review "):
        return _handle_review(stripped, workspace, edit_tracker, last_user_message)

    if lower == "/commands" or lower == "commands":
        emit_report(format_commands_help(workspace))
        return True

    if lower == "/trace" or lower == "trace":
        _print_trace_usage(trace_session)
        return True

    if lower == "/log" or lower == "log":
        from llgraph.config.logging_settings import format_log_status

        emit_report(format_log_status(workspace))
        return True

    if lower.startswith("/log ") or lower.startswith("log "):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            from llgraph.config.logging_settings import format_log_status

            emit_report(format_log_status(workspace))
            return True
        sub = parts[1].strip().lower()
        if sub in ("tail", "exec", "execution"):
            from llgraph.display.execution_log import format_execution_tail

            emit_report(format_execution_tail(workspace))
            return True
        if sub == "purge":
            from llgraph.display.log_retention import format_purge_report, run_log_retention

            report = run_log_retention(workspace, quiet=False)
            emit_report(format_purge_report(report))
            return True
        if sub in ("file on", "file off"):
            from llgraph.config.logging_settings import (
                level_name,
                resolve_log_level,
                set_runtime_log_level,
            )

            file_on = sub.endswith("on")
            level = resolve_log_level(workspace)
            set_runtime_log_level(
                workspace,
                level_name(level),
                search_file=file_on,
            )
            emit(
                f"向量检索落盘 search.log: {'开' if file_on else '关'}",
                colorize=True,
            )
            return True
        from llgraph.config.logging_settings import level_name, set_runtime_log_level

        try:
            effective = set_runtime_log_level(workspace, parts[1].strip())
        except Exception:
            emit_warn(
                f"未知级别: {parts[1]!r}，可用 debug | info | warning | error"
            )
            return True
        emit(f"已切换向量检索日志: {level_name(effective)}", colorize=True)
        return True

    if lower.startswith("/trace ") or lower.startswith("trace "):
        parts = stripped.split()
        if len(parts) < 2:
            _print_trace_usage(trace_session)
            return True
        sub = parts[1].strip().lower()
        if sub in ("rich",):
            from llgraph.display.trace_display import set_trace_rich_render
            from llgraph.terminal.markdown_render import _rich_import_ok

            if not _rich_import_ok():
                emit_warn("未安装 rich，请 pip install 'llgraph[terminal]' 或 pip install rich")
                return True
            if len(parts) >= 3:
                arg = parts[2].strip().lower()
                if arg in ("on", "off"):
                    enabled = set_trace_rich_render(trace_session, arg == "on")
                    emit(
                        f"Rich 终端渲染: {'开' if enabled else '关'}",
                        colorize=True,
                    )
                    return True
            enabled = set_trace_rich_render(trace_session, None)
            emit(
                f"Rich 终端渲染: {'开' if enabled else '关'}",
                colorize=True,
            )
            return True
        if sub in ("md", "markdown", "render"):
            from llgraph.display.trace_display import set_trace_render_markdown

            if len(parts) >= 3:
                arg = parts[2].strip().lower()
                if arg in ("on", "off"):
                    enabled = set_trace_render_markdown(trace_session, arg == "on")
                    emit(
                        f"Markdown 终端渲染: {'开' if enabled else '关'}",
                        colorize=True,
                    )
                    return True
            enabled = set_trace_render_markdown(trace_session, None)
            emit(
                f"Markdown 终端渲染: {'开' if enabled else '关'}",
                colorize=True,
            )
            return True
        if sub in ("stats",):
            _print_trace_stats(workspace, agent_session)
            return True
        if sub in ("token", "tokens"):
            if len(parts) >= 3:
                arg = parts[2].strip().lower()
                if arg in ("on", "off"):
                    enabled = set_trace_step_tokens(trace_session, arg == "on")
                    emit(
                        f"步骤 token 显示: {'开' if enabled else '关'}",
                        colorize=True,
                    )
                    return True
                if arg in ("stats",):
                    _print_trace_stats(workspace, agent_session)
                    return True
            enabled = set_trace_step_tokens(trace_session, None)
            emit(
                f"步骤 token 显示: {'开' if enabled else '关'}",
                colorize=True,
            )
            return True
        if sub in ("step", "expand"):
            if len(parts) >= 3:
                print_trace_step_detail(trace_session, parts[2])
            else:
                print_trace_step_list(trace_session)
            return True
        if sub == "context":
            from llgraph.context.context_stats import print_context_usage

            print_context_usage(
                workspace,
                context_session=context_session,
                last_user_message=last_user_message,
                allow_write=allow_write,
                agent_session=agent_session,
            )
            return True
        mode_arg = parts[1] if len(parts) == 2 else " ".join(parts[1:])
        mode = parse_trace_mode(mode_arg)
        if mode is None:
            emit_warn(
                f"未知子命令或模式: {parts[1]!r}，可用 all | steps | reply | none | step | token | stats"
            )
            _print_trace_usage(trace_session)
            return True
        trace_session.mode = mode
        from llgraph.display.trace_display import TRACE_MODE_LABELS

        emit_ok(
            f"已切换: {TRACE_MODE_LABELS[mode]} ({mode.value})"
        )
        return True

    if lower == "/rule" or lower == "rule":
        emit_report(format_rules_list(workspace, context_session, last_user_message))
        return True

    if lower.startswith("/rule ") or lower.startswith("rule "):
        return _handle_rule_subcommand(stripped, workspace, context_session, last_user_message)

    if lower == "/skill" or lower == "skill":
        _print_skills_usage(workspace, context_session)
        return True

    if lower.startswith("/skill ") or lower.startswith("skill "):
        return _handle_skill_subcommand(stripped, workspace, context_session)

    if lower == "/changes" or lower == "changes":
        return _handle_changes_command(stripped, edit_tracker)

    if lower == "/undo" or lower == "undo":
        return _handle_undo_command(stripped, edit_tracker, allow_write)

    if lower.startswith("/undo ") or lower.startswith("undo "):
        return _handle_undo_command(stripped, edit_tracker, allow_write)

    if lower.startswith("/diff ") or lower.startswith("diff "):
        if edit_tracker is None:
            emit("请使用 llgraph -w 启动以启用 /diff。", colorize=True)
            return True
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            emit("用法: /diff <相对路径>", colorize=True)
            return True
        emit_report(edit_tracker.format_diff(parts[1]))
        return True

    if _try_custom_command(
        stripped,
        workspace=workspace,
        allow_write=allow_write,
        agent_session=agent_session,
        last_user_message=last_user_message,
    ):
        return True

    return False


def _handle_compress(workspace: Path, agent_session: AgentSessionContext | None) -> bool:
    """处理 /compress。"""
    if agent_session is None or not agent_session.with_memory:
        emit("当前无会话历史（交互模式默认有 memory；/compress 需多轮对话）。", colorize=True)
        return True
    from llgraph.context.context_settings import is_auto_compress_strategy, resolve_context_settings

    settings = resolve_context_settings(workspace)
    preserve = False if is_auto_compress_strategy(settings.compress_strategy) else None
    report = apply_compress_to_agent_state(
        agent_session.agent,
        thread_id=agent_session.thread_id,
        workspace=workspace,
        force=True,
        preserve_current_turn=preserve,
    )
    if report is None:
        emit("无需压缩或消息为空。", colorize=True)
    else:
        emit_report(format_compress_report(report))
        from llgraph.display.execution_log import log_compress_event

        log_compress_event(
            workspace,
            thread_id=agent_session.thread_id,
            report=report,
            trigger="manual",
        )
        if agent_session.context_session is not None:
            from llgraph.session.session_manifest import sync_session_manifest_to_agent_state

            sync_session_manifest_to_agent_state(
                agent_session.agent,
                thread_id=agent_session.thread_id,
                workspace=workspace,
                session=agent_session.context_session,
                user_message="",
                with_memory=True,
                archive_path=report.archive_path,
                allow_write=agent_session.allow_write,
            )
    return True


def _handle_review(
    stripped: str,
    workspace: Path,
    edit_tracker: SessionEditTracker | None,
    last_user_message: str,
) -> bool:
    """处理 /review [topic]。"""
    parts = stripped.split(maxsplit=1)
    topic = parts[1].strip() if len(parts) > 1 else ""
    from llgraph.display.trace_display import print_command_prelude

    print_command_prelude(
        "正在评审",
        detail="收集 git diff 并调用模型（约 30s～2min，请稍候）",
    )
    try:
        review_path, summary = run_review(
            workspace,
            topic=topic,
            edit_tracker=edit_tracker,
            last_user_message=last_user_message,
        )
    except Exception as exc:
        emit(f"评审失败: {exc}", colorize=True)
        return True
    emit(f"评审已落盘: {review_path}", colorize=True)
    if summary:
        emit("--- 摘要 ---", colorize=True)
        emit(summary, colorize=True)
    return True


def _try_custom_command(
    stripped: str,
    *,
    workspace: Path,
    allow_write: bool,
    agent_session: AgentSessionContext | None,
    last_user_message: str,
) -> bool:
    """
    尝试执行 .llgraph/commands 自定义命令。

    @param stripped 用户输入
    @return 是否已处理
    """
    if not stripped.startswith("/"):
        return False
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return False
    if not parts:
        return False
    token = parts[0].lstrip("/")
    cmd = resolve_command(workspace, token)
    if cmd is None:
        return False

    if cmd.requires_write and not allow_write:
        emit(f"命令 /{cmd.name} 需要写权限，请使用 llgraph -w 启动。", colorize=True)
        return True

    args_tail = stripped.split(maxsplit=1)[1].strip() if " " in stripped else ""
    if cmd.handler == "prompt":
        if agent_session is None:
            emit("Agent 未就绪，无法执行 prompt 命令。", colorize=True)
            return True
        from llgraph.core.agent import invoke_agent

        user_tail = args_tail or last_user_message or "请按命令说明执行。"
        effective = (
            f"<custom-command name=\"{cmd.name}\">\n"
            f"{cmd.body}\n"
            f"</custom-command>\n\n"
            f"用户补充: {user_tail}"
        )
        emit(f"执行自定义命令 /{cmd.name} …\n", colorize=True)
        try:
            invoke_agent(
                agent_session.agent,
                user_tail,
                workspace_root=workspace,
                thread_id=agent_session.thread_id,
                with_memory=agent_session.with_memory,
                trace_session=agent_session.trace_session,
                context_session=agent_session.context_session,
                effective_message_override=effective,
                write_failure_tracker=agent_session.write_failure_tracker,
                context_spill=agent_session.context_spill,
                allow_write=agent_session.allow_write,
            )
            emit()
        except Exception as exc:
            emit(f"命令执行失败: {exc}", colorize=True)
        return True

    if cmd.handler == "builtin":
        if cmd.name.lower() == "compress":
            return _handle_compress(workspace, agent_session)
        if cmd.name.lower() == "review":
            tracker = agent_session.edit_tracker if agent_session else None
            return _handle_review(
                f"/review {args_tail}".strip(),
                workspace,
                tracker,
                last_user_message,
            )
        emit(f"未知 builtin 命令: {cmd.name}", colorize=True)
        return True

    emit(f"不支持的 handler: {cmd.handler}（仅 prompt / builtin）", colorize=True)
    return True


def _handle_changes_command(stripped: str, edit_tracker: SessionEditTracker | None) -> bool:
    """
    处理 /changes、/diff 子命令。

    @param stripped 用户输入
    @param edit_tracker 会话编辑账本
    @return 恒为 True
    """
    if edit_tracker is None:
        emit("当前为只读模式或未启用编辑追踪；请使用 llgraph -w 启动。", colorize=True)
        return True

    parts = stripped.split(maxsplit=2)
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub == "clear":
        edit_tracker.clear_display()
        emit("已清空本会话变更列表（内存）；落盘 edits.jsonl 仍保留。", colorize=True)
        return True

    if sub == "reset":
        edit_tracker.reset_persisted()
        emit("已重置本会话变更记录与快照（含落盘）。", colorize=True)
        return True

    if sub == "diff" and len(parts) >= 3:
        emit_report(edit_tracker.format_diff(parts[2]))
        return True

    if sub == "diff":
        emit("用法: /changes diff <相对路径>", colorize=True)
        return True

    emit_report(edit_tracker.format_changes_list())
    return True


def _handle_undo_command(
    stripped: str,
    edit_tracker: SessionEditTracker | None,
    allow_write: bool,
) -> bool:
    """
    处理 /undo：从会话快照批量或单文件还原。

    @param stripped 用户输入
    @param edit_tracker 会话编辑账本
    @param allow_write 是否允许写磁盘
    @return 恒为 True
    """
    if edit_tracker is None:
        emit("请使用 llgraph -w 启动以启用 /undo。", colorize=True)
        return True

    parts = stripped.split(maxsplit=1)
    if len(parts) < 2:
        emit_report(edit_tracker.format_undo_usage())
        if not allow_write:
            emit(
                "说明: 只读模式下可查看与 /undo 还原；Agent 写工具仍禁用。",
                colorize=True,
            )
        return True

    target = parts[1].strip()
    if not target:
        emit_report(edit_tracker.format_undo_usage())
        if not allow_write:
            emit(
                "说明: 只读模式下可查看与 /undo 还原；Agent 写工具仍禁用。",
                colorize=True,
            )
        return True

    if not allow_write:
        emit("正在只读模式下从快照还原（Agent 写工具仍禁用）…", colorize=True)

    if target.lower() == "all":
        results = edit_tracker.restore_all()
        emit_report(edit_tracker.format_undo_report(results))
        return True

    result = edit_tracker.restore_path(target)
    emit_report(edit_tracker.format_undo_report([result]))
    return True


def _handle_rule_subcommand(
    stripped: str,
    workspace: Path,
    session: ContextSession,
    last_user_message: str,
) -> bool:
    parts = stripped.split(maxsplit=2)
    if len(parts) < 2:
        emit_report(format_rules_list(workspace, session, last_user_message))
        return True

    sub = parts[1].lower()
    if sub in ("list", "ls"):
        emit_report(format_rules_list(workspace, session, last_user_message))
        return True

    if sub == "reset":
        session.disabled_rules.clear()
        session.forced_rules.clear()
        emit("已重置规则覆盖（恢复默认 alwaysApply + glob 自动匹配）", colorize=True)
        return True

    if sub in ("on", "enable") and len(parts) >= 3:
        rule_id = _resolve_rule_id(workspace, parts[2])
        if not rule_id:
            emit(f"未找到规则: {parts[2]!r}", colorize=True)
            return True
        session.disabled_rules.discard(rule_id)
        session.forced_rules.add(rule_id)
        emit(f"已强制启用: {rule_id}", colorize=True)
        return True

    if sub in ("off", "disable") and len(parts) >= 3:
        rule_id = _resolve_rule_id(workspace, parts[2])
        if not rule_id:
            emit(f"未找到规则: {parts[2]!r}", colorize=True)
            return True
        session.forced_rules.discard(rule_id)
        session.disabled_rules.add(rule_id)
        emit(f"已禁用: {rule_id}", colorize=True)
        return True

    emit_report(format_rules_list(workspace, session, last_user_message))
    return True


def _resolve_rule_id(workspace: Path, hint: str) -> str | None:
    hint = hint.strip()
    rules = discover_rules(workspace)
    for rule in rules:
        if rule.rule_id == hint or rule.rule_id.endswith(hint):
            return rule.rule_id
    for rule in rules:
        if hint in rule.rule_id or hint in rule.description:
            return rule.rule_id
    return None


def _handle_skill_subcommand(stripped: str, workspace: Path, session: ContextSession) -> bool:
    parts = stripped.split(maxsplit=2)
    if len(parts) < 2:
        _print_skills_usage(workspace, session)
        return True

    sub = parts[1].lower()

    if sub == "clear":
        session.clear_skills()
        emit("已清空置顶技能", colorize=True)
        return True

    if sub == "auto":
        emit("已移除 /skill auto；技能全量在 manifest，模型按需 read_file", colorize=True)
        return True

    if sub == "off" and len(parts) >= 3:
        name = parts[2]
        if session.deactivate_skill(name):
            emit(f"已关闭技能: {name}", colorize=True)
        else:
            emit(f"技能未启用: {name!r}", colorize=True)
        return True

    skill_name = parts[1]
    skills = discover_skills(workspace)
    known = {s.name.lower(): s.name for s in skills}
    key = skill_name.lower()
    if key not in known:
        emit(f"未找到技能: {skill_name!r}（见 /skill 列表）", colorize=True)
        return True
    session.activate_skill(known[key])
    emit(f"已置顶技能: {known[key]}（下一条消息 manifest ⭐；正文 read_file）", colorize=True)
    return True
