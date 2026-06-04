"""交互命令：/rule、/skill、/compress、/review、自定义 commands。"""

import re
import shlex
from pathlib import Path
from llgraph.core.agent_session import AgentSessionContext
from llgraph.loaders.commands_loader import format_commands_help, resolve_command
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

    print(
        format_spill_stats(
            messages_tokens=messages_tokens,
            spilled_bytes=spilled_bytes,
            spill_count=spill_count,
            cacheable_prefix_estimate=cacheable,
        ),
        flush=True,
    )
    spill_dir = workspace / ".llgraph" / "context" / "tool-results"
    print(f"落盘目录: {spill_dir}", flush=True)
    try:
        from llgraph.display.execution_log import (
            execution_log_path,
            extract_usage_from_messages,
            read_execution_tail,
        )
        from llgraph.core.llm_settings import resolve_effective_model

        print(f"执行日志: {execution_log_path(workspace)}", flush=True)
        print(f"当前模型: {resolve_effective_model(workspace)}", flush=True)
        if agent_session is not None:
            config = {"configurable": {"thread_id": agent_session.thread_id}}
            try:
                state = agent_session.agent.get_state(config)
                messages = list((state.values or {}).get("messages") or [])
                usage = extract_usage_from_messages(messages)
                totals = usage.get("totals") or {}
                if totals:
                    print(
                        "网关 usage 累计: "
                        f"in={totals.get('input_tokens', 0)} "
                        f"out={totals.get('output_tokens', 0)} "
                        f"cache_read={totals.get('cache_read_input_tokens', 0)}",
                        flush=True,
                    )
            except Exception:
                pass
        last = read_execution_tail(workspace, limit=1)
        if last:
            from llgraph.display.execution_log import format_execution_record

            print(f"最近一轮: {format_execution_record(last[0])}", flush=True)
    except Exception:
        pass
    print("更多: /log tail | /log purge", flush=True)


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
        print(f"无法解析 /index 参数: {exc}", flush=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/index", "index"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens:
        print_index_status(workspace)
        print("", flush=True)
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
    处理 /survey：off|on|status 或打开确认向导。

    @param line 用户输入
    @param workspace 工作区根
    @param agent_session Agent 会话
    @param context_session Rule/Skill 会话状态
    @return 恒为 True
    """
    from llgraph.config.survey_settings import (
        format_survey_status,
        survey_command_enabled,
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
        print("Survey 交互已关闭（本会话）。", flush=True)
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
        print("Survey 交互已开启（本会话）。", flush=True)
        return True
    if sub in ("status", "?"):
        print(format_survey_status(workspace, context_session), flush=True)
        return True

    if not survey_command_enabled(workspace, context_session):
        print(
            "Survey 已禁用（--no-survey / agent.json survey.enabled=false / /survey off）。",
            flush=True,
        )
        print(format_survey_status(workspace, context_session), flush=True)
        return True

    from llgraph.survey.survey_prompt import (
        default_project_organize_survey,
        format_survey_answers_for_agent,
        run_survey_interactive,
    )

    spec = default_project_organize_survey()
    answers = run_survey_interactive(spec)
    if answers is None:
        return True
    payload = format_survey_answers_for_agent(answers)
    print("\n--- 确认结果 ---\n", flush=True)
    print(payload, flush=True)
    if agent_session is None:
        print("\n（无 Agent 会话，请复制以上内容作为下一条消息发送）", flush=True)
        return True
    from llgraph.core.agent import invoke_agent

    try:
        print("\n▶ 正在将确认结果提交给 Agent…\n", flush=True)
        invoke_agent(
            agent_session.agent,
            payload,
            workspace_root=workspace,
            thread_id=agent_session.thread_id,
            with_memory=agent_session.with_memory,
            trace_session=agent_session.trace_session,
            context_session=agent_session.context_session,
            write_failure_tracker=agent_session.write_failure_tracker,
            context_spill=agent_session.context_spill,
        )
        print()
    except Exception as exc:
        print(f"运行失败: {exc}", flush=True)
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
        print("仅交互模式支持 /watch。", flush=True)
        return True

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        print(f"无法解析 /watch 参数: {exc}", flush=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/watch", "watch"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens or tokens[0].lower() in ("status", "?"):
        print(format_watch_status(agent_session.watch_service, workspace), flush=True)
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
            print(err, flush=True)
        elif was_active:
            print("index watch 已在运行。", flush=True)
        else:
            print("已启动 index watch（保存文件后自动增量索引）。", flush=True)
        return True

    if sub in ("off", "false", "0", "stop", "disable"):
        service = agent_session.watch_service
        if service is None or not service.active:
            print("index watch 未运行。", flush=True)
            return True
        service.stop()
        print("已停止 index watch。", flush=True)
        return True

    print(
        "用法: /watch  |  /watch status  |  /watch on  |  /watch off",
        flush=True,
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
        print("仅交互模式支持 /write。", flush=True)
        return True

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        print(f"无法解析 /write 参数: {exc}", flush=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/write", "write"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens:
        print(format_write_mode_status(agent_session), flush=True)
        return True

    sub = tokens[0].lower()
    if sub in ("on", "true", "1", "enable", "w"):
        if set_session_write_mode(agent_session, enabled=True, context_session=context_session):
            print("已切换为可写模式（写工具与受限 shell 已启用，会话历史已保留）。", flush=True)
        else:
            print("当前已是可写模式。", flush=True)
        return True

    if sub in ("off", "false", "0", "disable", "ro", "readonly"):
        if set_session_write_mode(agent_session, enabled=False, context_session=context_session):
            print("已切换为只读模式（禁止写文件；/changes · /undo 仍可用）。", flush=True)
        else:
            print("当前已是只读模式。", flush=True)
        return True

    print(
        "用法: /write  |  /write on  |  /write off\n"
        "说明: 切换后重建 Agent，对话历史保留；等价于启动时 -w 开关。",
        flush=True,
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
        print("仅交互模式支持 /web。", flush=True)
        return True

    try:
        parts = shlex.split(line.strip())
    except ValueError as exc:
        print(f"无法解析 /web 参数: {exc}", flush=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/web", "web"):
        tokens = parts[1:]
    else:
        tokens = parts

    if not tokens or tokens[0].lower() in ("status", "?"):
        print(format_web_search_status(agent_session), flush=True)
        return True

    sub = tokens[0].lower()
    if sub in ("on", "true", "1", "start", "enable"):
        changed, msg = set_session_web_search_mode(agent_session, enabled=True)
        print(msg, flush=True)
        if not changed and "未配置" in msg:
            pass
        return True

    if sub in ("off", "false", "0", "stop", "disable"):
        changed, msg = set_session_web_search_mode(agent_session, enabled=False)
        print(msg, flush=True)
        return True

    print(
        "用法: /web  |  /web status  |  /web on  |  /web off\n"
        "说明: 启用后注册 web_search（Tavily）；切换后重建 Agent，对话历史保留。",
        "Skill/Rule: 仅注入描述+路径，正文请 read_file；锚点见 <session-manifest>。",
        flush=True,
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
        print(f"无法解析 /model 参数: {exc}", flush=True)
        return True

    tokens: list[str] = []
    if parts and parts[0].lower() in ("/model", "model"):
        tokens = parts[1:]
    else:
        tokens = parts

    current = resolve_effective_model(workspace)
    write_mode = _effective_allow_write(agent_session, allow_write)

    if not tokens or tokens[0].lower() in ("list", "ls", "?"):
        print(format_model_status(workspace), flush=True)
        print("", flush=True)
        try:
            print(format_models_list(workspace, current=current), flush=True)
        except RuntimeError as exc:
            print(f"拉取网关模型列表失败: {exc}", flush=True)
            print("仍可用 /model <模型名> 手动切换。", flush=True)
        return True

    if tokens[0].lower() == "refresh":
        try:
            print(
                format_models_list(workspace, current=current, force_refresh=True),
                flush=True,
            )
        except RuntimeError as exc:
            print(f"刷新失败: {exc}", flush=True)
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
        print(f"已恢复默认模型: {restored}", flush=True)
        return True

    new_model = tokens[0].strip()
    if not new_model:
        print("用法: /model <模型名>  |  /model list  |  /model reset", flush=True)
        return True

    set_runtime_model(new_model)
    if not is_catalog_model(workspace, new_model):
        print(
            f"提示: {new_model!r} 不在 agent.json 模型目录（llm.models）内，"
            "若调用失败请 /model list 换模型。",
            flush=True,
        )
    if agent_session is not None:
        rebuild_agent_preserving_memory(
            agent_session,
            allow_write=write_mode,
            mcp_tools=agent_session.mcp_tools,
            on_file_changed=agent_session.on_file_changed if write_mode else None,
        )
    print(f"已切换模型: {new_model}（下一条消息起生效，会话历史已保留）", flush=True)
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
        print("需要交互会话才能切换 thread_id。", flush=True)
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
        print(
            format_session_command_help(workspace, agent_session.thread_id),
            flush=True,
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
        print(msg, flush=True)
        return True

    if sub in ("use", "switch", "resume") and len(parts) >= 3:
        tid = parts[2].strip()
        _, msg = switch_agent_thread(
            agent_session, tid, no_spill=_no_spill()
        )
        print(msg, flush=True)
        return True

    if sub in ("title", "rename", "name"):
        from llgraph.session.session_meta import get_session_title, set_session_title

        m = re.match(
            r"(?i)^(?:/session|session)\s+(?:title|rename|name)\s+"
            r"(?:(cli-[0-9a-f]{8})\s+)?(.+)$",
            stripped.strip(),
        )
        if not m:
            print(
                "用法: /session title <新标题>\n"
                "      /session title <thread_id> <新标题>",
                flush=True,
            )
            return True
        tid = m.group(1) or agent_session.thread_id
        title_text = m.group(2).strip()
        ok, msg = set_session_title(workspace, tid, title_text, source="manual")
        print(msg, flush=True)
        if ok and tid == agent_session.thread_id:
            from llgraph.session.session_meta import resolve_session_display_title

            print(
                f"当前展示: {resolve_session_display_title(workspace, tid)}",
                flush=True,
            )
        return True

    if sub in ("delete", "del", "rm"):
        from llgraph.session.session_delete import (
            delete_session,
            delete_sessions,
            format_delete_report,
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
            print(
                "用法:\n"
                "  /session delete <thread_id>           删除指定会话\n"
                "  /session delete <thread_id> --force   删除当前会话\n"
                "  /session delete empty                 删除空壳会话\n"
                "  /session delete all                   删除除当前外全部\n"
                "  /session delete all --including-current  全量删除并切到新会话",
                flush=True,
            )
            return True

        target = tokens[2]
        if target.lower() == "empty":
            empty_ids = list_empty_session_ids(workspace)
            if not empty_ids:
                print("（无空壳会话）", flush=True)
                return True
            report = delete_sessions(workspace, empty_ids)
            print(format_delete_report(report), flush=True)
            return True

        if target.lower() == "all":
            including_current = (
                "--including-current" in flags or "--all" in flags
            )
            ids = list_workspace_session_ids(workspace)
            if not ids:
                print("（无可删除会话）", flush=True)
                return True
            if not including_current:
                ids = [i for i in ids if i != agent_session.thread_id]
            if not ids:
                print(
                    "除当前外无其它会话；全量删除请加: /session delete all --including-current",
                    flush=True,
                )
                return True
            deleted_current = agent_session.thread_id in ids
            report = delete_sessions(workspace, ids)
            print(format_delete_report(report), flush=True)
            if deleted_current and report.success_count > 0:
                new_id = create_new_thread_id()
                set_session_title(
                    workspace, new_id, default_session_title(new_id), source="fallback"
                )
                _, msg = switch_agent_thread(
                    agent_session, new_id, no_spill=_no_spill()
                )
                print(msg, flush=True)
            return True

        try:
            tid = validate_thread_id(target)
        except ValueError as exc:
            print(str(exc), flush=True)
            return True

        if tid == agent_session.thread_id and "--force" not in flags:
            print(
                "不能删除当前会话。请先 /session new，或:\n"
                "  /session delete "
                + tid
                + " --force",
                flush=True,
            )
            return True

        result = delete_session(workspace, tid)
        if result.ok:
            print(f"已删除会话 {tid}。", flush=True)
            for path in result.removed_paths:
                print(f"  - {path}", flush=True)
            if not result.removed_paths:
                print("  （该会话无落盘文件或已不存在）", flush=True)
        else:
            print(f"删除失败 {tid}: {result.error}", flush=True)
            return True

        if tid == agent_session.thread_id:
            new_id = create_new_thread_id()
            set_session_title(
                workspace, new_id, default_session_title(new_id), source="fallback"
            )
            _, msg = switch_agent_thread(
                agent_session, new_id, no_spill=_no_spill()
            )
            print(msg, flush=True)
        return True

    print(
        format_session_command_help(workspace, agent_session.thread_id),
        flush=True,
    )
    return True


def _print_skills_usage(workspace: Path, session: ContextSession) -> None:
    from llgraph.config.catalog_paths import scope_label

    skills = discover_skills(workspace)
    print(
        "技能（Skills）— 项目: .llgraph/skills/<name>/SKILL.md  |  个人: ~/.llgraph/skills/",
        flush=True,
    )
    if not skills:
        print(
            "  （未找到技能；llgraph --init-config / --init-user-config）",
            flush=True,
        )
    else:
        for skill in skills:
            active = "✓" if skill.name.lower() in [s.lower() for s in session.active_skills] else " "
            origin = scope_label(skill.scope)
            print(f"  [{active}] {skill.name} [{origin}] — {skill.description}", flush=True)
    active = ", ".join(session.active_skills) if session.active_skills else "（无）"
    print(f"当前启用: {active}", flush=True)
    print("自动匹配: " + ("开" if session.auto_match_skills else "关"), flush=True)
    print("", flush=True)
    print("命令:", flush=True)
    print("  /skill              列出技能", flush=True)
    print("  /skill <name>       启用技能（可多次）", flush=True)
    print("  /skill off <name>   关闭指定技能", flush=True)
    print("  /skill clear        清空已启用技能", flush=True)
    print("  /skill auto on|off  开关按消息自动匹配", flush=True)
    print("  说明: 仅注入描述+路径，正文须 read_file；会话锚点 <session-manifest>", flush=True)


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
    from llgraph.context.context_builder import format_rules_list
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
        print(
            "请直接输入 /paste 后回车进入粘贴模式，或:\n"
            "  1. 输入 /paste 回车\n"
            "  2. 粘贴完整报错/日志\n"
            "  3. 单独一行输入 --- 结束",
            flush=True,
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

        print(format_agent_config_sources(workspace), flush=True)
        return True

    if lower == "/session" or lower == "session":
        return _handle_session_command(stripped, workspace, agent_session)

    if lower.startswith("/session ") or lower.startswith("session "):
        return _handle_session_command(stripped, workspace, agent_session)

    if lower in ("/sessionid", "sessionid", "/session-id", "session-id"):
        if agent_session is None:
            print("需要交互会话才能查看 thread_id。", flush=True)
            return True
        from llgraph.session.session_switch import print_current_session_info

        print_current_session_info(workspace, agent_session.thread_id)
        return True

    if lower in ("/sessions", "sessions"):
        from llgraph.session.session_registry import format_sessions_list

        current = agent_session.thread_id if agent_session is not None else None
        print(format_sessions_list(workspace, current_thread_id=current), flush=True)
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
        print(format_commands_help(workspace), flush=True)
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
        print(
            format_agent_tools_report(
                workspace,
                allow_write=allow_write,
                web_search_enabled=web_enabled,
                mcp_tools=mcp,
            ),
            flush=True,
        )
        return True

    if lower == "/review" or lower.startswith("/review "):
        return _handle_review(stripped, workspace, edit_tracker, last_user_message)

    if lower == "/commands" or lower == "commands":
        print(format_commands_help(workspace), flush=True)
        return True

    if lower == "/trace" or lower == "trace":
        _print_trace_usage(trace_session)
        return True

    if lower == "/log" or lower == "log":
        from llgraph.config.logging_settings import format_log_status

        print(format_log_status(workspace), flush=True)
        return True

    if lower.startswith("/log ") or lower.startswith("log "):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            from llgraph.config.logging_settings import format_log_status

            print(format_log_status(workspace), flush=True)
            return True
        sub = parts[1].strip().lower()
        if sub in ("tail", "exec", "execution"):
            from llgraph.display.execution_log import format_execution_tail

            print(format_execution_tail(workspace), flush=True)
            return True
        if sub == "purge":
            from llgraph.display.log_retention import format_purge_report, run_log_retention

            report = run_log_retention(workspace, quiet=False)
            print(format_purge_report(report), flush=True)
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
            print(
                f"向量检索落盘 search.log: {'开' if file_on else '关'}",
                flush=True,
            )
            return True
        from llgraph.config.logging_settings import level_name, set_runtime_log_level

        try:
            effective = set_runtime_log_level(workspace, parts[1].strip())
        except Exception:
            print(
                f"未知级别: {parts[1]!r}，可用 debug | info | warning | error",
                flush=True,
            )
            return True
        print(f"已切换向量检索日志: {level_name(effective)}", flush=True)
        return True

    if lower.startswith("/trace ") or lower.startswith("trace "):
        parts = stripped.split()
        if len(parts) < 2:
            _print_trace_usage(trace_session)
            return True
        sub = parts[1].strip().lower()
        if sub in ("stats",):
            _print_trace_stats(workspace, agent_session)
            return True
        if sub in ("token", "tokens"):
            if len(parts) >= 3:
                arg = parts[2].strip().lower()
                if arg in ("on", "off"):
                    enabled = set_trace_step_tokens(trace_session, arg == "on")
                    print(
                        f"步骤 token 显示: {'开' if enabled else '关'}",
                        flush=True,
                    )
                    return True
                if arg in ("stats",):
                    _print_trace_stats(workspace, agent_session)
                    return True
            enabled = set_trace_step_tokens(trace_session, None)
            print(
                f"步骤 token 显示: {'开' if enabled else '关'}",
                flush=True,
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
            print(
                f"未知子命令或模式: {parts[1]!r}，可用 all | steps | reply | none | step | token | stats",
                flush=True,
            )
            _print_trace_usage(trace_session)
            return True
        trace_session.mode = mode
        from llgraph.display.trace_display import TRACE_MODE_LABELS

        print(
            f"已切换: {TRACE_MODE_LABELS[mode]} ({mode.value})",
            flush=True,
        )
        return True

    if lower == "/rule" or lower == "rule":
        print(format_rules_list(workspace, context_session, last_user_message), flush=True)
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
            print("请使用 llgraph -w 启动以启用 /diff。", flush=True)
            return True
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            print("用法: /diff <相对路径>", flush=True)
            return True
        print(edit_tracker.format_diff(parts[1]), flush=True)
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
        print("当前无会话历史（交互模式默认有 memory；/compress 需多轮对话）。", flush=True)
        return True
    report = apply_compress_to_agent_state(
        agent_session.agent,
        thread_id=agent_session.thread_id,
        workspace=workspace,
        force=True,
    )
    if report is None:
        print("无需压缩或消息为空。", flush=True)
    else:
        print(format_compress_report(report), flush=True)
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
        print(f"评审失败: {exc}", flush=True)
        return True
    print(f"评审已落盘: {review_path}", flush=True)
    if summary:
        print("--- 摘要 ---", flush=True)
        print(summary, flush=True)
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
        print(f"命令 /{cmd.name} 需要写权限，请使用 llgraph -w 启动。", flush=True)
        return True

    args_tail = stripped.split(maxsplit=1)[1].strip() if " " in stripped else ""
    if cmd.handler == "prompt":
        if agent_session is None:
            print("Agent 未就绪，无法执行 prompt 命令。", flush=True)
            return True
        from llgraph.core.agent import invoke_agent

        user_tail = args_tail or last_user_message or "请按命令说明执行。"
        effective = (
            f"<custom-command name=\"{cmd.name}\">\n"
            f"{cmd.body}\n"
            f"</custom-command>\n\n"
            f"用户补充: {user_tail}"
        )
        print(f"执行自定义命令 /{cmd.name} …\n", flush=True)
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
            )
            print()
        except Exception as exc:
            print(f"命令执行失败: {exc}", flush=True)
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
        print(f"未知 builtin 命令: {cmd.name}", flush=True)
        return True

    print(f"不支持的 handler: {cmd.handler}（仅 prompt / builtin）", flush=True)
    return True


def _handle_changes_command(stripped: str, edit_tracker: SessionEditTracker | None) -> bool:
    """
    处理 /changes、/diff 子命令。

    @param stripped 用户输入
    @param edit_tracker 会话编辑账本
    @return 恒为 True
    """
    if edit_tracker is None:
        print("当前为只读模式或未启用编辑追踪；请使用 llgraph -w 启动。", flush=True)
        return True

    parts = stripped.split(maxsplit=2)
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub == "clear":
        edit_tracker.clear_display()
        print("已清空本会话变更列表（内存）；落盘 edits.jsonl 仍保留。", flush=True)
        return True

    if sub == "reset":
        edit_tracker.reset_persisted()
        print("已重置本会话变更记录与快照（含落盘）。", flush=True)
        return True

    if sub == "diff" and len(parts) >= 3:
        print(edit_tracker.format_diff(parts[2]), flush=True)
        return True

    if sub == "diff":
        print("用法: /changes diff <相对路径>", flush=True)
        return True

    print(edit_tracker.format_changes_list(), flush=True)
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
        print("请使用 llgraph -w 启动以启用 /undo。", flush=True)
        return True
    if not allow_write:
        print("当前为只读模式，无法写回文件。请使用 llgraph -w 启动。", flush=True)
        return True

    parts = stripped.split(maxsplit=1)
    if len(parts) < 2:
        print(edit_tracker.format_undo_usage(), flush=True)
        return True

    target = parts[1].strip()
    if not target:
        print(edit_tracker.format_undo_usage(), flush=True)
        return True

    if target.lower() == "all":
        results = edit_tracker.restore_all()
        print(edit_tracker.format_undo_report(results), flush=True)
        return True

    result = edit_tracker.restore_path(target)
    print(edit_tracker.format_undo_report([result]), flush=True)
    return True


def _handle_rule_subcommand(
    stripped: str,
    workspace: Path,
    session: ContextSession,
    last_user_message: str,
) -> bool:
    parts = stripped.split(maxsplit=2)
    if len(parts) < 2:
        print(format_rules_list(workspace, session, last_user_message), flush=True)
        return True

    sub = parts[1].lower()
    if sub in ("list", "ls"):
        print(format_rules_list(workspace, session, last_user_message), flush=True)
        return True

    if sub == "reset":
        session.disabled_rules.clear()
        session.forced_rules.clear()
        print("已重置规则覆盖（恢复默认 alwaysApply + glob 自动匹配）", flush=True)
        return True

    if sub in ("on", "enable") and len(parts) >= 3:
        rule_id = _resolve_rule_id(workspace, parts[2])
        if not rule_id:
            print(f"未找到规则: {parts[2]!r}", flush=True)
            return True
        session.disabled_rules.discard(rule_id)
        session.forced_rules.add(rule_id)
        print(f"已强制启用: {rule_id}", flush=True)
        return True

    if sub in ("off", "disable") and len(parts) >= 3:
        rule_id = _resolve_rule_id(workspace, parts[2])
        if not rule_id:
            print(f"未找到规则: {parts[2]!r}", flush=True)
            return True
        session.forced_rules.discard(rule_id)
        session.disabled_rules.add(rule_id)
        print(f"已禁用: {rule_id}", flush=True)
        return True

    print(format_rules_list(workspace, session, last_user_message), flush=True)
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
        print("已清空启用的技能", flush=True)
        return True

    if sub == "auto" and len(parts) >= 3:
        flag = parts[2].lower()
        if flag in ("on", "1", "true"):
            session.auto_match_skills = True
            print("已开启技能自动匹配", flush=True)
        elif flag in ("off", "0", "false"):
            session.auto_match_skills = False
            print("已关闭技能自动匹配", flush=True)
        else:
            print("用法: /skill auto on|off", flush=True)
        return True

    if sub == "off" and len(parts) >= 3:
        name = parts[2]
        if session.deactivate_skill(name):
            print(f"已关闭技能: {name}", flush=True)
        else:
            print(f"技能未启用: {name!r}", flush=True)
        return True

    skill_name = parts[1]
    skills = discover_skills(workspace)
    known = {s.name.lower(): s.name for s in skills}
    key = skill_name.lower()
    if key not in known:
        print(f"未找到技能: {skill_name!r}（见 /skill 列表）", flush=True)
        return True
    session.activate_skill(known[key])
    print(f"已启用技能: {known[key]}（下一条消息起注入）", flush=True)
    return True
