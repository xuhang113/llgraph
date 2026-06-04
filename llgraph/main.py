"""命令行入口：运行 Gateway-backed LangGraph Agent。"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from llgraph.core.agent import build_agent
from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.core.agent_session import AgentSessionContext
from llgraph.config.config import load_llgraph_env
from llgraph.context.context_session import ContextSession
from llgraph.session.session_edits import SessionEditTracker
from llgraph.display.trace_display import TraceMode, TraceSession, parse_trace_mode
from llgraph.display.ui_mode import UiMode, parse_ui_mode, resolve_ui_mode
from llgraph.config.workspace_config import init_user_llgraph, init_workspace_llgraph


def _run_once(
    agent,
    message: str,
    *,
    workspace: Path,
    thread_id: str,
    with_memory: bool,
    trace_session: TraceSession,
    context_session: ContextSession,
    allow_write: bool = False,
    write_failure_tracker: WriteFailureTracker | None = None,
    ui_mode: UiMode = UiMode.TERMINAL,
) -> None:
    """单轮模式：执行一条消息后退出。"""
    if ui_mode == UiMode.TERMINAL:
        from llgraph.terminal.session import TerminalSessionParams, run_terminal_session

        run_terminal_session(
            TerminalSessionParams(
                agent=agent,
                workspace=workspace,
                thread_id=thread_id,
                trace_session=trace_session,
                context_session=context_session,
                allow_write=allow_write,
                write_failure_tracker=write_failure_tracker,
                opening_message=message,
                single_turn=True,
            )
        )
        return

    from llgraph.ui.app import TuiSessionParams, run_tui_session

    run_tui_session(
        TuiSessionParams(
            agent=agent,
            workspace=workspace,
            thread_id=thread_id,
            trace_session=trace_session,
            context_session=context_session,
            allow_write=allow_write,
            write_failure_tracker=write_failure_tracker,
            opening_message=message,
            single_turn=True,
        )
    )


def _run_interactive(
    agent,
    *,
    thread_id: str,
    workspace: Path,
    allow_write: bool,
    trace_session: TraceSession,
    context_session: ContextSession,
    edit_tracker: SessionEditTracker | None = None,
    watch_active: bool = False,
    web_search_enabled: bool = False,
    mcp_summary: str = "",
    agent_session: AgentSessionContext | None = None,
    opening_message: str | None = None,
    write_failure_tracker: WriteFailureTracker | None = None,
    resume_hint: str = "",
    memory_kind: str = "",
    ui_mode: UiMode = UiMode.TERMINAL,
) -> None:
    """交互会话（默认经典终端，可选 TUI）。"""
    tid = agent_session.thread_id if agent_session is not None else thread_id

    if ui_mode == UiMode.TERMINAL:
        from llgraph.terminal.session import TerminalSessionParams, run_terminal_session

        run_terminal_session(
            TerminalSessionParams(
                agent=agent,
                workspace=workspace,
                thread_id=tid,
                trace_session=trace_session,
                context_session=context_session,
                allow_write=allow_write,
                agent_session=agent_session,
                edit_tracker=edit_tracker,
                write_failure_tracker=write_failure_tracker,
                watch_active=watch_active,
                web_search_enabled=web_search_enabled,
                mcp_summary=mcp_summary,
                resume_hint=resume_hint,
                memory_kind=memory_kind,
                opening_message=opening_message,
            )
        )
    else:
        from llgraph.ui.app import TuiSessionParams, run_tui_session

        run_tui_session(
            TuiSessionParams(
                agent=agent,
                workspace=workspace,
                thread_id=tid,
                trace_session=trace_session,
                context_session=context_session,
                allow_write=allow_write,
                agent_session=agent_session,
                edit_tracker=edit_tracker,
                write_failure_tracker=write_failure_tracker,
                watch_active=watch_active,
                web_search_enabled=web_search_enabled,
                mcp_summary=mcp_summary,
                resume_hint=resume_hint,
                memory_kind=memory_kind,
                opening_message=opening_message,
            )
        )

    from llgraph.session.session_switch import print_session_exit_hint

    print_session_exit_hint(workspace, tid)


def main() -> None:
    """解析参数并执行 Agent（默认交互会话）。"""
    if len(sys.argv) >= 2 and sys.argv[1] == "index":
        from llgraph.cli.index_cli import main as index_main

        index_main(sys.argv[2:])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "search":
        from llgraph.cli.search_cli import main as search_main

        search_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description="llgraph — LangGraph Agent（默认经典终端，可用 --ui tui）",
    )
    parser.add_argument(
        "--ui",
        default=None,
        metavar="MODE",
        help="交互界面：terminal（默认）| tui（Textual）；也可用环境变量 LLGRAPH_UI",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="首条用户消息（省略则进入交互；配合 --once 为单轮）",
    )
    parser.add_argument(
        "--once",
        "-1",
        action="store_true",
        help="单轮模式：执行一条消息后退出（不进入交互循环）",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="单轮模式下启用会话记忆（交互模式默认已开启）",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="恢复或指定会话 thread_id（如 cli-c7a2fbca）；省略则新建",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="列出本工作区可恢复的会话后退出",
    )
    parser.add_argument(
        "--delete-session",
        default=None,
        metavar="ID",
        help="删除指定 thread_id 会话落盘后退出（如 cli-c7a2fbca）",
    )
    parser.add_argument(
        "--purge-sessions",
        action="store_true",
        help="删除本工作区全部会话落盘；须同时指定 --including-current 确认",
    )
    parser.add_argument(
        "--including-current",
        action="store_true",
        help="与 --purge-sessions 合用，确认全量删除",
    )
    parser.add_argument(
        "-C",
        "--workspace",
        default=None,
        metavar="DIR",
        help="工作区根目录（文件工具限制在此目录内），默认为启动时的当前目录",
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="允许写入文件（默认只读；含 search_replace / write_file）",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="启用 OS 沙箱（macOS sandbox-exec / Linux bwrap；覆盖 sandbox.json enabled=false）",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="禁用 OS 沙箱（覆盖 sandbox.json enabled=true）",
    )
    parser.add_argument(
        "--no-spill",
        action="store_true",
        help="禁用大工具结果落盘（调试；默认开启 P6 动态上下文）",
    )
    parser.add_argument(
        "--no-watch-index",
        action="store_true",
        help="不随 Agent 启动工作区文件监听与自动增量索引",
    )
    parser.add_argument(
        "--trace",
        choices=[m.value for m in TraceMode],
        default=TraceMode.STEPS.value,
        help="过程展示：all=完整，steps=折叠步骤，reply=仅回复，none=静默（默认 steps）",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=4,
        metavar="N",
        help="steps 模式下每轮末步预览行数（默认 4）",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="将包内默认 .llgraph/ 复制到工作区（不读取 .cursor；不覆盖已有文件）",
    )
    parser.add_argument(
        "--init-config-force",
        action="store_true",
        help="与 --init-config 相同，但覆盖已存在的模板文件",
    )
    parser.add_argument(
        "--init-user-config",
        action="store_true",
        help="将默认配置复制到 ~/.llgraph/（用户级 agent.json，全工作区共享）",
    )
    parser.add_argument(
        "--init-user-config-force",
        action="store_true",
        help="与 --init-user-config 相同，但覆盖已有用户配置",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        metavar="LEVEL",
        help="向量检索日志级别：debug|info|warning|error（默认仅 search.log，不刷对话区）",
    )
    parser.add_argument(
        "--log-console",
        action="store_true",
        help="将向量检索 [vector] 日志同时输出到终端 stderr",
    )
    parser.add_argument(
        "--no-survey",
        action="store_true",
        help="禁用交互式 survey（前置向导/助手确认/弹窗；适合长期非交互 Agent）",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="启动时使用的 AI 网关模型（等同会话内 /model；默认 LLGRAPH_MODEL 或 agent.json llm.model）",
    )
    args = parser.parse_args()

    if args.ui is not None and parse_ui_mode(args.ui) is None:
        print(
            f"错误: 未知 --ui 模式 {args.ui!r}，可选: tui, terminal",
            file=sys.stderr,
        )
        sys.exit(1)
    ui_mode = resolve_ui_mode(args.ui)

    load_llgraph_env()

    workspace = Path(args.workspace or os.getcwd()).expanduser().resolve()
    if not workspace.is_dir():
        print(f"错误: 工作区不是有效目录: {workspace}", file=sys.stderr)
        sys.exit(1)

    if args.list_sessions:
        from llgraph.session.session_registry import format_sessions_list

        print(format_sessions_list(workspace), flush=True)
        return

    if args.delete_session:
        from llgraph.session.session_delete import delete_session, validate_thread_id

        try:
            tid = validate_thread_id(args.delete_session.strip())
        except ValueError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            sys.exit(1)
        result = delete_session(workspace, tid)
        if result.ok:
            print(f"已删除会话 {tid}。", flush=True)
            for path in result.removed_paths:
                print(f"  - {path}", flush=True)
            if not result.removed_paths:
                print("  （无落盘文件或已不存在）", flush=True)
        else:
            print(f"删除失败: {result.error}", file=sys.stderr)
            sys.exit(1)
        return

    if args.purge_sessions:
        if not args.including_current:
            print(
                "错误: 全量删除须加 --including-current 确认",
                file=sys.stderr,
            )
            sys.exit(1)
        from llgraph.session.session_delete import delete_sessions, format_delete_report
        from llgraph.session.session_registry import list_workspace_session_ids

        ids = list_workspace_session_ids(workspace)
        if not ids:
            print("（无可删除会话）", flush=True)
            return
        report = delete_sessions(workspace, ids)
        print(format_delete_report(report), flush=True)
        if report.failure_count:
            sys.exit(1)
        return

    from llgraph.config.logging_settings import level_name, setup_search_logging
    from llgraph.core.llm_settings import resolve_effective_model, set_runtime_model

    from llgraph.display.execution_log import startup_logging_maintenance

    startup_logging_maintenance(workspace)

    if args.log_console:
        os.environ["LLGRAPH_LOG_CONSOLE"] = "1"

    effective_log = setup_search_logging(
        workspace,
        cli_override=args.log_level,
    )
    if args.log_level or args.log_console:
        where = "终端+search.log" if args.log_console or os.environ.get("LLGRAPH_LOG_CONSOLE") == "1" else "仅 search.log"
        print(
            f"[llgraph] 向量检索日志级别: {level_name(effective_log)}（{where}）",
            file=sys.stderr,
            flush=True,
        )

    if args.model:
        set_runtime_model(args.model.strip())
        print(
            f"[llgraph] 使用模型: {resolve_effective_model(workspace)}",
            file=sys.stderr,
            flush=True,
        )

    if args.no_survey:
        from llgraph.config.survey_settings import set_survey_cli_disabled

        set_survey_cli_disabled(True)
        print("[llgraph] Survey 交互已禁用（--no-survey）", file=sys.stderr, flush=True)

    if args.init_config or args.init_config_force:
        try:
            copied = init_workspace_llgraph(
                workspace,
                force=args.init_config_force,
            )
        except FileNotFoundError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            sys.exit(1)
        if copied:
            print(f"已写入 {len(copied)} 个文件到 {workspace}/.llgraph/")
            for rel in copied:
                print(f"  {rel}")
        else:
            print(
                f"{workspace}/.llgraph/ 已存在，未覆盖（使用 --init-config-force 强制覆盖）",
                flush=True,
            )
        if args.message is None and not args.once and not (
            args.init_user_config or args.init_user_config_force
        ):
            return

    if args.init_user_config or args.init_user_config_force:
        try:
            user_copied = init_user_llgraph(force=args.init_user_config_force)
        except FileNotFoundError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            sys.exit(1)
        if user_copied:
            print(f"已写入 {len(user_copied)} 个用户配置文件:")
            for rel in user_copied:
                print(f"  {rel}")
        else:
            print(
                "~/.llgraph/ 已存在，未覆盖（使用 --init-user-config-force 强制覆盖）",
                flush=True,
            )
        if args.message is None and not args.once and not (
            args.init_config or args.init_config_force
        ):
            return

    initial_mode = parse_trace_mode(args.trace) or TraceMode.STEPS
    trace_session = TraceSession(
        mode=initial_mode,
        preview_lines=max(1, args.preview_lines),
    )
    allow_write = args.write
    if args.sandbox and args.no_sandbox:
        print("错误: 不能同时指定 --sandbox 与 --no-sandbox", file=sys.stderr)
        sys.exit(1)

    from llgraph.config.sandbox_settings import resolve_sandbox_settings
    from llgraph.sandbox.policy import build_sandbox_policy

    cli_sandbox: bool | None = True if args.sandbox else False if args.no_sandbox else None
    sandbox_settings = resolve_sandbox_settings(workspace)
    sandbox_policy = build_sandbox_policy(
        workspace, sandbox_settings, cli_enabled=cli_sandbox
    )
    sandbox_warning = sandbox_policy.startup_warning()
    if sandbox_policy.active and sandbox_warning:
        print(f"警告: {sandbox_warning}", file=sys.stderr)
    elif sandbox_policy.enabled:
        print(
            f"沙箱已启用 ({sandbox_policy.backend}, mode={sandbox_policy.mode}, "
            f"network={sandbox_policy.network})",
            flush=True,
        )

    context_session = ContextSession()

    from llgraph.code_index.index_watch import (
        attach_watch_shutdown,
        start_index_watch_with_agent,
    )

    watch_service = start_index_watch_with_agent(
        workspace,
        no_watch=args.no_watch_index,
    )
    attach_watch_shutdown(watch_service)

    from llgraph.context.context_spill import ContextSpill

    edit_tracker: SessionEditTracker | None = None
    write_failure_tracker: WriteFailureTracker | None = None
    mcp_registry = None
    mcp_summary = ""
    context_spill: ContextSpill | None = None
    agent_session: AgentSessionContext | None = None
    try:
        from llgraph.core.tools import load_mcp_tool_bundle

        mcp_tools, mcp_registry, mcp_summary = load_mcp_tool_bundle(
            workspace,
            allow_write=allow_write and not sandbox_policy.enabled,
        )
        from llgraph.session.session_web_search import resolve_initial_web_search_enabled

        web_search_enabled = resolve_initial_web_search_enabled(workspace)
        if args.once:
            thread_id = args.thread_id or "default"
            context_spill = ContextSpill.create(
                workspace,
                session_id=thread_id,
                disabled=args.no_spill,
            )
            edit_tracker = SessionEditTracker(workspace, session_id=thread_id)
            edit_settings = resolve_edit_settings(workspace)
            write_failure_tracker = (
                WriteFailureTracker(
                    context_session,
                    failures_before_hint=edit_settings.write_failures_before_hint,
                    chunk_max_chars=edit_settings.write_chunk_max_chars,
                )
                if allow_write
                else None
            )

            def _on_changed(rel: str) -> None:
                if watch_service is not None:
                    watch_service.notify_changed(rel)

            from llgraph.survey.edit_confirm import EditConfirmGate

            edit_confirm_gate = (
                EditConfirmGate(workspace) if allow_write else None
            )
            agent = build_agent(
                with_memory=args.memory,
                workspace_root=workspace,
                allow_write=allow_write,
                edit_tracker=edit_tracker if allow_write else None,
                on_file_changed=_on_changed if allow_write else None,
                mcp_tools=mcp_tools,
                context_spill=context_spill,
                write_failure_tracker=write_failure_tracker,
                web_search_enabled=web_search_enabled,
                edit_confirm_gate=edit_confirm_gate,
                context_session=context_session,
                sandbox_policy=sandbox_policy,
            )
            if not args.message:
                print(
                    "错误: --once 模式需要提供消息，例如: llgraph --once \"你好\"",
                    file=sys.stderr,
                )
                sys.exit(1)
            _run_once(
                agent,
                args.message,
                workspace=workspace,
                thread_id=thread_id,
                with_memory=args.memory,
                trace_session=trace_session,
                context_session=context_session,
                allow_write=allow_write,
                write_failure_tracker=write_failure_tracker,
                ui_mode=ui_mode,
            )
            return

        explicit_thread = (args.thread_id or "").strip()
        thread_id = explicit_thread or f"cli-{uuid.uuid4().hex[:8]}"
        from llgraph.core.checkpointer_factory import checkpointer_kind
        from llgraph.session.session_registry import session_is_resumable

        resume_hint = ""
        if explicit_thread:
            _ok, resume_hint = session_is_resumable(workspace, thread_id)
        memory_kind = checkpointer_kind(workspace, with_memory=True)
        context_spill = ContextSpill.create(
            workspace,
            session_id=thread_id,
            disabled=args.no_spill,
        )
        edit_tracker = SessionEditTracker(workspace, session_id=thread_id)
        edit_settings = resolve_edit_settings(workspace)
        write_failure_tracker = (
            WriteFailureTracker(
                context_session,
                failures_before_hint=edit_settings.write_failures_before_hint,
                chunk_max_chars=edit_settings.write_chunk_max_chars,
            )
            if allow_write
            else None
        )

        def on_file_changed(rel: str) -> None:
            sess = agent_session
            if (
                sess is not None
                and sess.watch_service is not None
                and sess.watch_service.active
            ):
                sess.watch_service.notify_changed(rel)

        from llgraph.survey.edit_confirm import EditConfirmGate

        edit_confirm_gate = EditConfirmGate(workspace) if allow_write else None
        agent = build_agent(
            with_memory=True,
            workspace_root=workspace,
            allow_write=allow_write,
            edit_tracker=edit_tracker if allow_write else None,
            on_file_changed=on_file_changed if allow_write else None,
            mcp_tools=mcp_tools,
            context_spill=context_spill,
            write_failure_tracker=write_failure_tracker,
            web_search_enabled=web_search_enabled,
            edit_confirm_gate=edit_confirm_gate,
            context_session=context_session,
            sandbox_policy=sandbox_policy,
        )
        agent_session = AgentSessionContext(
            agent=agent,
            workspace=workspace,
            thread_id=thread_id,
            trace_session=trace_session,
            context_session=context_session,
            with_memory=True,
            edit_tracker=edit_tracker,
            context_spill=context_spill,
            write_failure_tracker=write_failure_tracker,
            allow_write=allow_write,
            mcp_tools=mcp_tools,
            mcp_registry=mcp_registry,
            on_file_changed=on_file_changed,
            watch_service=watch_service,
            web_search_enabled=web_search_enabled,
            edit_confirm_gate=edit_confirm_gate,
            sandbox_policy=sandbox_policy,
        )
        from llgraph.session.session_file_store import restore_session_to_agent

        if explicit_thread:
            from llgraph.session.session_manifest import sync_session_manifest_to_agent_state

            sync_session_manifest_to_agent_state(
                agent,
                thread_id=thread_id,
                workspace=workspace,
                session=context_session,
                user_message="",
                with_memory=True,
            )
            msg_count = restore_session_to_agent(agent, workspace, thread_id)
            if msg_count > 1:
                suffix = f"已加载 {msg_count} 条历史消息（messages.jsonl）。"
                resume_hint = f"{resume_hint} {suffix}" if resume_hint else suffix
        _run_interactive(
            agent,
            thread_id=thread_id,
            workspace=workspace,
            allow_write=allow_write,
            trace_session=trace_session,
            context_session=context_session,
            edit_tracker=edit_tracker,
            watch_active=watch_service is not None and watch_service.active,
            web_search_enabled=web_search_enabled,
            mcp_summary=mcp_summary,
            agent_session=agent_session,
            opening_message=args.message,
            write_failure_tracker=write_failure_tracker,
            resume_hint=resume_hint,
            memory_kind=memory_kind,
            ui_mode=ui_mode,
        )
    except RuntimeError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        stop_watch = (
            agent_session.watch_service
            if agent_session is not None
            else watch_service
        )
        if stop_watch is not None:
            stop_watch.stop()
        if mcp_registry is not None:
            mcp_registry.stop()
        if edit_tracker is not None:
            summary = edit_tracker.exit_summary()
            if summary:
                print(summary)


if __name__ == "__main__":
    main()
