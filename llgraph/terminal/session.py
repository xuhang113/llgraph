"""经典终端交互会话。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llgraph.context.context_session import ContextSession
from llgraph.session.session_edits import SessionEditTracker
from llgraph.display.trace_display import TraceSession
from llgraph.terminal.banner import print_terminal_session_banner
from llgraph.terminal.keys import MSG_GOODBYE, MSG_INTERRUPT_EXIT, is_exit_command
from llgraph.terminal.output import emit, emit_error, emit_milestone, emit_ok, emit_warn, write_dialog_line
from llgraph.display.trace_sink import StdoutTraceSink
from llgraph.core.write_failure_tracker import WriteFailureTracker


@dataclass
class TerminalSessionParams:
    """终端会话参数。"""

    agent: Any
    workspace: Path
    thread_id: str
    trace_session: TraceSession
    context_session: ContextSession
    allow_write: bool
    agent_session: Any | None = None
    edit_tracker: SessionEditTracker | None = None
    write_failure_tracker: WriteFailureTracker | None = None
    watch_active: bool = False
    web_search_enabled: bool = False
    mcp_summary: str = ""
    resume_hint: str = ""
    memory_kind: str = ""
    opening_message: str | None = None
    single_turn: bool = False
    with_memory: bool = True


def _active_thread_id(params: TerminalSessionParams) -> str:
    if params.agent_session is not None:
        return params.agent_session.thread_id
    return params.thread_id


def _try_expand_trace_step(params: TerminalSessionParams, text: str) -> bool:
    """
    纯数字或 #N：展开对应步骤详情。

    @param params 会话参数
    @param text 用户输入
    @return 是否已处理
    """
    import re

    matched = re.fullmatch(r"#?(\d+)", text.strip())
    if not matched:
        return False
    from llgraph.display.trace_display import print_trace_step_detail

    print_trace_step_detail(params.trace_session, matched.group(1))
    return True


def _run_turn(params: TerminalSessionParams, user_input: str) -> str | None:
    """
    执行一轮 Agent。

    @param params 会话参数
    @param user_input 用户消息
    @return 助手回复；失败返回 None
    """
    agent = (
        params.agent_session.agent
        if params.agent_session is not None
        else params.agent
    )
    wft = (
        params.agent_session.write_failure_tracker
        if params.agent_session is not None
        else params.write_failure_tracker
    )
    spill = (
        params.agent_session.context_spill
        if params.agent_session is not None
        else None
    )
    from llgraph.core.agent import invoke_agent

    try:
        allow_write = (
            params.agent_session.allow_write
            if params.agent_session is not None
            else params.allow_write
        )
        return invoke_agent(
            agent,
            user_input,
            workspace_root=params.workspace,
            thread_id=_active_thread_id(params),
            with_memory=params.with_memory,
            trace_session=params.trace_session,
            context_session=params.context_session,
            write_failure_tracker=wft,
            context_spill=spill,
            allow_write=allow_write,
        )
    except KeyboardInterrupt:
        emit(f"\n{MSG_INTERRUPT_EXIT}", colorize=True)
        return None
    except Exception as exc:
        emit_error(f"错误: {exc}")
        return None


def _handle_meta(params: TerminalSessionParams, text: str, *, last_user: str) -> bool:
    """
    处理元命令。

    @param params 会话参数
    @param text 用户输入
    @param last_user 上一条用户消息
    @return 是否已处理
    """
    from llgraph.commands.meta_commands import handle_meta_command

    return handle_meta_command(
        text,
        workspace=params.workspace,
        trace_session=params.trace_session,
        context_session=params.context_session,
        allow_write=params.allow_write,
        last_user_message=last_user,
        edit_tracker=params.edit_tracker,
        agent_session=params.agent_session,
        mcp_summary=params.mcp_summary,
    )


def _maybe_survey_followup(
    params: TerminalSessionParams,
    assistant_text: str,
    *,
    raw_assistant_text: str | None = None,
) -> None:
    """
    问卷确认后自动续聊。

    @param params 会话参数
    @param assistant_text 展示用助手回复
    @param raw_assistant_text 含 survey 块的原文（解析用）
    """
    from llgraph.survey.survey_prompt import resolve_survey_from_assistant, try_run_survey_followup
    from llgraph.config.survey_settings import survey_followup_enabled

    if not survey_followup_enabled(params.workspace, params.context_session):
        return
    parse_text = (raw_assistant_text or assistant_text or "").strip()
    if not parse_text:
        return
    if resolve_survey_from_assistant(parse_text) is None:
        return
    emit_milestone("检测到确认问卷，请在下方菜单中选择（Esc 取消）…")
    allow_write = (
        params.agent_session.allow_write
        if params.agent_session is not None
        else params.allow_write
    )
    followup = try_run_survey_followup(
        parse_text,
        workspace=params.workspace,
        context_session=params.context_session,
        allow_write=allow_write,
    )
    if not followup:
        return
    emit_milestone("正在将确认结果提交给 Agent…")
    _run_turn(params, followup)


def _process_user_message(params: TerminalSessionParams, text: str) -> bool:
    """
    处理一条用户输入。

    @param params 会话参数
    @param text 用户消息
    @return False 表示应退出会话
    """
    if is_exit_command(text):
        emit_ok(MSG_GOODBYE)
        return False

    last_user = text

    if _try_expand_trace_step(params, text):
        return True

    from llgraph.commands.meta_commands import is_registered_meta_command

    if is_registered_meta_command(text, params.workspace):
        from llgraph.terminal.style import sty

        preview = text.split("\n", 1)[0]
        emit(sty(f"❯ {preview}", "prompt"))
        handled = _handle_meta(params, text, last_user=last_user)
        if handled and params.agent_session is not None:
            params.agent = params.agent_session.agent
        elif not handled:
            emit_warn(f"未知命令 {preview}，输入 /help 查看。")
        return True

    reply = _run_turn(params, text)
    if reply is None:
        return True
    if params.agent_session is not None:
        params.agent = params.agent_session.agent
    _maybe_survey_followup(
        params,
        reply,
        raw_assistant_text=params.trace_session.last_turn_raw_reply or reply,
    )
    return True


from llgraph.session.mode_switch import SessionModeTransition


def run_terminal_session(params: TerminalSessionParams) -> SessionModeTransition | None:
    """
    启动经典终端交互循环。

    @param params 会话参数
    @return 模式切换请求；正常 exit 返回 None
    """
    from llgraph.terminal.input_reader import (
        init_input_history,
        read_interactive_user_message,
        save_input_history,
    )

    params.trace_session.trace_sink = StdoutTraceSink()
    init_input_history()

    print_terminal_session_banner(
        workspace=params.workspace,
        allow_write=params.allow_write,
        thread_id=_active_thread_id(params),
        trace_session=params.trace_session,
        watch_active=params.watch_active,
        web_search_enabled=params.web_search_enabled,
        mcp_summary=params.mcp_summary,
        resume_hint=params.resume_hint,
        memory_kind=params.memory_kind,
    )
    emit()

    def _maybe_mode_switch() -> SessionModeTransition | None:
        if params.agent_session is not None and params.agent_session.mode_switch is not None:
            transition = params.agent_session.mode_switch
            params.agent_session.mode_switch = None
            return transition
        return None

    if params.opening_message:
        if not _process_user_message(params, params.opening_message.strip()):
            save_input_history()
            return None
        sw = _maybe_mode_switch()
        if sw is not None:
            save_input_history()
            return sw
        if params.single_turn:
            save_input_history()
            return None

    transition: SessionModeTransition | None = None
    try:
        while True:
            try:
                user_input = read_interactive_user_message(params.workspace).strip()
            except KeyboardInterrupt:
                emit("\n[已取消输入]", colorize=True)
                continue
            except EOFError:
                emit(f"\n{MSG_GOODBYE}", colorize=True)
                break
            if not user_input:
                continue
            if not _process_user_message(params, user_input):
                break
            sw = _maybe_mode_switch()
            if sw is not None:
                transition = sw
                break
    finally:
        save_input_history()

    if transition is not None:
        return transition

    from llgraph.session.session_switch import print_session_exit_hint

    print_session_exit_hint(params.workspace, _active_thread_id(params))
    return None
