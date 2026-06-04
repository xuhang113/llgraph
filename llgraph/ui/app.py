"""Textual 主应用：Claude Code 风格聊天交互。"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog, Static

from llgraph.core.agent_session import AgentSessionContext
from llgraph.context.context_session import ContextSession
from llgraph.session.session_edits import SessionEditTracker
from llgraph.display.trace_display import TraceSession, print_trace_step_detail
from llgraph.ui.banner import build_session_banner_text
from llgraph.ui.context import set_ui_app
from llgraph.ui.keys import MSG_GOODBYE, MSG_INTERRUPT_EXIT, is_exit_command
from llgraph.ui.prompts import MenuModal, MenuOption, SurveyModal
from llgraph.ui.sink import strip_ansi
from llgraph.core.write_failure_tracker import WriteFailureTracker

if True:
    from llgraph.display.trace_display import TraceStepRecord


class LogLine(Message):
    """过程日志（dim）。"""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StreamChunk(Message):
    """助手流式文本。"""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StreamEnd(Message):
    """流式结束。"""


class StepAdded(Message):
    """新步骤。"""

    def __init__(self, step: TraceStepRecord) -> None:
        super().__init__()
        self.step = step


class TurnFinished(Message):
    """一轮结束。"""

    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply


class TurnFailed(Message):
    """一轮失败。"""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class ClearSteps(Message):
    """清空步骤侧栏。"""


class TuiTraceSink:
    """追踪输出：里程碑进对话区，详情可展开。"""

    _MILESTONE_MARKERS = ("▶", "✓", "▼", "用户消息", "思考中", "本轮", "token 合计")

    def __init__(self, app: LlgraphApp) -> None:
        self._app = app

    def line(self, text: str) -> None:
        plain = strip_ansi(text).strip()
        if not plain:
            return
        session = self._app.trace_session
        if session.is_verbose() or any(m in plain for m in self._MILESTONE_MARKERS):
            self._app.post_message(LogLine(plain))

    def stream(self, text: str) -> None:
        if text:
            self._app.post_message(StreamChunk(text))

    def stream_end(self) -> None:
        self._app.post_message(StreamEnd())

    def step_added(self, step: TraceStepRecord) -> None:
        self._app.post_message(StepAdded(step))

    def step_selected(self, step_id: int) -> None:
        print_trace_step_detail(self._app.trace_session, str(step_id))


@dataclass
class TuiSessionParams:
    """启动 TUI 所需上下文。"""

    agent: Any
    workspace: Path
    thread_id: str
    trace_session: TraceSession
    context_session: ContextSession
    allow_write: bool
    agent_session: AgentSessionContext | None = None
    edit_tracker: SessionEditTracker | None = None
    write_failure_tracker: WriteFailureTracker | None = None
    watch_active: bool = False
    web_search_enabled: bool = False
    mcp_summary: str = ""
    resume_hint: str = ""
    memory_kind: str = ""
    opening_message: str | None = None
    single_turn: bool = False


class LlgraphApp(App[None]):
    """llgraph · Claude 风格主界面。"""

    TITLE = "llgraph"
    SUB_TITLE = "LangGraph Agent"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        layout: vertical;
    }

    #session_bar {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }

    #body {
        layout: horizontal;
        height: 1fr;
        min-height: 12;
    }

    #steps_panel {
        layout: vertical;
        width: 0;
        min-width: 0;
        max-width: 0;
        overflow: hidden;
        border-right: none;
        display: none;
    }

    #steps_panel.-show {
        display: block;
        width: 30;
        min-width: 26;
        max-width: 36;
        border-right: solid $primary-darken-2;
        background: $panel;
    }

    #steps_title {
        height: 1;
        padding: 0 1;
        background: $primary-darken-3;
    }

    #steps {
        height: 1fr;
    }

    #chat_column {
        layout: vertical;
        width: 1fr;
        height: 1fr;
        min-height: 8;
    }

    #chat {
        width: 100%;
        height: 1fr;
        min-height: 8;
        padding: 0 1;
        border: none;
        background: $background;
    }

    #status_bar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }

    #input_bar {
        layout: horizontal;
        height: 3;
        min-height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary-darken-1;
        align: left middle;
    }

    #prompt_prefix {
        width: 2;
        color: $accent;
        text-style: bold;
    }

    #prompt_input {
        width: 1fr;
        height: 3;
        border: none;
        padding: 0 1;
    }

    #prompt_input.-disabled {
        opacity: 0.5;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", show=True),
        Binding("ctrl+q", "quit", "退出", show=True),
        Binding("ctrl+o", "toggle_steps", "步骤侧栏", show=True),
    ]

    def __init__(self, params: TuiSessionParams) -> None:
        super().__init__()
        self._params = params
        self.trace_session = params.trace_session
        self.trace_session.trace_sink = TuiTraceSink(self)
        self._running = False
        self._last_user_message = ""
        self._steps_visible = False
        self._assistant_stream_open = False

    def compose(self) -> ComposeResult:
        session_line = build_session_banner_text(
            workspace=self._params.workspace,
            allow_write=self._params.allow_write,
            thread_id=self._params.thread_id,
            trace_session=self.trace_session,
            watch_active=self._params.watch_active,
            web_search_enabled=self._params.web_search_enabled,
            mcp_summary=self._params.mcp_summary,
            resume_hint=self._params.resume_hint,
            memory_kind=self._params.memory_kind,
        )
        first_line = session_line.split("\n", 1)[0]
        yield Header(show_clock=False)
        yield Static(first_line, id="session_bar", markup=True)
        with Horizontal(id="body"):
            with Vertical(id="steps_panel"):
                yield Static("步骤", id="steps_title")
                yield ListView(id="steps")
            with Vertical(id="chat_column"):
                yield RichLog(
                    id="chat",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    auto_scroll=True,
                )
        yield Static("", id="status_bar", markup=True)
        with Horizontal(id="input_bar"):
            yield Static("❯", id="prompt_prefix")
            yield Input(
                placeholder="询问 llgraph…  /help  /trace  exit 退出",
                id="prompt_input",
            )
        yield Footer()

    def on_mount(self) -> None:
        set_ui_app(self)
        self.call_after_refresh(self._deferred_startup)

    def _deferred_startup(self) -> None:
        """布局完成后再写入对话区（RichLog 需已知尺寸）。"""
        self._write_welcome()
        self.query_one("#prompt_input", Input).focus()
        if self._params.opening_message:
            self._submit_message(self._params.opening_message)

    def on_unmount(self) -> None:
        set_ui_app(None)

    def _chat(self) -> RichLog:
        return self.query_one("#chat", RichLog)

    def _write_welcome(self) -> None:
        """Claude 风格欢迎块。"""
        mode = "可读写" if self._params.allow_write else "只读"
        ws = str(self._params.workspace)
        if len(ws) > 50:
            ws = "…" + ws[-47:]
        self.write_chat_block(
            "[bold]llgraph[/] · LangGraph Agent\n"
            f"{ws} · {mode}\n"
            "[dim]输入问题 Enter 发送 · /help 帮助 · Ctrl+O 步骤侧栏[/]\n"
        )

    def write_chat_line(self, text: str, *, dim: bool = False) -> None:
        """
        向对话区追加一行（/help 等用正常色，过程用 dim）。

        @param text 文本
        @param dim 是否弱化显示
        """
        if not text.strip():
            return
        if dim:
            self._chat().write(f"[dim]{text}[/]", scroll_end=True)
        else:
            self._chat().write(text, scroll_end=True)

    def write_chat_block(self, text: str, *, dim: bool = False) -> None:
        """
        向对话区追加多行文本。

        @param text 多行内容
        @param dim 是否弱化
        """
        for line in text.splitlines():
            if line.strip():
                self.write_chat_line(line, dim=dim)
            else:
                self._chat().write(" ", scroll_end=True)

    def _chat_user(self, text: str) -> None:
        """
        用户消息（Claude 绿标风格）。

        @param text 用户输入
        """
        self._chat().write("")
        self._chat().write(f"[bold cyan]❯ {text}[/]")

    def _chat_process(self, text: str) -> None:
        """
        过程行（缩进 dim）。

        @param text 过程文本
        """
        self._chat().write(f"[dim]  {text}[/]")

    def _set_running(self, running: bool) -> None:
        """
        运行态：禁用输入、状态栏提示。

        @param running 是否运行中
        """
        self._running = running
        inp = self.query_one("#prompt_input", Input)
        inp.disabled = running
        if running:
            inp.add_class("-disabled")
            self.query_one("#status_bar", Static).update(
                "[yellow]● 思考中…[/] [dim]Esc 无法中断本轮，请等待完成[/]"
            )
        else:
            inp.remove_class("-disabled")
            self.query_one("#status_bar", Static).update("")
            inp.focus()

    def log_line(self, text: str) -> None:
        self.write_chat_line(text, dim=False)

    def on_log_line(self, message: LogLine) -> None:
        self.log_line(message.text)

    def on_stream_chunk(self, message: StreamChunk) -> None:
        if not self._assistant_stream_open:
            self._assistant_stream_open = True
            self._chat().write("")
            self._chat().write("[bold]●[/] ")
        self._chat().write(message.text, scroll_end=True)

    def on_stream_end(self, _message: StreamEnd) -> None:
        if self._assistant_stream_open:
            self._chat().write("")
            self._assistant_stream_open = False

    def on_clear_steps(self, _message: ClearSteps) -> None:
        self.query_one("#steps", ListView).clear()

    def on_step_added(self, message: StepAdded) -> None:
        step = message.step
        label = f"#{step.step_id} {step.title}"
        self.query_one("#steps", ListView).append(ListItem(Label(label)))
        if not self._steps_visible and self.trace_session.shows_process():
            self.action_toggle_steps()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "steps":
            return
        event.stop()
        steps = self.trace_session.last_turn_steps
        if event.list_view.index < 0 or event.list_view.index >= len(steps):
            return
        step = steps[event.list_view.index]
        print_trace_step_detail(self.trace_session, str(step.step_id))

    def action_toggle_steps(self) -> None:
        """显示/隐藏步骤侧栏。"""
        panel = self.query_one("#steps_panel")
        self._steps_visible = not self._steps_visible
        if self._steps_visible:
            panel.add_class("-show")
        else:
            panel.remove_class("-show")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt_input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if is_exit_command(text):
            self.exit(MSG_GOODBYE)
            return
        if self._running:
            return
        self._submit_message(text)

    def _submit_message(self, text: str) -> None:
        self._last_user_message = text
        if text.startswith("/"):
            self._chat_user(text)
            self._handle_meta(text)
            return

        self._chat_user(text)
        effective = text
        ctx = self._params.context_session
        if ctx is not None:
            from llgraph.survey.survey_prompt import maybe_preflight_survey_for_user_message

            preflight = maybe_preflight_survey_for_user_message(
                text,
                preflight_done=ctx.organize_preflight_done,
                workspace=self._params.workspace,
                context_session=ctx,
            )
            if preflight is not None:
                effective, ctx.organize_preflight_done = preflight
                self._chat_user(f"[dim]（确认后）[/] {effective}")

        self._set_running(True)
        self._assistant_stream_open = False
        self.post_message(ClearSteps())
        self._run_agent_turn(effective)

    def _active_thread_id(self) -> str:
        if self._params.agent_session is not None:
            return self._params.agent_session.thread_id
        return self._params.thread_id

    @contextmanager
    def _capture_stdout(self):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            yield buf
        finally:
            sys.stdout = old
            for line in buf.getvalue().splitlines():
                self.write_chat_line(line, dim=False)

    def _handle_meta(self, text: str) -> None:
        from llgraph.commands.meta_commands import handle_meta_command

        with self._capture_stdout():
            handled = handle_meta_command(
                text,
                workspace=self._params.workspace,
                trace_session=self.trace_session,
                context_session=self._params.context_session,
                allow_write=self._params.allow_write,
                last_user_message=self._last_user_message,
                edit_tracker=self._params.edit_tracker,
                agent_session=self._params.agent_session,
                mcp_summary=self._params.mcp_summary,
            )
        if handled and self._params.agent_session is not None:
            self._params.agent = self._params.agent_session.agent

    @work(thread=True, exclusive=True)
    def _run_agent_turn(self, user_input: str) -> None:
        try:
            agent = (
                self._params.agent_session.agent
                if self._params.agent_session is not None
                else self._params.agent
            )
            wft = (
                self._params.agent_session.write_failure_tracker
                if self._params.agent_session is not None
                else self._params.write_failure_tracker
            )
            spill = (
                self._params.agent_session.context_spill
                if self._params.agent_session is not None
                else None
            )
            ctx = self._params.context_session
            buf = io.StringIO()
            old_out = sys.stdout
            sys.stdout = buf
            try:
                from llgraph.core.agent import invoke_agent

                reply = invoke_agent(
                    agent,
                    user_input,
                    workspace_root=self._params.workspace,
                    thread_id=self._active_thread_id(),
                    with_memory=True,
                    trace_session=self.trace_session,
                    context_session=ctx,
                    write_failure_tracker=wft,
                    context_spill=spill,
                )
            finally:
                sys.stdout = old_out
            for line in buf.getvalue().splitlines():
                if line.strip():
                    self.call_from_thread(
                        self.post_message,
                        LogLine(line),
                    )
            self.call_from_thread(self.post_message, TurnFinished(reply))
        except KeyboardInterrupt:
            self.call_from_thread(self.post_message, TurnFailed(MSG_INTERRUPT_EXIT))
        except Exception as exc:
            self.call_from_thread(self.post_message, TurnFailed(str(exc)))

    def on_turn_finished(self, message: TurnFinished) -> None:
        self.trace_session.trace_sink.stream_end()
        self._set_running(False)
        self._maybe_survey_followup(message.reply)
        if self._params.single_turn:
            self.exit()

    def on_turn_failed(self, message: TurnFailed) -> None:
        self._set_running(False)
        self._chat().write(f"[red]● {message.error}[/]")

    def _maybe_survey_followup(self, assistant_text: str) -> None:
        from llgraph.survey.survey_prompt import try_run_survey_followup

        followup = try_run_survey_followup(
            assistant_text,
            workspace=self._params.workspace,
            context_session=self._params.context_session,
        )
        if not followup:
            return
        self._chat_process("▶ 正在将确认结果提交给 Agent…")
        self._set_running(True)
        self._assistant_stream_open = False
        self.post_message(ClearSteps())
        self._run_agent_turn(followup)

    def push_screen_wait_menu(
        self,
        title: str,
        options: list[MenuOption],
        *,
        default_index: int = 0,
    ) -> int | None:
        return self.call_from_thread(
            self.push_screen_wait,
            MenuModal(title, options, default_index=default_index),
        )

    def push_screen_wait_survey(self, spec) -> dict[str, str] | None:
        return self.call_from_thread(self.push_screen_wait, SurveyModal(spec))


def run_tui_session(params: TuiSessionParams) -> None:
    """
    启动 Textual 会话。

    @param params 会话参数
    """
    app = LlgraphApp(params)
    app.run()
