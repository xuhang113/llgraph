"""TUI 模态：Claude 风格菜单与 Survey 向导。"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from llgraph.survey.survey_prompt import SurveySpec


@dataclass(frozen=True)
class MenuOption:
    """菜单项。"""

    label: str
    hint: str = ""


class MenuModal(ModalScreen[int | None]):
    """写确认等单选（Claude 风格列表）。"""

    DEFAULT_CSS = """
    MenuModal {
        align: center middle;
    }
    #menu_frame {
        width: 72;
        height: auto;
        max-height: 80%;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    #menu_title {
        margin-bottom: 1;
        text-style: bold;
    }
    #menu_list {
        height: auto;
        max-height: 20;
        border: none;
    }
    #menu_footer {
        margin-top: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
        Binding("up", "cursor_up", "上", show=False),
        Binding("down", "cursor_down", "下", show=False),
    ]

    def __init__(
        self,
        title: str,
        options: list[MenuOption],
        *,
        default_index: int = 0,
    ) -> None:
        super().__init__()
        self._title = title
        self._options = options
        self._default_index = default_index

    def compose(self) -> ComposeResult:
        with Container(id="menu_frame"):
            yield Static(self._title, id="menu_title")
            items = [
                ListItem(
                    Label(f"{opt.label}  ({opt.hint})" if opt.hint else opt.label)
                )
                for opt in self._options
            ]
            yield ListView(*items, id="menu_list")
            yield Static(
                "Enter 确认 · ↑↓ 移动 · Esc 取消",
                id="menu_footer",
            )

    def on_mount(self) -> None:
        lst = self.query_one("#menu_list", ListView)
        lst.focus()
        if self._options:
            lst.index = min(self._default_index, len(self._options) - 1)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        self.dismiss(event.list_view.index)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SurveyModal(ModalScreen[dict[str, str] | None]):
    """多步确认向导（对齐 Claude Code Survey UI）。"""

    DEFAULT_CSS = """
    SurveyModal {
        align: center middle;
    }
    #survey_frame {
        width: 80;
        height: auto;
        max-height: 85%;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    #survey_title {
        text-style: bold;
        margin-bottom: 1;
    }
    #survey_progress {
        color: $accent;
        margin-bottom: 1;
    }
    #survey_prompt {
        margin-bottom: 1;
        text-style: bold;
    }
    #survey_options {
        height: auto;
        max-height: 16;
        border: none;
        margin-bottom: 1;
    }
    #survey_footer {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
        Binding("tab", "next_tab", "下一题"),
        Binding("shift+tab", "prev_tab", "上一题"),
        Binding("up", "cursor_up", "上", show=False),
        Binding("down", "cursor_down", "下", show=False),
    ]

    def __init__(self, spec: SurveySpec) -> None:
        super().__init__()
        self._spec = spec
        self._step = 0
        self._option_index = 0
        self._selections: list[int] = [q.default_index for q in spec.questions]
        self._answers: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Container(id="survey_frame"):
            yield Static(self._spec.title, id="survey_title")
            yield Static("", id="survey_progress")
            yield Static("", id="survey_prompt")
            yield ListView(id="survey_options")
            yield Static(
                "Enter 确认 · Tab/→ 下一题 · ← 上一题 · ↑↓ 移动 · Esc 取消",
                id="survey_footer",
            )

    def _step_labels(self) -> list[str]:
        labels: list[str] = []
        for question in self._spec.questions:
            label = getattr(question, "step_label", "") or question.prompt[:10]
            labels.append(label)
        labels.append("Submit")
        return labels

    def _refresh_view(self) -> None:
        labels = self._step_labels()
        bar_parts: list[str] = []
        for idx, name in enumerate(labels):
            if idx == self._step:
                bar_parts.append(f"[bold reverse] ▣ {name} [/]")
            elif idx < len(self._spec.questions) and (
                self._spec.questions[idx].question_id in self._answers
            ):
                bar_parts.append(f"[green] ✓ {name} [/]")
            else:
                bar_parts.append(f"   {name} ")
        self.query_one("#survey_progress", Static).update("  ·  ".join(bar_parts))

        lst = self.query_one("#survey_options", ListView)
        if self._step >= len(self._spec.questions):
            self.query_one("#survey_prompt", Static).update("确认并提交？")
            lst.clear()
            for question in self._spec.questions:
                key = question.question_id
                val = self._answers.get(key, "—")
                lst.append(ListItem(Label(f"· {key}: {val}")))
            if lst.children:
                lst.index = 0
            return

        question = self._spec.questions[self._step]
        self.query_one("#survey_prompt", Static).update(question.prompt)
        lst.clear()
        hints = getattr(question, "option_hints", None)
        for idx, opt in enumerate(question.options):
            hint = ""
            if isinstance(hints, tuple) and idx < len(hints) and hints[idx]:
                hint = f"\n    [dim]{hints[idx]}[/]"
            lst.append(ListItem(Label(f"{idx + 1}. {opt}{hint}")))
        if lst.children:
            lst.index = min(self._option_index, len(lst.children) - 1)

    def on_mount(self) -> None:
        self._refresh_view()
        self.query_one("#survey_options", ListView).focus()

    def action_next_tab(self) -> None:
        if self._step < len(self._spec.questions):
            self._commit_current_option()
            self._step = min(self._step + 1, len(self._spec.questions))
            if self._step < len(self._spec.questions):
                self._option_index = self._selections[self._step]
        self._refresh_view()

    def action_prev_tab(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._option_index = self._selections[self._step]
            self._refresh_view()

    def _commit_current_option(self) -> None:
        if self._step >= len(self._spec.questions):
            return
        question = self._spec.questions[self._step]
        self._selections[self._step] = self._option_index
        self._answers[question.question_id] = question.options[self._option_index]

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        lst = event.list_view
        if self._step >= len(self._spec.questions):
            self._finish()
            return
        self._option_index = lst.index
        question = self._spec.questions[self._step]
        choice = question.options[self._option_index]
        if question.allow_free_text and (
            "其他" in choice or "手动" in choice or "type" in choice.lower()
        ):
            self.app.push_screen(
                _FreeTextModal(
                    question.prompt,
                    on_done=lambda text: self._apply_free_text(text, question),
                )
            )
            return
        self._selections[self._step] = self._option_index
        self._answers[question.question_id] = choice
        self._step += 1
        if self._step < len(self._spec.questions):
            self._option_index = self._selections[self._step]
        self._refresh_view()

    def _apply_free_text(self, text: str, question) -> None:
        base = question.options[self._option_index]
        self._answers[question.question_id] = text.strip() or base
        self._selections[self._step] = self._option_index
        self._step += 1
        if self._step < len(self._spec.questions):
            self._option_index = self._selections[self._step]
        self._refresh_view()

    def _finish(self) -> None:
        for idx, question in enumerate(self._spec.questions):
            if question.question_id not in self._answers:
                self._answers[question.question_id] = question.options[
                    self._selections[idx]
                ]
        self.dismiss(self._answers)

    def action_cancel(self) -> None:
        self.dismiss(None)


class _FreeTextModal(ModalScreen[None]):
    """补充说明输入。"""

    DEFAULT_CSS = """
    _FreeTextModal {
        align: center middle;
    }
    #free_frame {
        width: 60;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    def __init__(self, prompt: str, *, on_done) -> None:
        super().__init__()
        self._prompt = prompt
        self._on_done = on_done

    def compose(self) -> ComposeResult:
        from textual.widgets import Input

        with Container(id="free_frame"):
            yield Static(self._prompt)
            yield Input(placeholder="请输入…", id="free_input")

    def on_mount(self) -> None:
        from textual.widgets import Input

        self.query_one("#free_input", Input).focus()

    def on_input_submitted(self, event) -> None:
        from textual.widgets import Input

        if event.input.id != "free_input":
            return
        text = event.value.strip()
        self.dismiss(None)
        self._on_done(text)


def prompt_menu_blocking(
    title: str,
    options: list[MenuOption],
    *,
    default_index: int = 0,
) -> int | None:
    """
    阻塞式菜单。

    @param title 标题
    @param options 选项
    @param default_index 默认选中
    @return 下标或 None
    """
    from llgraph.ui.context import get_ui_app

    app = get_ui_app()
    if app is not None:
        return app.push_screen_wait_menu(title, options, default_index=default_index)
    from llgraph.terminal.interactive_prompt import prompt_menu_tty

    return prompt_menu_tty(title, options, default_index=default_index)


def run_survey_wizard(spec: SurveySpec) -> dict[str, str] | None:
    """
    TUI 问卷向导。

    @param spec 问卷
    @return 答案或 None
    """
    from llgraph.ui.context import get_ui_app

    app = get_ui_app()
    if app is not None:
        return app.push_screen_wait_survey(spec)
    from llgraph.terminal.survey_wizard import run_survey_tty

    return run_survey_tty(spec)
