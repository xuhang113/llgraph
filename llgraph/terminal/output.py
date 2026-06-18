"""经典终端对话输出与报告分色。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from llgraph.terminal.style import sty
from llgraph.terminal.terminal_theme import colorize_terminal_text

_emit_capture: ContextVar[list[str] | None] = ContextVar("emit_capture", default=None)


def _append_capture(text: str) -> bool:
    """
    将输出写入 Web 元命令捕获缓冲。

    @param text 待写入文本（可含 ANSI）
    @return 是否处于捕获模式
    """
    buf = _emit_capture.get()
    if buf is None:
        return False
    from llgraph.display.trace_sink import strip_ansi

    cleaned = strip_ansi(text).strip()
    if cleaned:
        buf.append(cleaned)
    return True


@contextmanager
def capture_terminal_output() -> Iterator[list[str]]:
    """
    捕获 emit / emit_report 等终端输出（供 Web 元命令 API 返回）。

    @yield 按块收集的纯文本列表
    """
    buf: list[str] = []
    token = _emit_capture.set(buf)
    try:
        yield buf
    finally:
        _emit_capture.reset(token)


def format_captured_output(buf: list[str]) -> str:
    """
    将捕获块合并为单段 Markdown/纯文本。

    @param buf capture_terminal_output 收集的列表
    @return 合并后的文本
    """
    return "\n".join(buf)


def write_dialog_block(text: str) -> None:
    """
    向终端写入多行文本。

    @param text 多行内容
    """
    if _append_capture(text):
        return
    if text.strip():
        print(text.strip(), flush=True)


def write_dialog_line(text: str) -> None:
    """
    向终端写入单行。

    @param text 一行文本
    """
    if text.strip():
        print(text, flush=True)


def emit(
    text: str = "",
    *,
    style: str | None = None,
    colorize: bool = False,
) -> None:
    """
    终端输出单行（空串仅换行）。

    @param text 一行文本
    @param style 直接套 STYLES 键名
    @param colorize 是否按 terminal_theme 规则分色
    """
    if style:
        payload = sty(text, style)
    elif colorize and text.strip():
        payload = colorize_terminal_text(text)
    else:
        payload = text
    if _append_capture(payload):
        return
    print(payload, flush=True)


def emit_block(
    text: str,
    *,
    colorize: bool = True,
    render_markdown: bool = False,
) -> None:
    """
    输出多行报告块（/help、/plan results 等元命令；默认仅分色，不 Rich）。

    @param text 多行 Markdown/纯文本原文
    @param colorize 是否套用 terminal_theme 分色
    @param render_markdown 是否额外做 Rich/ANSI Markdown 渲染（默认 False）
    """
    if render_markdown and text.strip():
        from llgraph.terminal.markdown_render import render_for_terminal, rich_render_enabled

        rendered = render_for_terminal(
            text,
            force=True,
            use_rich=rich_render_enabled(),
        )
        if rendered.strip():
            write_dialog_block(rendered)
            return
    payload = colorize_terminal_text(text) if colorize else text
    write_dialog_block(payload)


def emit_markdown_block(text: str) -> None:
    """
    显式对 Markdown 做终端 Rich/ANSI 渲染（元命令一般勿用，留给助手 trace）。

    @param text Markdown 原文
    """
    emit_block(text, colorize=False, render_markdown=True)


def emit_report(text: str) -> None:
    """
    输出元命令/状态报告（Markdown 原文 + 终端分色，不做 Rich 渲染）。

    @param text 多行文本
    """
    emit_block(text, colorize=True, render_markdown=False)


def emit_error(text: str) -> None:
    """
    错误提示（stderr 优先，经典终端 err 色）。

    @param text 错误说明
    """
    msg = text if text.startswith("●") else f"● {text}"
    if _append_capture(msg):
        return
    if sys.stderr.isatty() or sys.stdout.isatty():
        print(sty(msg, "err"), file=sys.stderr, flush=True)
    else:
        print(msg, file=sys.stderr, flush=True)


def emit_ok(text: str) -> None:
    """
    成功/确认提示。

    @param text 说明
    """
    emit(text, style="ok")


def emit_warn(text: str) -> None:
    """
    警告提示。

    @param text 说明
    """
    emit(text, style="warn")


def emit_hint(text: str) -> None:
    """
    灰色辅助说明。

    @param text 说明
    """
    emit(text, style="hint")


def emit_milestone(text: str) -> None:
    """
    流程里程碑（▶ 检测问卷、提交 Agent 等）。

    @param text 说明
    """
    body = text.lstrip("\n")
    if not body.startswith("▶"):
        body = f"▶ {body}"
    emit(f"\n{body}" if text.startswith("\n") else body, style="accent")
