"""经典终端对话输出与报告分色。"""

from __future__ import annotations

import sys

from llgraph.terminal.style import sty
from llgraph.terminal.terminal_theme import colorize_terminal_text


def write_dialog_block(text: str) -> None:
    """
    向终端写入多行文本。

    @param text 多行内容
    """
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
    print(payload, flush=True)


def emit_block(text: str, *, colorize: bool = True) -> None:
    """
    输出多行报告块（默认自动分色）。

    @param text 多行文本
    @param colorize 是否套用 terminal_theme
    """
    payload = colorize_terminal_text(text) if colorize else text
    write_dialog_block(payload)


def emit_report(text: str) -> None:
    """
    输出元命令/状态报告（emit_block 别名，始终分色）。

    @param text 多行文本
    """
    emit_block(text, colorize=True)


def emit_error(text: str) -> None:
    """
    错误提示（stderr 优先，经典终端 err 色）。

    @param text 错误说明
    """
    msg = text if text.startswith("●") else f"● {text}"
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
