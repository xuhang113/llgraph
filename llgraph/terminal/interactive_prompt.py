"""终端交互式选项菜单（↑↓ · Enter · Esc）。"""

from __future__ import annotations

import sys

from llgraph.terminal.redraw import redraw_tty_block, reset_tty_redraw_slot
from llgraph.ui.prompts import MenuOption

_ANSI_HIDE_CURSOR = "\033[?25l"
_ANSI_SHOW_CURSOR = "\033[?25h"


def _stdin_is_tty() -> bool:
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _read_key_tty() -> str:
    """
    读取单键（TTY raw 模式）。

    @return 键序列
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return ch + ch2 + ch3
            return ch + ch2
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def prompt_menu_tty(
    title: str,
    options: list[MenuOption],
    *,
    default_index: int = 0,
) -> int | None:
    """
    阻塞式终端菜单。

    @param title 标题
    @param options 选项
    @param default_index 默认选中
    @return 下标或 None（取消）
    """
    if not options:
        return None
    if not _stdin_is_tty():
        print(title, flush=True)
        for idx, opt in enumerate(options):
            print(f"  [{idx}] {opt.label}", flush=True)
        try:
            raw = input("选择序号: ").strip()
            picked = int(raw)
            if 0 <= picked < len(options):
                return picked
        except (ValueError, EOFError):
            pass
        return default_index

    slot = "llgraph_menu"
    reset_tty_redraw_slot(slot)
    index = min(default_index, len(options) - 1)
    sys.stdout.write(_ANSI_HIDE_CURSOR)
    sys.stdout.flush()
    try:
        while True:
            lines = [title, ""]
            for idx, opt in enumerate(options):
                mark = "›" if idx == index else " "
                hint = f"  ({opt.hint})" if opt.hint else ""
                lines.append(f" {mark} {opt.label}{hint}")
            lines.append("")
            lines.append("Enter 确认 · ↑↓ 移动 · Esc 取消")
            redraw_tty_block("\n".join(lines), slot=slot)
            key = _read_key_tty()
            if key in ("\r", "\n"):
                reset_tty_redraw_slot(slot)
                sys.stdout.write(_ANSI_SHOW_CURSOR)
                sys.stdout.flush()
                return index
            if key in ("\x1b", "q", "Q"):
                reset_tty_redraw_slot(slot)
                sys.stdout.write(_ANSI_SHOW_CURSOR)
                sys.stdout.flush()
                return None
            if key == "\x1b[A":
                index = (index - 1) % len(options)
            elif key == "\x1b[B":
                index = (index + 1) % len(options)
    finally:
        sys.stdout.write(_ANSI_SHOW_CURSOR)
        sys.stdout.flush()
