"""终端原地重绘：上移 + 清屏尾 + 逐行写入，避免菜单切换时残字。"""

from __future__ import annotations

import re
import shutil
import sys
import unicodedata

_SLOTS: dict[str, int] = {}

_ANSI_RE = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """
    去掉 ANSI 转义序列。

    @param text 含 ANSI 的文本
    @return 纯文本
    """
    return _ANSI_RE.sub("", text)


def _display_width(text: str) -> int:
    """
    估算终端显示宽度（CJK 计 2 列）。

    @param text 单行文本
    @return 显示列宽
    """
    width = 0
    for ch in _strip_ansi(text):
        if unicodedata.east_asian_width(ch) in ("F", "W"):
            width += 2
        elif unicodedata.category(ch) != "Mn":
            width += 1
    return width


def _terminal_width() -> int:
    """
    当前终端列宽。

    @return 列数，至少 40
    """
    try:
        return max(40, shutil.get_terminal_size(fallback=(120, 24)).columns)
    except OSError:
        return 120


def visual_line_count(text: str, *, terminal_width: int | None = None) -> int:
    """
    统计输出块占用的终端行数（含软换行）。

    @param text 待输出文本
    @param terminal_width 终端列宽
    @return 行数（空串为 0）
    """
    if not text:
        return 0
    width = terminal_width or _terminal_width()
    total = 0
    for line in text.split("\n"):
        w = _display_width(line)
        total += max(1, (w + width - 1) // width) if w else 1
    return total


def redraw_tty_block(block: str, *, slot: str = "default") -> int:
    """
    在 TTY 上原地重绘一块文本（覆盖上一帧）。

    先按上一帧占用的**显示行数**上移光标，再清屏尾，避免长行折行后残字。

    @param block 多行文本（可含 ANSI）
    @param slot 重绘槽位 id（多组件互不干扰）
    @return 本帧占用的显示行数
    """
    width = _terminal_width()
    new_rows = visual_line_count(block, terminal_width=width)
    prev_rows = _SLOTS.get(slot, 0)

    if prev_rows > 0:
        sys.stdout.write(f"\033[{prev_rows}A")
    # 从当前光标清到屏幕末尾，比逐行 2K 更可靠（长行折行时 prev_rows 已含折行）
    sys.stdout.write("\033[0J")
    if block:
        sys.stdout.write(block)
        if not block.endswith("\n"):
            sys.stdout.write("\n")
    sys.stdout.flush()
    _SLOTS[slot] = new_rows
    return new_rows


def reset_tty_redraw_slot(slot: str = "default") -> None:
    """
    重置重绘槽位行数记录。

    @param slot 槽位 id
    """
    _SLOTS.pop(slot, None)
