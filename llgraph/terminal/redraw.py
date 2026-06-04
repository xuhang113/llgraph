"""终端原地重绘：上移 + 逐行清除，避免菜单切换时残字。"""

from __future__ import annotations

import sys

_SLOTS: dict[str, int] = {}


def visual_line_count(text: str) -> int:
    """
    统计输出块占用的终端行数。

    @param text 待输出文本
    @return 行数（空串为 0）
    """
    if not text:
        return 0
    return text.count("\n") + 1


def redraw_tty_block(block: str, *, slot: str = "default") -> int:
    """
    在 TTY 上原地重绘一块文本（覆盖上一帧）。

    @param block 多行文本（可含 ANSI）
    @param slot 重绘槽位 id（多组件互不干扰）
    @return 本帧占用的行数
    """
    lines = block.split("\n")
    new_rows = len(lines) if block else 0
    prev_rows = _SLOTS.get(slot, 0)

    if prev_rows > 0:
        sys.stdout.write(f"\033[{prev_rows}A")
        for _ in range(prev_rows):
            sys.stdout.write("\033[2K\r")
    for line in lines:
        sys.stdout.write(line + "\n")
    if prev_rows > new_rows:
        extra = prev_rows - new_rows
        sys.stdout.write(f"\033[{extra}A")
        for _ in range(extra):
            sys.stdout.write("\033[2K\r")
    sys.stdout.flush()
    _SLOTS[slot] = new_rows
    return new_rows


def reset_tty_redraw_slot(slot: str = "default") -> None:
    """
    重置重绘槽位行数记录。

    @param slot 槽位 id
    """
    _SLOTS.pop(slot, None)
