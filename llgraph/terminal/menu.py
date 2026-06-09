"""终端交互式菜单（↑↓ · Enter · Esc）。"""

from __future__ import annotations

from llgraph.terminal.interactive_prompt import prompt_menu_tty
from llgraph.terminal.menu_option import MenuOption

__all__ = ["MenuOption", "prompt_menu_blocking"]


def prompt_menu_blocking(
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
    @return 下标或 None
    """
    return prompt_menu_tty(title, options, default_index=default_index)
