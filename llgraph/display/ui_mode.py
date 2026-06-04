"""交互 UI 模式：TUI 与经典终端。"""

from __future__ import annotations

import os
from enum import Enum


class UiMode(str, Enum):
    """启动交互时使用的界面。"""

    TUI = "tui"
    TERMINAL = "terminal"


def parse_ui_mode(name: str) -> UiMode | None:
    """
    解析 UI 模式名。

    @param name 如 tui、terminal、plain
    @return 对应枚举；无法识别时 None
    """
    key = name.strip().lower()
    aliases = {
        "tui": UiMode.TUI,
        "textual": UiMode.TUI,
        "terminal": UiMode.TERMINAL,
        "term": UiMode.TERMINAL,
        "plain": UiMode.TERMINAL,
        "cli": UiMode.TERMINAL,
    }
    return aliases.get(key)


def resolve_ui_mode(cli_value: str | None) -> UiMode:
    """
    决定本次会话 UI：命令行优先，其次 LLGRAPH_UI，默认经典终端。

    @param cli_value --ui 传入值
    @return 最终模式
    """
    if cli_value:
        parsed = parse_ui_mode(cli_value)
        if parsed is not None:
            return parsed
    env = os.environ.get("LLGRAPH_UI", "").strip()
    if env:
        parsed = parse_ui_mode(env)
        if parsed is not None:
            return parsed
    return UiMode.TERMINAL
