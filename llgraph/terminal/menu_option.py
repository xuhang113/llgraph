"""终端菜单选项。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MenuOption:
    """菜单项。"""

    label: str
    hint: str = ""
