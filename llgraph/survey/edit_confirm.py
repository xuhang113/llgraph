"""写文件前终端确认。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.terminal.menu import MenuOption, prompt_menu_blocking


@dataclass
class EditConfirmGate:
    """
    会话级写确认闸门。

    allow_all_session 为 True 时本会话不再弹窗。
    """

    workspace: Path
    allow_all_session: bool = False

    def should_prompt(self) -> bool:
        """
        本操作是否需要弹窗确认。

        @return 是否弹窗
        """
        mode = resolve_edit_settings(self.workspace).confirm_writes
        if mode == "never":
            return False
        if mode == "always":
            return True
        if self.allow_all_session:
            return False
        return mode == "interactive"

    def confirm_write(self, rel_path: str, action_label: str) -> bool:
        """
        写操作前确认。

        @param rel_path 相对工作区路径
        @param action_label 动作描述（创建/覆盖/编辑等）
        @return 是否允许写入
        """
        if not self.should_prompt():
            return True

        title = f"{action_label} `{rel_path}`？"
        options = [
            MenuOption("Yes", hint="执行本次写入"),
            MenuOption(
                "Yes, allow all edits during this session",
                hint="本会话不再询问",
            ),
            MenuOption("No", hint="拒绝本次写入"),
        ]
        picked = prompt_menu_blocking(title, options, default_index=0)
        if picked is None or picked == 2:
            return False
        if picked == 1:
            self.allow_all_session = True
        return True
