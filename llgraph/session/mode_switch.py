"""Agent ↔ Plan 模式切换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SessionModeTransition:
    """会话模式切换请求。"""

    mode: Literal["agent", "plan"]
    thread_id: str | None = None
    opening_goal: str = ""
    from_thread_id: str | None = None
    handoff_report: str | None = None


def parse_session_mode_command(text: str) -> tuple[str | None, str, str]:
    """
    解析 /session plan|agent 子命令。

    @param text 用户输入
    @return (mode, thread_id, goal)；非模式切换命令返回 (None, "", "")
    """
    stripped = text.strip()
    lower = stripped.lower()
    if not (lower == "/session" or lower.startswith("/session ")):
        return None, "", ""

    tokens = stripped.split()
    if len(tokens) < 2:
        return None, "", ""

    sub = tokens[1].lower()
    if sub == "plan":
        if len(tokens) >= 3 and tokens[2].lower() in ("use", "switch", "resume"):
            return "plan_switch_removed", "", ""
        goal = stripped.split(None, 2)[2].strip() if len(stripped.split(None, 2)) > 2 else ""
        return "plan", "", goal

    if sub == "agent":
        if len(tokens) >= 4 and tokens[2].lower() in ("use", "switch", "resume"):
            return "agent", tokens[3].strip(), ""
        if len(tokens) >= 3 and not tokens[2].startswith("-"):
            tid = tokens[2].strip()
            if tid.startswith("cli-"):
                return "agent", tid, ""
        return "agent", "", ""

    return None, "", ""
