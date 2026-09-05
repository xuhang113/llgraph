"""Web 会话标题更新。"""

from __future__ import annotations

from pathlib import Path

from llgraph.session.session_meta import get_session_title, set_session_title


def update_session_display_title(
    workspace: Path,
    thread_id: str,
    title: str,
) -> tuple[bool, str, str]:
    """
    更新侧边栏展示标题。

    @param workspace 工作区根
    @param thread_id 会话 thread_id
    @param title 新标题
    @return (成功, 提示, 规范化后的标题)
    """
    ok, msg = set_session_title(workspace, thread_id, title, source="manual")
    if not ok:
        return False, msg, ""

    normalized = get_session_title(workspace, thread_id) or title.strip()
    return True, msg, normalized
