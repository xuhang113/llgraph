"""Web 文件变更：diff、回滚、评审。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.commands.review_command import run_review
from llgraph.session.session_edits import SessionEditTracker, UndoItemResult


def session_edit_tracker(workspace: Path, thread_id: str) -> SessionEditTracker:
    """
    加载会话编辑账本。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @return SessionEditTracker
    """
    return SessionEditTracker(workspace, session_id=thread_id.strip())


def session_diff_text(workspace: Path, thread_id: str, rel_path: str) -> str:
    """
    单文件 diff（快照 vs 当前磁盘）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param rel_path 相对路径
    @return diff 文本
    """
    tracker = session_edit_tracker(workspace, thread_id)
    return tracker.format_diff(rel_path)


def _undo_payload(results: list[UndoItemResult], tracker: SessionEditTracker) -> dict[str, Any]:
    restored = sum(1 for r in results if r.action == "restored")
    deleted = sum(1 for r in results if r.action == "deleted")
    skipped = sum(1 for r in results if r.action == "skipped")
    failed = sum(1 for r in results if r.action == "failed")
    effective = restored + deleted
    return {
        "ok": failed == 0 and skipped == 0 and (not results or effective > 0),
        "summary": {
            "restored": restored,
            "deleted": deleted,
            "skipped": skipped,
            "failed": failed,
        },
        "results": [
            {"rel_path": r.rel_path, "action": r.action, "detail": r.detail}
            for r in results
        ],
        "changes": tracker.web_changes_payload(),
    }


def undo_session_files(
    workspace: Path,
    thread_id: str,
    *,
    target: str,
) -> dict[str, Any]:
    """
    还原单文件或全部改动。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param target all 或相对路径
    @return undo 结果
    """
    tracker = session_edit_tracker(workspace, thread_id)
    target = target.strip()
    if not target:
        raise ValueError("target 不能为空")
    if target.lower() == "all":
        results = tracker.restore_all()
    else:
        results = [tracker.restore_path(target)]
    return _undo_payload(results, tracker)


def run_session_review(
    workspace: Path,
    thread_id: str,
    *,
    topic: str = "",
) -> dict[str, Any]:
    """
    对指定会话变更执行 /review 并落盘。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param topic 评审主题
    @return 路径与摘要
    """
    tracker = session_edit_tracker(workspace, thread_id)
    if not tracker.paths_for_review():
        return {"ok": False, "message": "本会话尚无文件改动可评审"}
    review_path, summary = run_review(
        workspace,
        topic=topic,
        edit_tracker=tracker,
    )
    return {
        "ok": True,
        "review_path": str(review_path),
        "summary": summary,
    }
