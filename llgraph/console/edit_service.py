"""Web 文件变更：diff、Plan 聚合、回滚、评审。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.commands.review_command import run_review
from llgraph.console.discovery import load_plan_detail
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
    return {
        "ok": failed == 0,
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


def plan_worker_thread_ids(plan_thread_id: str, tasks: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """
  列出 Plan 下各 Worker 的 thread_id。

    @param plan_thread_id Plan thread
    @param tasks plan tasks
    @return (task_id, title, worker_thread_id)
    """
    rows: list[tuple[str, str, str]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "").strip()
        if not tid:
            continue
        title = str(task.get("title") or tid)
        rows.append((tid, title, f"{plan_thread_id}:worker:{tid}"))
    return rows


def plan_file_changes(workspace: Path, plan_thread_id: str) -> dict[str, Any]:
    """
    聚合 Plan 下各 Worker 的文件改动。

    @param workspace 工作区根
    @param plan_thread_id plan-* thread
    @return groups / total / can_undo
    """
    detail = load_plan_detail(workspace, plan_thread_id)
    tasks = detail.get("tasks") if isinstance(detail.get("tasks"), list) else []
    groups: list[dict[str, Any]] = []
    total = 0
    for task_id, title, worker_thread in plan_worker_thread_ids(plan_thread_id, tasks):
        tracker = session_edit_tracker(workspace, worker_thread)
        payload = tracker.web_changes_payload()
        count = int(payload.get("total") or 0)
        if count <= 0:
            continue
        total += count
        groups.append(
            {
                "task_id": task_id,
                "title": title,
                "thread_id": worker_thread,
                **payload,
            }
        )
    return {
        "plan_thread_id": plan_thread_id,
        "groups": groups,
        "total": total,
        "can_undo": total > 0,
    }


def plan_undo_files(
    workspace: Path,
    plan_thread_id: str,
    *,
    target: str,
    task_id: str | None = None,
) -> dict[str, Any]:
    """
    Plan 级或单 Work 回滚。

    @param workspace 工作区根
    @param plan_thread_id plan-* thread
    @param target all 或相对路径
    @param task_id 指定 Work 时仅回滚该 Worker
    @return 聚合 undo 结果
    """
    detail = load_plan_detail(workspace, plan_thread_id)
    tasks = detail.get("tasks") if isinstance(detail.get("tasks"), list) else []
    workers = plan_worker_thread_ids(plan_thread_id, tasks)
    if task_id:
        workers = [row for row in workers if row[0] == task_id.strip()]
        if not workers:
            raise ValueError(f"Work 不存在: {task_id}")

    all_results: list[dict[str, Any]] = []
    restored = deleted = skipped = failed = 0
    group_changes: list[dict[str, Any]] = []

    for wid, _title, worker_thread in workers:
        tracker = session_edit_tracker(workspace, worker_thread)
        if target.lower() == "all":
            results = tracker.restore_all()
        else:
            results = [tracker.restore_path(target)]
        payload = _undo_payload(results, tracker)
        restored += int(payload["summary"]["restored"])
        deleted += int(payload["summary"]["deleted"])
        skipped += int(payload["summary"]["skipped"])
        failed += int(payload["summary"]["failed"])
        all_results.extend(payload["results"])
        if int(payload["changes"].get("total") or 0) > 0 or results:
            group_changes.append(
                {
                    "task_id": wid,
                    "thread_id": worker_thread,
                    "changes": payload["changes"],
                }
            )

    return {
        "ok": failed == 0,
        "summary": {
            "restored": restored,
            "deleted": deleted,
            "skipped": skipped,
            "failed": failed,
        },
        "results": all_results,
        "groups": group_changes,
        "plan_changes": plan_file_changes(workspace, plan_thread_id),
    }


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


def plan_run_review(workspace: Path, plan_thread_id: str, *, topic: str = "") -> dict[str, Any]:
    """
    对 Plan 各 Work 有改动的会话分别执行评审。

    @param workspace 工作区根
    @param plan_thread_id plan-* thread
    @param topic 评审主题
    @return 聚合结果
    """
    aggregated = plan_file_changes(workspace, plan_thread_id)
    reviews: list[dict[str, Any]] = []
    for group in aggregated.get("groups") or []:
        thread_id = str(group.get("thread_id") or "")
        if not thread_id:
            continue
        res = run_session_review(workspace, thread_id, topic=topic)
        if res.get("ok"):
            res["task_id"] = group.get("task_id")
            reviews.append(res)
    if not reviews:
        return {"ok": False, "message": "Plan 尚无文件改动可评审"}
    return {"ok": True, "reviews": reviews}
