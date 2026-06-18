"""删除会话落盘数据（Agent / Plan 含 Worker 子节点）。"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from llgraph.session.user_storage import session_thread_dir, user_sessions_root

_THREAD_ID_STRICT = re.compile(r"^cli-[0-9a-f]{8}$", re.IGNORECASE)
_PLAN_MAIN_THREAD = re.compile(r"^plan-[0-9a-f]{8}$", re.IGNORECASE)
# 删除时允许 plan-* 子会话（含 :planner: / :worker:），仅限制路径安全字符
_THREAD_ID_LOOSE = re.compile(r"^[\w\-.:]{1,128}$")


@dataclass(frozen=True)
class SessionDeleteResult:
    """单会话删除结果。"""

    thread_id: str
    ok: bool
    removed_paths: tuple[str, ...]
    related_removed: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class SessionDeleteBatchReport:
    """批量删除报告。"""

    results: tuple[SessionDeleteResult, ...]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)


def is_plan_main_thread(thread_id: str) -> bool:
    """
    是否为 Plan 主会话 thread_id（plan-xxxxxxxx）。

    @param thread_id 线程 ID
    @return 是否 Plan 主会话
    """
    return bool(_PLAN_MAIN_THREAD.match(thread_id.strip()))


def is_plan_subsession_thread(thread_id: str) -> bool:
    """
    是否为 Plan 子 Agent 会话（planner / worker）。

    @param thread_id 线程 ID
    @return 是否 Plan 子会话
    """
    tid = thread_id.strip()
    if not tid.startswith("plan-"):
        return False
    return ":" in tid


def validate_thread_id(thread_id: str, *, strict: bool = False) -> str:
    """
    校验 thread_id，防止路径穿越。

    @param thread_id 会话 ID
    @param strict True 时仅允许 cli-xxxxxxxx；False 时允许 plan-* 及子会话
    @return 规范化 ID
    """
    tid = thread_id.strip()
    if not tid:
        raise ValueError("thread_id 不能为空")
    if "/" in tid or "\\" in tid or ".." in tid:
        raise ValueError(f"非法 thread_id: {thread_id}")
    if strict and not _THREAD_ID_STRICT.match(tid):
        raise ValueError(
            f"thread_id 格式应为 cli-xxxxxxxx（8 位十六进制）: {thread_id}"
        )
    if not strict and not _THREAD_ID_LOOSE.match(tid):
        raise ValueError(f"非法 thread_id 字符: {thread_id}")
    return tid


def collect_session_artifact_paths(workspace: Path, thread_id: str) -> list[Path]:
    """
    收集会话相关落盘路径。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 待删除路径列表
    """
    tid = validate_thread_id(thread_id)
    root = workspace.expanduser().resolve()
    from llgraph.session.user_storage import legacy_workspace_session_dir

    paths: list[Path] = [
        session_thread_dir(root, tid),
        user_sessions_root(root) / f"{tid}.jsonl",
        legacy_workspace_session_dir(root, tid),
    ]
    return paths


def _collect_plan_related_session_dirs(sessions_root: Path, plan_thread_id: str) -> list[Path]:
    """
    收集 Plan 主会话及子 Agent 目录（planner / worker）。

    @param sessions_root sessions 根目录
    @param plan_thread_id plan-* 主 thread
    @return 待删目录列表
    """
    prefix = f"{plan_thread_id}:"
    paths: list[Path] = []
    if not sessions_root.is_dir():
        return paths
    for child in sessions_root.iterdir():
        name = child.name
        if name == plan_thread_id or name.startswith(prefix):
            paths.append(child)
    return paths


def _resolve_plan_id_for_delete(workspace: Path, thread_id: str) -> str:
    """
    删除前解析 plan_id（读 meta / plan_state）。

    @param workspace 工作区根
    @param thread_id plan-* thread
    @return plan_id 或空串
    """
    from llgraph.plan.plan_state_store import load_plan_state
    from llgraph.session.session_meta import load_session_meta

    meta = load_session_meta(workspace, thread_id)
    plan_id = str(meta.get("plan_id") or "").strip()
    if plan_id:
        return plan_id
    state = load_plan_state(workspace, thread_id)
    if isinstance(state, dict):
        plan_id = str(state.get("plan_id") or "").strip()
        if plan_id:
            return plan_id
        plan = state.get("plan")
        if isinstance(plan, dict):
            return str(plan.get("plan_id") or "").strip()
    return ""


def _remove_path(path: Path) -> tuple[bool, str | None]:
    if not path.exists() and not path.is_symlink():
        return True, None
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, None
    except OSError as exc:
        return False, str(exc)


def delete_session(workspace: Path, thread_id: str) -> SessionDeleteResult:
    """
    删除单个会话的全部落盘数据（不含 Plan 级联）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 删除结果
    """
    try:
        tid = validate_thread_id(thread_id)
    except ValueError as exc:
        return SessionDeleteResult(
            thread_id=thread_id,
            ok=False,
            removed_paths=(),
            error=str(exc),
        )

    removed_display: list[str] = []
    errors: list[str] = []
    for path in collect_session_artifact_paths(workspace, tid):
        existed = path.exists()
        ok, err = _remove_path(path)
        if existed and ok:
            removed_display.append(str(path))
        if not ok and err:
            errors.append(f"{path}: {err}")

    if errors:
        return SessionDeleteResult(
            thread_id=tid,
            ok=False,
            removed_paths=tuple(removed_display),
            error="; ".join(errors),
        )
    return SessionDeleteResult(
        thread_id=tid,
        ok=True,
        removed_paths=tuple(removed_display),
    )


def delete_plan_session(workspace: Path, thread_id: str) -> SessionDeleteResult:
    """
    删除 Plan 主会话及关联子节点（planner / worker 目录、.llgraph/plans 落盘）。

    @param workspace 工作区根
    @param thread_id plan-* 主 thread
    @return 删除结果
    """
    try:
        tid = validate_thread_id(thread_id)
    except ValueError as exc:
        return SessionDeleteResult(
            thread_id=thread_id,
            ok=False,
            removed_paths=(),
            error=str(exc),
        )
    if not is_plan_main_thread(tid):
        return SessionDeleteResult(
            thread_id=tid,
            ok=False,
            removed_paths=(),
            error=f"须为 Plan 主会话 plan-xxxxxxxx: {tid}",
        )

    from llgraph.plan.execution_coordinator import is_running

    if is_running(tid):
        return SessionDeleteResult(
            thread_id=tid,
            ok=False,
            removed_paths=(),
            error="Plan 正在执行，请先 /plan stop 或等待完成后再删除",
        )

    root = workspace.expanduser().resolve()
    from llgraph.plan.config import resolve_plan_settings
    from llgraph.plan.plan_store import plan_dir
    from llgraph.session.user_storage import legacy_workspace_session_dir

    settings = resolve_plan_settings(root)
    plan_id = _resolve_plan_id_for_delete(root, tid)

    removed: list[str] = []
    related: list[str] = []
    errors: list[str] = []

    sessions_root = user_sessions_root(root)
    for path in _collect_plan_related_session_dirs(sessions_root, tid):
        existed = path.exists()
        ok, err = _remove_path(path)
        if existed and ok:
            related.append(str(path))
        if not ok and err:
            errors.append(f"{path}: {err}")

    for artifact in (
        sessions_root / f"{tid}.jsonl",
        legacy_workspace_session_dir(root, tid),
    ):
        existed = artifact.exists()
        ok, err = _remove_path(artifact)
        if existed and ok:
            related.append(str(artifact))
        if not ok and err:
            errors.append(f"{artifact}: {err}")

    if plan_id:
        pd = plan_dir(root, plan_id, plans_dir=settings.plans_dir)
        if pd.is_dir():
            ok, err = _remove_path(pd)
            if ok:
                related.append(str(pd))
            elif err:
                errors.append(f"{pd}: {err}")

    if errors:
        return SessionDeleteResult(
            thread_id=tid,
            ok=False,
            removed_paths=tuple(removed),
            related_removed=tuple(related),
            error="; ".join(errors),
        )
    return SessionDeleteResult(
        thread_id=tid,
        ok=True,
        removed_paths=tuple(removed),
        related_removed=tuple(related),
    )


def delete_workspace_session(workspace: Path, thread_id: str) -> SessionDeleteResult:
    """
    按 thread 类型删除 Agent 或 Plan 会话（Plan 含 Worker 级联）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 删除结果
    """
    tid = thread_id.strip()
    if is_plan_main_thread(tid):
        return delete_plan_session(workspace, tid)
    if is_plan_subsession_thread(tid):
        return delete_session(workspace, tid)
    return delete_session(workspace, tid)


def _filter_batch_delete_ids(thread_ids: list[str]) -> list[str]:
    """
    批量删除时跳过将被 Plan 主会话级联删除的子会话。

    @param thread_ids 原始 ID 列表
    @return 过滤后的 ID 列表
    """
    plan_mains = {tid for tid in thread_ids if is_plan_main_thread(tid)}
    filtered: list[str] = []
    for tid in thread_ids:
        if is_plan_subsession_thread(tid):
            parent = tid.split(":", 1)[0]
            if parent in plan_mains:
                continue
        filtered.append(tid)
    return filtered


def delete_sessions(
    workspace: Path,
    thread_ids: list[str],
) -> SessionDeleteBatchReport:
    """
    批量删除会话（Plan 主会话级联删除子节点）。

    @param workspace 工作区根
    @param thread_ids 线程 ID 列表
    @return 批量报告
    """
    ids = _filter_batch_delete_ids(thread_ids)
    results = [delete_workspace_session(workspace, tid) for tid in ids]
    return SessionDeleteBatchReport(results=tuple(results))


def format_delete_report(report: SessionDeleteBatchReport) -> str:
    """
    格式化删除结果说明。

    @param report 批量报告
    @return 多行文本
    """
    lines = [
        f"已删除 {report.success_count} 个会话",
    ]
    if report.failure_count:
        lines[0] += f"，失败 {report.failure_count} 个"
    lines.append("")
    for item in report.results:
        if item.ok:
            lines.append(f"  ✓ {item.thread_id}")
            for p in item.removed_paths[:3]:
                lines.append(f"      - {p}")
            if len(item.removed_paths) > 3:
                lines.append(f"      - …共 {len(item.removed_paths)} 项主路径")
            for p in item.related_removed[:3]:
                lines.append(f"      - {p}")
            if len(item.related_removed) > 3:
                lines.append(f"      - …共 {len(item.related_removed)} 项关联路径")
        else:
            lines.append(f"  ✗ {item.thread_id}: {item.error or '未知错误'}")
    return "\n".join(lines)
