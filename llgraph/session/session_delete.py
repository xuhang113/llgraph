"""删除会话落盘数据（用户目录 + 工作区编辑账本）。"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from llgraph.session.user_storage import session_thread_dir, user_sessions_root

_THREAD_ID_STRICT = re.compile(r"^cli-[0-9a-f]{8}$", re.IGNORECASE)
# 删除时允许历史脏数据（如 cli-test），仅限制路径安全字符
_THREAD_ID_LOOSE = re.compile(r"^[\w\-.]{1,128}$")


@dataclass(frozen=True)
class SessionDeleteResult:
    """单会话删除结果。"""

    thread_id: str
    ok: bool
    removed_paths: tuple[str, ...]
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


def validate_thread_id(thread_id: str, *, strict: bool = False) -> str:
    """
    校验 thread_id，防止路径穿越。

    @param thread_id 会话 ID
    @param strict True 时仅允许 cli-xxxxxxxx；False 时允许清理历史脏 ID（如 cli-test）
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
    删除单个会话的全部落盘数据。

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


def delete_sessions(
    workspace: Path,
    thread_ids: list[str],
) -> SessionDeleteBatchReport:
    """
    批量删除会话。

    @param workspace 工作区根
    @param thread_ids 线程 ID 列表
    @return 批量报告
    """
    results = [delete_session(workspace, tid) for tid in thread_ids]
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
                lines.append(f"      - …共 {len(item.removed_paths)} 项")
        else:
            lines.append(f"  ✗ {item.thread_id}: {item.error or '未知错误'}")
    return "\n".join(lines)
