"""当前会话文件编辑记录与首次编辑快照（P7）。"""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from llgraph.config.edit_settings import EditSettings, resolve_edit_settings
from llgraph.session.user_storage import (
    migrate_legacy_workspace_session_dir,
    resolve_session_storage_dir,
)


def encode_rel_path(rel_path: str) -> str:
    """
    将相对路径编码为快照文件名。

    @param rel_path 工作区相对路径
    @return 安全文件名
    """
    return rel_path.replace("/", "__").replace("\\", "__")


def decode_snapshot_name(name: str) -> str:
    """
    快照文件名还原为相对路径（展示用）。

    @param name 快照文件名
    @return 相对路径
    """
    stem = name
    if stem.endswith(".txt"):
        stem = stem[:-4]
    return stem.replace("__", "/")


@dataclass
class EditRecord:
    """单次写操作记录。"""

    rel_path: str
    op: str
    timestamp: str
    replacements: int = 1
    lines_added: int = 0
    lines_removed: int = 0


@dataclass(frozen=True)
class UndoItemResult:
    """单文件还原结果。"""

    rel_path: str
    action: str
    detail: str = ""


@dataclass
class SessionEditTracker:
    """
    会话级编辑账本：内存聚合 + 落盘 edits.jsonl / snapshots/。
    """

    workspace: Path
    session_id: str
    settings: EditSettings | None = None
    records: list[EditRecord] = field(default_factory=list)
    _display_cleared: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.settings is None:
            self.settings = resolve_edit_settings(self.workspace)
        sessions_cfg = self.settings.sessions_dir if self.settings else None
        self._session_dir = resolve_session_storage_dir(
            self.workspace,
            self.session_id,
            sessions_dir_config=sessions_cfg,
        )
        migrate_legacy_workspace_session_dir(
            self.workspace, self.session_id, self._session_dir
        )
        self._snapshots_dir = self._session_dir / "snapshots"
        self._edits_path = self._session_dir / "edits.jsonl"
        self._init_session_dir()
        self._load_persisted_edits()

    def _init_session_dir(self) -> None:
        """创建 meta.json、edits.jsonl 与会话目录。"""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        if self.settings and self.settings.persist_edits:
            try:
                if not self._edits_path.is_file():
                    self._edits_path.touch()
            except OSError:
                pass
        meta_path = self._session_dir / "meta.json"
        if not meta_path.is_file():
            meta = {
                "session_id": self.session_id,
                "workspace": str(self.workspace),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_persisted_edits(self) -> None:
        """从 edits.jsonl 恢复记录。"""
        if not self._edits_path.is_file():
            return
        try:
            for line in self._edits_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.records.append(
                    EditRecord(
                        rel_path=data["rel_path"],
                        op=data["op"],
                        timestamp=data["timestamp"],
                        replacements=int(data.get("replacements", 1)),
                        lines_added=int(data.get("lines_added", 0)),
                        lines_removed=int(data.get("lines_removed", 0)),
                    )
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    def _ensure_edits_file(self) -> None:
        """确保会话目录与 edits.jsonl 存在（delete empty 后同 thread 写文件时需重建）。"""
        if not self.settings or not self.settings.persist_edits:
            return
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            if not self._edits_path.is_file():
                self._edits_path.touch()
        except OSError:
            pass

    def _append_jsonl(self, record: EditRecord) -> None:
        """追加一条记录到 edits.jsonl。"""
        if not self.settings or not self.settings.persist_edits:
            return
        self._ensure_edits_file()
        try:
            with self._edits_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def ensure_snapshot(self, rel_path: str) -> None:
        """
        首次编辑前缓存磁盘原文。

        @param rel_path 相对工作区路径
        """
        if not self.settings or not self.settings.snapshot_on_first_edit:
            return
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap_name = f"{encode_rel_path(rel_path)}.txt"
        snap_path = self._snapshots_dir / snap_name
        if snap_path.is_file():
            return
        full = self.workspace / rel_path
        if not full.is_file():
            return
        try:
            snap_path.write_text(full.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass

    def record(
        self,
        rel_path: str,
        op: str,
        *,
        replacements: int = 1,
        lines_added: int = 0,
        lines_removed: int = 0,
    ) -> None:
        """
        记录一次写操作。

        @param rel_path 相对路径
        @param op 操作类型
        @param replacements 替换次数
        @param lines_added 新增行数估计
        @param lines_removed 删除行数估计
        """
        self._display_cleared = False
        entry = EditRecord(
            rel_path=rel_path,
            op=op,
            timestamp=datetime.now(timezone.utc).isoformat(),
            replacements=replacements,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )
        self.records.append(entry)
        self._append_jsonl(entry)

    def unique_paths(self) -> list[str]:
        """
        去重后的已编辑路径（保持首次出现顺序）。

        @return 路径列表
        """
        if self._display_cleared:
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        for rec in self.records:
            if rec.rel_path not in seen:
                seen.add(rec.rel_path)
                ordered.append(rec.rel_path)
        return ordered

    def clear_display(self) -> None:
        """清空内存展示列表，保留落盘审计。"""
        self._display_cleared = True

    def reset_persisted(self) -> None:
        """删除落盘 edits.jsonl 与 snapshots（/changes reset）。"""
        self.records.clear()
        self._display_cleared = False
        if self._edits_path.is_file():
            self._edits_path.unlink()
        if self._snapshots_dir.is_dir():
            for item in self._snapshots_dir.iterdir():
                if item.is_file():
                    item.unlink()

    def format_changes_list(self) -> str:
        """
        格式化 /changes 输出。

        @return 多行文本
        """
        paths = self.unique_paths()
        if not paths:
            return "本会话尚未通过 Agent 修改任何文件。"
        lines = [f"本会话已修改 {len(paths)} 个文件（session: {self.session_id}）:", ""]
        for rel in paths:
            count = sum(1 for r in self.records if r.rel_path == rel)
            last = next(r for r in reversed(self.records) if r.rel_path == rel)
            lines.append(
                f"  {rel}  （{count} 次操作，末次 {last.op} @ {last.timestamp[:19]}Z）"
            )
        lines.append("")
        lines.append(f"落盘: {self._edits_path}")
        lines.append(
            "命令: /diff <path>  |  /undo <path>|all  |  /changes clear  |  /changes reset"
        )
        return "\n".join(lines)

    def _normalize_rel(self, rel_path: str) -> str:
        """
        规范化相对路径。

        @param rel_path 用户输入路径
        @return 去首尾空白与 leading /
        """
        return rel_path.strip().lstrip("/")

    def _snapshot_path_for(self, rel: str) -> Path:
        """
        快照文件路径。

        @param rel 工作区相对路径
        @return snapshots 目录下文件
        """
        return self._snapshots_dir / f"{encode_rel_path(rel)}.txt"

    def _resolve_target(self, rel: str) -> Path:
        """
        解析工作区内目标文件路径。

        @param rel 相对路径
        @return 绝对路径
        @raises ValueError 路径越界
        """
        candidate = (self.workspace / rel).resolve()
        candidate.relative_to(self.workspace.resolve())
        return candidate

    def _rewrite_edits_jsonl(self) -> None:
        """将内存账本写回 edits.jsonl。"""
        if not self.settings or not self.settings.persist_edits:
            return
        self._ensure_edits_file()
        try:
            payload = "\n".join(
                json.dumps(asdict(record), ensure_ascii=False) for record in self.records
            )
            if payload:
                payload += "\n"
            self._edits_path.write_text(payload, encoding="utf-8")
        except OSError:
            pass

    def _drop_path_ledger(self, rel: str) -> None:
        """
        回滚成功后从账本移除路径。

        @param rel 工作区相对路径
        """
        rel = self._normalize_rel(rel)
        if not rel:
            return
        self.records = [record for record in self.records if record.rel_path != rel]
        self._rewrite_edits_jsonl()

    def _is_undo_pending(self, rel: str) -> bool:
        """
        是否仍有可回滚的磁盘改动。

        @param rel 工作区相对路径
        @return 新建文件仍存在，或已修改文件与快照不一致
        """
        rel = self._normalize_rel(rel)
        if not rel:
            return False
        snap_path = self._snapshot_path_for(rel)
        try:
            target = self._resolve_target(rel)
        except ValueError:
            return False
        if snap_path.is_file():
            if not target.is_file():
                return True
            try:
                return snap_path.read_text(encoding="utf-8") != target.read_text(
                    encoding="utf-8"
                )
            except OSError:
                return True
        return target.is_file()

    def list_snapshot_paths(self) -> list[str]:
        """
        列出本会话所有快照对应的路径。

        @return 相对路径列表（排序）
        """
        if not self._snapshots_dir.is_dir():
            return []
        paths: list[str] = []
        for item in sorted(self._snapshots_dir.iterdir()):
            if item.is_file() and item.name.endswith(".txt"):
                paths.append(decode_snapshot_name(item.name))
        return paths

    def _all_edited_paths(self) -> list[str]:
        """
        账本中所有编辑过的路径（不受 /changes clear 影响）。

        @return 去重路径列表
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for rec in self.records:
            if rec.rel_path not in seen:
                seen.add(rec.rel_path)
                ordered.append(rec.rel_path)
        return ordered

    def _undo_targets(self) -> list[str]:
        """
        汇总仍可回滚的目标（账本路径 + 快照路径，且磁盘状态待还原）。

        @return 去重相对路径
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for rel in self._all_edited_paths():
            if rel in seen or not self._is_undo_pending(rel):
                continue
            seen.add(rel)
            ordered.append(rel)
        for rel in self.list_snapshot_paths():
            if rel in seen or not self._is_undo_pending(rel):
                continue
            seen.add(rel)
            ordered.append(rel)
        return ordered

    def _finalize_undo(self, rel: str, result: UndoItemResult) -> UndoItemResult:
        """
        回滚成功后清理账本，避免 Web 列表仍显示已处理文件。

        @param rel 工作区相对路径
        @param result 还原结果
        @return 原结果
        """
        if result.action in {"restored", "deleted"}:
            self._drop_path_ledger(rel)
        return result

    def restore_path(self, rel_path: str) -> UndoItemResult:
        """
        还原单个文件：有快照则写回；无快照则删除本会话新建文件。

        @param rel_path 工作区相对路径
        @return 还原结果
        """
        rel = self._normalize_rel(rel_path)
        if not rel:
            return UndoItemResult(rel_path, "failed", "路径为空")

        snap_path = self._snapshot_path_for(rel)
        try:
            target = self._resolve_target(rel)
        except ValueError as exc:
            return UndoItemResult(rel, "failed", str(exc))

        if snap_path.is_file():
            try:
                content = snap_path.read_text(encoding="utf-8")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return self._finalize_undo(
                    rel,
                    UndoItemResult(rel, "restored", "已从会话首次编辑前快照还原"),
                )
            except OSError as exc:
                return UndoItemResult(rel, "failed", str(exc))

        if target.is_file():
            try:
                target.unlink()
                return self._finalize_undo(
                    rel,
                    UndoItemResult(rel, "deleted", "已删除本会话新建的文件（无编辑前快照）"),
                )
            except OSError as exc:
                return UndoItemResult(rel, "failed", str(exc))

        return UndoItemResult(
            rel,
            "failed",
            "账本有记录但磁盘上找不到该路径的文件（可能路径不一致或通过 shell 写入到其他位置）",
        )

    def restore_all(self) -> list[UndoItemResult]:
        """
        还原本会话所有可处理文件。

        @return 每个路径的还原结果
        """
        return [self.restore_path(rel) for rel in self._undo_targets()]

    def web_changes_payload(self) -> dict[str, object]:
        """
        Web UI：本会话可还原文件摘要。

        @return paths / total / can_undo
        """
        targets = self._undo_targets()
        paths: list[dict[str, object]] = []
        for rel in targets:
            has_snap = self._snapshot_path_for(rel).is_file()
            count = sum(1 for r in self.records if r.rel_path == rel)
            paths.append(
                {
                    "path": rel,
                    "has_snapshot": has_snap,
                    "kind": "modified" if has_snap else "created",
                    "edit_count": count,
                }
            )
        return {
            "session_id": self.session_id,
            "paths": paths,
            "total": len(paths),
            "can_undo": bool(targets),
        }

    def format_undo_report(self, results: list[UndoItemResult]) -> str:
        """
        格式化 /undo 输出。

        @param results restore_path / restore_all 结果
        @return 多行文本
        """
        if not results:
            return "无可还原项：本会话尚未修改任何文件。"

        restored = [r for r in results if r.action == "restored"]
        deleted = [r for r in results if r.action == "deleted"]
        skipped = [r for r in results if r.action == "skipped"]
        failed = [r for r in results if r.action == "failed"]

        lines = [
            f"Undo 完成（session: {self.session_id}）:",
            f"  还原 {len(restored)}  |  删除新建 {len(deleted)}  |  "
            f"跳过 {len(skipped)}  |  失败 {len(failed)}",
            "",
        ]
        for group, label in (
            (restored, "已还原"),
            (deleted, "已删除"),
            (skipped, "已跳过"),
            (failed, "失败"),
        ):
            if not group:
                continue
            lines.append(f"{label}:")
            for item in group:
                suffix = f" — {item.detail}" if item.detail else ""
                lines.append(f"  {item.rel_path}{suffix}")
            lines.append("")

        lines.append("提示: edits.jsonl 仍保留审计记录；可用 /changes reset 清空账本与快照。")
        return "\n".join(lines).rstrip()

    def format_undo_usage(self) -> str:
        """
        /undo 无参数时的用法说明。

        @return 多行文本
        """
        targets = self._undo_targets()
        snap_count = len(self.list_snapshot_paths())
        lines = [
            "用法:",
            "  /undo all           还原本会话全部改动（有快照写回，新建文件删除）",
            "  /undo <相对路径>     还原单个文件",
            "",
            f"当前 session: {self.session_id}",
            f"可处理路径: {len(targets)} 个（快照 {snap_count} 个）",
        ]
        if targets:
            lines.append("")
            preview = targets[:12]
            for rel in preview:
                has_snap = self._snapshot_path_for(rel).is_file()
                tag = "快照" if has_snap else "新建"
                lines.append(f"  [{tag}] {rel}")
            if len(targets) > 12:
                lines.append(f"  … 另有 {len(targets) - 12} 个")
        return "\n".join(lines)

    def format_diff(self, rel_path: str) -> str:
        """
        对比首次编辑快照与当前磁盘。

        @param rel_path 相对路径
        @return diff 文本或提示
        """
        rel = rel_path.strip().lstrip("/")
        snap_path = self._snapshots_dir / f"{encode_rel_path(rel)}.txt"
        full = self.workspace / rel
        if not snap_path.is_file():
            if full.is_file():
                try:
                    size = full.stat().st_size
                except OSError:
                    size = 0
                return (
                    f"新建文件: {rel}（{size} 字节）。\n"
                    "本会话首次写入前无快照；点击「回滚」或 /undo 将删除该文件。"
                )
            return (
                f"无快照: 本会话尚未编辑过 {rel!r}，或该文件已回滚删除。\n"
                "可先 read_file 查看当前内容，或使用 git diff。"
            )
        full = self.workspace / rel
        if not full.is_file():
            return f"文件已删除: {rel}"
        try:
            old_lines = snap_path.read_text(encoding="utf-8").splitlines(keepends=True)
            new_lines = full.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError as exc:
            return f"读取失败: {exc}"
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{rel} (会话首次编辑前)",
            tofile=f"{rel} (当前)",
            lineterm="",
        )
        body = "".join(diff)
        if not body.strip():
            return f"{rel}: 与首次编辑前快照相同（无差异）。"
        return body

    def paths_for_review(self) -> list[str]:
        """
        供 /review 使用的路径列表（不受 /changes clear 影响）。

        @return 去重相对路径
        """
        return self._all_edited_paths()

    def collect_review_diff(self, paths: list[str], *, max_chars: int = 12000) -> str:
        """
        聚合本会话变更 diff（首次编辑快照 vs 当前磁盘），供 /review 使用。

        @param paths 待评审相对路径
        @param max_chars 总字符上限
        @return unified diff 或新建/删除说明
        """
        chunks: list[str] = []
        remaining = max_chars
        for rel in paths:
            if remaining <= 200:
                break
            rel = self._normalize_rel(rel)
            if not rel:
                continue
            snap_path = self._snapshot_path_for(rel)
            try:
                full = self._resolve_target(rel)
            except ValueError:
                continue

            part = ""
            if snap_path.is_file() and full.is_file():
                part = self.format_diff(rel)
            elif snap_path.is_file() and not full.is_file():
                try:
                    old_lines = snap_path.read_text(encoding="utf-8").splitlines(
                        keepends=True,
                    )
                except OSError:
                    old_lines = []
                diff = difflib.unified_diff(
                    old_lines,
                    [],
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                    lineterm="",
                )
                part = "".join(diff) or f"（文件已删除: {rel}）\n"
            elif full.is_file() and not snap_path.is_file():
                try:
                    content = full.read_text(encoding="utf-8")
                except OSError:
                    continue
                preview_lines = content.splitlines()[:300]
                part = (
                    f"--- {rel} (本会话新建，无编辑前快照) ---\n"
                    + "\n".join(f"+{line}" for line in preview_lines)
                )
                if len(content.splitlines()) > 300:
                    part += f"\n…（共 {len(content.splitlines())} 行，已截断）"

            if not part.strip() or part.startswith("无快照"):
                continue
            block = f"--- {rel} ---\n{part.rstrip()}\n"
            if len(block) > remaining:
                block = block[:remaining] + "\n…（截断）\n"
            chunks.append(block)
            remaining -= len(block)
        return "\n".join(chunks).strip()

    def exit_summary(self) -> str:
        """
        退出时会话变更摘要。

        @return 一行摘要
        """
        paths = self.unique_paths()
        if not paths:
            return ""
        preview = ", ".join(paths[:5])
        suffix = f" 等 {len(paths)} 个" if len(paths) > 5 else ""
        return f"本会话共修改 {len(paths)} 个文件: {preview}{suffix}"
