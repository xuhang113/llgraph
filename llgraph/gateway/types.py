"""Control gateway 可序列化 DTO（便于后续 HTTP/JSON 传输）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkspaceRecord:
    """工作区摘要。"""

    slug: str
    path: str
    session_count: int = 0
    plan_count: int = 0
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        转为 JSON 友好 dict。

        @return dict
        """
        return {
            "slug": self.slug,
            "path": self.path,
            "session_count": self.session_count,
            "plan_count": self.plan_count,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SessionDeleteRecord:
    """会话删除结果。"""

    thread_id: str
    kind: str
    ok: bool
    removed_paths: tuple[str, ...]
    related_removed: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        转为 JSON 友好 dict。

        @return dict
        """
        return {
            "thread_id": self.thread_id,
            "kind": self.kind,
            "ok": self.ok,
            "removed_paths": list(self.removed_paths),
            "related_removed": list(self.related_removed),
            "error": self.error,
        }
