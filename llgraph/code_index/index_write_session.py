"""索引写入会话：复用 Lance 表连接，减少 connect/open 开销。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llgraph.code_index.paths import DEFAULT_VECTOR_DIM
from llgraph.code_index.store import get_table


@dataclass
class IndexWriteSession:
    """
    单次索引运行内复用 Lance 表句柄。

    @param workspace 工作区根
    """

    workspace: Path
    _table: Any = field(default=None, init=False, repr=False)
    _vector_dim: int = field(default=DEFAULT_VECTOR_DIM, init=False)

    def _open_table(self, vector_dim: int) -> Any:
        if self._table is None or vector_dim != self._vector_dim:
            self._vector_dim = vector_dim
            self._table = get_table(self.workspace, vector_dim=vector_dim)
        return self._table

    @staticmethod
    def _quote_rel_path(rel: str) -> str:
        """Lance delete 谓词中的路径字面量。"""
        return '"' + rel.replace('"', '\\"') + '"'

    def delete_rel_paths(self, rel_paths: set[str]) -> None:
        """
        按 rel_path 删除旧 chunk（IN 批量谓词，避免每文件一次全表扫描）。

        仅用于「文件内容已变更、需替换旧向量」的路径；首次入库的文件不应调用。

        @param rel_paths 相对路径集合
        """
        if not rel_paths:
            return
        table = self._open_table(self._vector_dim)
        ordered = sorted(rel_paths)
        # 单次 IN 列表不宜过长
        in_batch = 40
        for offset in range(0, len(ordered), in_batch):
            chunk = ordered[offset : offset + in_batch]
            quoted = ", ".join(self._quote_rel_path(rel) for rel in chunk)
            predicate = f"rel_path IN ({quoted})"
            try:
                table.delete(predicate)
            except Exception:
                for rel in chunk:
                    try:
                        table.delete(f"rel_path = {self._quote_rel_path(rel)}")
                    except Exception:
                        pass

    def add_records(self, records: list[dict[str, Any]]) -> int:
        """
        单次 Lance add 写入多条 chunk。

        @param records chunk 记录
        @return 写入条数
        """
        if not records:
            return 0
        dim = len(records[0].get("vector", [])) or self._vector_dim
        table = self._open_table(dim)
        table.add(records)
        return len(records)
