"""记忆 Lance 批量写入会话（禁止逐条 delete）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llgraph.code_index.paths import DEFAULT_VECTOR_DIM
from llgraph.memory.paths import DELETE_BATCH_SIZE
from llgraph.memory.store import connect_memory_db, get_memory_table


@dataclass
class MemoryWriteSession:
    """
    单次整理/写入内复用 Lance 表句柄。

    @param user_id 用户 ID
    @param workspace_key 工作区键
    """

    user_id: str
    workspace_key: str
    _table: Any = field(default=None, init=False, repr=False)
    _vector_dim: int = field(default=DEFAULT_VECTOR_DIM, init=False)

    def _open_table(self, vector_dim: int) -> Any:
        if self._table is None or vector_dim != self._vector_dim:
            self._vector_dim = vector_dim
            self._table = get_memory_table(self.user_id, self.workspace_key, vector_dim=vector_dim)
        return self._table

    @staticmethod
    def _quote_id(memory_id: str) -> str:
        return '"' + memory_id.replace('"', '\\"') + '"'

    def delete_memory_ids(self, memory_ids: list[str]) -> int:
        """
        按 memory_id 批量删除（IN 谓词）。

        @param memory_ids ID 列表
        @return 请求删除条数
        """
        if not memory_ids:
            return 0
        table = self._open_table(self._vector_dim)
        ordered = sorted({mid.strip() for mid in memory_ids if mid.strip()})
        deleted = 0
        batch = DELETE_BATCH_SIZE
        for offset in range(0, len(ordered), batch):
            chunk = ordered[offset : offset + batch]
            quoted = ", ".join(self._quote_id(mid) for mid in chunk)
            predicate = f"memory_id IN ({quoted})"
            try:
                table.delete(predicate)
                deleted += len(chunk)
            except Exception:
                for mid in chunk:
                    try:
                        table.delete(f"memory_id = {self._quote_id(mid)}")
                        deleted += 1
                    except Exception:
                        pass
        return deleted

    def add_records(self, records: list[dict[str, Any]]) -> int:
        """
        批量 add。

        @param records 行记录
        @return 写入条数
        """
        if not records:
            return 0
        dim = len(records[0].get("vector", [])) or self._vector_dim
        table = self._open_table(dim)
        table.add(records)
        return len(records)
