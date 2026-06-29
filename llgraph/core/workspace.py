"""工作区路径解析与安全边界。"""

from __future__ import annotations

import os
from pathlib import Path

from llgraph.permissions.file_write import require_file_write
from llgraph.permissions.paths import resolve_workspace_path
from llgraph.sandbox.policy import SandboxPolicy

# 默认跳过检索/遍历的目录名
_SKIP_DIR_NAMES = frozenset({
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "target",
    ".idea",
    ".cursor",
})

# 单文件读取默认上限（字节 / 行）；工作区可在 agent.json context 覆盖
DEFAULT_READ_FILE_MAX_BYTES = 600_000
DEFAULT_READ_FILE_MAX_LINES = 6000

# 内容检索最多返回条数
MAX_GREP_MATCHES = 80

# 文件名检索最多返回条数
MAX_SEARCH_RESULTS = 100

# 目录列表最多条目
MAX_LIST_ENTRIES = 200


class WorkspaceContext:
    """
    将 Agent 文件操作限制在 workspace_root 内，防止路径穿越。
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        allow_write: bool = False,
        extra_skip_dirs: frozenset[str] | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        max_read_bytes: int | None = None,
        max_read_lines: int | None = None,
    ) -> None:
        root = Path(workspace_root).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"工作区不存在或不是目录: {root}")
        self.root = root
        self.allow_write = allow_write
        self._extra_skip_dirs = extra_skip_dirs or frozenset()
        self.sandbox_policy = sandbox_policy
        self.max_read_bytes = max(
            50_000,
            int(max_read_bytes or DEFAULT_READ_FILE_MAX_BYTES),
        )
        self.max_read_lines = max(
            200,
            int(max_read_lines or DEFAULT_READ_FILE_MAX_LINES),
        )

    def resolve_path(self, relative_path: str, *, for_write: bool = False) -> Path:
        """
        解析相对工作区的路径，禁止跳出根目录。

        @param relative_path 相对路径，空或 "." 表示工作区根
        @param for_write 写操作前设为 True 以触发沙箱写校验
        @return 绝对路径
        """
        return resolve_workspace_path(
            self.root,
            relative_path,
            sandbox=self.sandbox_policy,
            for_write=for_write,
        )

    def ensure_write_allowed(self) -> None:
        """写操作前校验是否已启用 -w。"""
        require_file_write(allow_write=self.allow_write)

    def should_skip_dir(self, dir_name: str) -> bool:
        """是否跳过该目录（不参与检索与遍历）。"""
        return dir_name in _SKIP_DIR_NAMES or dir_name in self._extra_skip_dirs

    def iter_files(
        self,
        relative_dir: str = ".",
        *,
        name_glob: str | None = None,
    ):
        """
        遍历工作区下文件（跳过常见构建/依赖目录）。

        @param relative_dir 起始相对目录
        @param name_glob 可选，仅匹配文件名 glob，如 *.java
        @yield 相对工作区的文件路径字符串
        """
        base = self.resolve_path(relative_dir)
        if not base.is_dir():
            return

        count = 0
        for dirpath, dirnames, filenames in os.walk(base, topdown=True):
            dirnames[:] = [d for d in dirnames if not self.should_skip_dir(d)]
            for filename in filenames:
                if name_glob and not Path(filename).match(name_glob):
                    continue
                full = Path(dirpath) / filename
                try:
                    rel = full.relative_to(self.root)
                except ValueError:
                    continue
                yield rel.as_posix()
                count += 1
                if count >= MAX_SEARCH_RESULTS * 10:
                    return
