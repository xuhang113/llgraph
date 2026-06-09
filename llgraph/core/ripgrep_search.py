"""ripgrep 封装：全工作区 Glob / Grep（对齐 Cursor Agent 工具）。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from llgraph.core.text_file_types import resolve_grep_suffixes

_DEFAULT_SKIP_DIRS: tuple[str, ...] = (
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
    ".llgraph",
)

_RG_BIN: str | None = None


def _decode_subprocess_bytes(raw: bytes | None) -> str:
    """
    将子进程 stdout/stderr 解码为 str；非法 UTF-8 用替换字符，避免整轮对话崩溃。

    @param raw 原始字节
    @return 解码后的文本
    """
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def ripgrep_available() -> bool:
    """本机是否可用 rg。"""
    global _RG_BIN
    if _RG_BIN is not None:
        return True
    found = shutil.which("rg")
    if found:
        _RG_BIN = found
        return True
    return False


def _rg_executable() -> str:
    if not ripgrep_available():
        raise RuntimeError("ripgrep (rg) 未安装")
    return _RG_BIN or "rg"


def _merge_skip_dirs(extra: frozenset[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    merged: list[str] = list(_DEFAULT_SKIP_DIRS)
    if extra:
        for name in extra:
            if name and name not in merged:
                merged.append(name)
    return tuple(merged)


def _rg_skip_glob_args(skip_dirs: tuple[str, ...]) -> list[str]:
    args: list[str] = []
    for name in skip_dirs:
        args.extend(["--glob", f"!{name}/**"])
    return args


def _rg_type_glob_args(suffixes: tuple[str, ...]) -> list[str]:
    if not suffixes:
        return []
    args: list[str] = []
    for suffix in suffixes:
        ext = suffix.lstrip(".")
        if ext:
            args.extend(["--glob", f"*.{ext}"])
    for name in ("Dockerfile", "Makefile", "Jenkinsfile", ".gitignore"):
        args.extend(["--glob", name])
    return args


def ripgrep_files(
    workspace: Path,
    glob_pattern: str,
    *,
    path_prefix: str = ".",
    limit: int = 100,
    skip_dirs: frozenset[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], str]:
    """
    按 glob 列出工作区内文件（rg --files）。

    @param workspace 工作区根
    @param glob_pattern 如 **/*.sh、**/collect_alert.sh
    @param path_prefix 起始相对目录
    @param limit 最多返回条数
    @param skip_dirs 额外跳过目录
    @return (相对路径列表, 错误/说明文案)
    """
    root = workspace.expanduser().resolve()
    search_root = (root / path_prefix.strip().lstrip("/")).resolve()
    if not str(search_root).startswith(str(root)):
        return [], "path_prefix 超出工作区"
    if not search_root.is_dir():
        return [], f"目录不存在: {path_prefix}"

    pattern = (glob_pattern or "**/*").strip()
    cmd = [
        _rg_executable(),
        "--files",
        "--iglob",
        pattern,
        *_rg_skip_glob_args(_merge_skip_dirs(skip_dirs)),
        str(search_root),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], "ripgrep 超时（>30s）"
    except OSError as exc:
        return [], f"ripgrep 启动失败: {exc}"

    stdout = _decode_subprocess_bytes(proc.stdout)
    stderr = _decode_subprocess_bytes(proc.stderr)

    if proc.returncode not in (0, 1):
        err = (stderr or stdout or "").strip()
        return [], f"ripgrep 失败 (exit {proc.returncode}): {err[:200]}"

    rel_paths: list[str] = []
    for line in stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            rel = Path(raw).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        rel_paths.append(rel)
        if len(rel_paths) >= limit:
            break
    return rel_paths, ""


def ripgrep_content(
    workspace: Path,
    pattern: str,
    *,
    path_prefix: str = ".",
    file_glob: str = "",
    limit: int = 80,
    skip_dirs: frozenset[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], str]:
    """
    在工作区文本文件中搜索（rg -n）。

    @param workspace 工作区根
    @param pattern 正则或字面量
    @param path_prefix 起始相对目录
    @param file_glob 可选文件名 glob，如 *.md
    @param limit 最多匹配行数
    @param skip_dirs 额外跳过目录
    @return (path:line: snippet 列表, 错误文案)
    """
    root = workspace.expanduser().resolve()
    search_root = (root / path_prefix.strip().lstrip("/")).resolve()
    if not str(search_root).startswith(str(root)):
        return [], "path_prefix 超出工作区"
    if not search_root.exists():
        return [], f"路径不存在: {path_prefix}"

    needle = (pattern or "").strip()
    if not needle:
        return [], "pattern 不能为空"

    merged_skip = _merge_skip_dirs(skip_dirs)
    cmd = [
        _rg_executable(),
        "-n",
        "--no-heading",
        "--color=never",
        "-m",
        str(max(1, limit)),
        *_rg_skip_glob_args(merged_skip),
    ]
    if file_glob.strip():
        cmd.extend(["--iglob", file_glob.strip()])
    else:
        cmd.extend(_rg_type_glob_args(resolve_grep_suffixes(root)))

    # 先按正则；失败则字面量
    try:
        re.compile(needle)
        cmd.extend(["-e", needle])
    except re.error:
        cmd.extend(["-F", needle])

    cmd.append(str(search_root))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=45,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], "ripgrep 超时（>45s）"
    except OSError as exc:
        return [], f"ripgrep 启动失败: {exc}"

    stdout = _decode_subprocess_bytes(proc.stdout)
    stderr = _decode_subprocess_bytes(proc.stderr)

    if proc.returncode not in (0, 1):
        err = (stderr or stdout or "").strip()
        return [], f"ripgrep 失败 (exit {proc.returncode}): {err[:200]}"

    hits: list[str] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        # rg 输出: /abs/path:line:content 或 path:line:content
        parts = text.split(":", 2)
        if len(parts) < 3:
            continue
        abs_or_rel, line_no, snippet = parts[0], parts[1], parts[2]
        try:
            rel = Path(abs_or_rel).resolve().relative_to(root).as_posix()
        except ValueError:
            rel = abs_or_rel
        snippet = snippet.strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        hits.append(f"{rel}:{line_no}: {snippet}")
        if len(hits) >= limit:
            break
    return hits, ""
