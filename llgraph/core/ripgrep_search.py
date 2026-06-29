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


_RG_CTX_LINE = re.compile(r"^(.+)-(\d+)-(.*)$")
_RG_MATCH_LINE = re.compile(r"^(.+):(\d+):(.*)$")


def _format_rg_context_blocks(
    stdout: str,
    workspace: Path,
    *,
    limit: int,
) -> list[str]:
    """
    解析 rg -C 输出为可读块（path:line + 上下文字行）。

    @param stdout rg 标准输出
    @param workspace 工作区根（转相对路径）
    @param limit 最多匹配块数
    @return 格式化块列表
    """
    root = workspace.expanduser().resolve()
    blocks: list[str] = []
    current_lines: list[tuple[int, str, bool]] = []
    current_rel = ""
    match_count = 0

    def flush_block() -> None:
        nonlocal match_count, current_lines, current_rel
        if not current_lines:
            return
        match_line = next((ln for ln, _, is_match in current_lines if is_match), None)
        if match_line is None:
            current_lines = []
            current_rel = ""
            return
        header = f"--- {current_rel}:{match_line} ---"
        body_lines: list[str] = []
        for line_no, text, is_match in current_lines:
            prefix = ">>>" if is_match else "   "
            clipped = text if len(text) <= 200 else text[:200] + "..."
            body_lines.append(f"{prefix} {line_no}| {clipped}")
        blocks.append(header + "\n" + "\n".join(body_lines))
        match_count += 1
        current_lines = []
        current_rel = ""

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "--":
            flush_block()
            if match_count >= limit:
                break
            continue
        match_m = _RG_MATCH_LINE.match(line)
        if match_m:
            abs_or_rel, line_no_s, snippet = match_m.group(1), match_m.group(2), match_m.group(3)
            try:
                rel = Path(abs_or_rel).resolve().relative_to(root).as_posix()
            except ValueError:
                rel = abs_or_rel
            try:
                line_no = int(line_no_s)
            except ValueError:
                continue
            if current_lines and rel != current_rel:
                flush_block()
                if match_count >= limit:
                    break
            current_rel = rel
            current_lines.append((line_no, snippet.strip(), True))
            continue
        ctx_m = _RG_CTX_LINE.match(line)
        if ctx_m:
            abs_or_rel, line_no_s, snippet = ctx_m.group(1), ctx_m.group(2), ctx_m.group(3)
            try:
                rel = Path(abs_or_rel).resolve().relative_to(root).as_posix()
            except ValueError:
                rel = abs_or_rel
            try:
                line_no = int(line_no_s)
            except ValueError:
                continue
            if not current_lines:
                current_rel = rel
            current_lines.append((line_no, snippet.strip(), False))

    flush_block()
    return blocks[:limit]


def ripgrep_content(
    workspace: Path,
    pattern: str,
    *,
    path_prefix: str = ".",
    file_glob: str = "",
    limit: int = 80,
    context_lines: int = 0,
    skip_dirs: frozenset[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], str]:
    """
    在工作区文本文件中搜索（rg -n，可选 -C 上下文）。

    @param workspace 工作区根
    @param pattern 正则或字面量
    @param path_prefix 起始相对目录
    @param file_glob 可选文件名 glob，如 *.md
    @param limit 最多匹配行数
    @param context_lines 每条命中附加上下文行数（0=仅匹配行）
    @param skip_dirs 额外跳过目录
    @return (格式化命中列表, 错误文案)
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
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
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

    if context_lines > 0:
        blocks = _format_rg_context_blocks(stdout, root, limit=limit)
        return blocks, ""

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
