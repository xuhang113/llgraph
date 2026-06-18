"""工作区文本/可索引文件后缀（配置源：.llgraph/embedding.json）。"""

from __future__ import annotations

from pathlib import Path

# 默认参与向量索引的后缀（可被 embedding.json index.include_suffixes 覆盖）
DEFAULT_INDEX_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".mdc",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".properties",
    ".sql",
    ".sh",
    ".zsh",
    ".toml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".vue",
    ".gradle",
    ".pom",
    ".csv",
    ".proto",
)

# 默认参与 ripgrep / grep_files 的后缀（可与索引列表不同，通常更广）
DEFAULT_GREP_SUFFIXES: tuple[str, ...] = DEFAULT_INDEX_SUFFIXES

_SPECIAL_FILENAMES: frozenset[str] = frozenset({
    "Dockerfile",
    "Makefile",
    "Jenkinsfile",
    ".gitignore",
    ".dockerignore",
    "CODEOWNERS",
})

# read_file 拒绝的二进制/库文件后缀
_BINARY_READ_SUFFIXES: frozenset[str] = frozenset({
    ".so",
    ".dylib",
    ".dll",
    ".jar",
    ".war",
    ".ear",
    ".class",
    ".pack",
    ".a",
    ".o",
    ".bin",
    ".exe",
    ".dmg",
    ".iso",
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp4",
    ".mp3",
    ".db",
    ".dmp",
    ".lance",
    ".parquet",
    ".wasm",
    ".pyc",
    ".pyo",
})

# read_file 跳过的路径段（构建产物、依赖、原生库目录）
_READ_SKIP_PATH_SEGMENTS: frozenset[str] = frozenset({
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".dylibs",
    ".framework",
    ".xcframework",
    ".mypy_cache",
    ".pytest_cache",
})

# lib/、libs/ 下默认视为 native/第三方库，非业务源码
_LIB_DIR_SEGMENTS: frozenset[str] = frozenset({"lib", "libs"})


def _normalize_suffix(raw: str) -> str:
    text = raw.strip().lower()
    if not text:
        return ""
    return text if text.startswith(".") else f".{text}"


def _parse_suffix_list(raw: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace(";", ",").split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw]
    else:
        return default
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        suffix = _normalize_suffix(item)
        if suffix and suffix not in seen:
            seen.add(suffix)
            out.append(suffix)
    return tuple(out) if out else default


def resolve_index_suffixes(workspace: Path | None) -> tuple[str, ...]:
    """
    解析向量索引 include_suffixes（embedding.json → index.include_suffixes）。

    @param workspace 工作区根；None 时用默认
    @return 小写后缀元组（含前导点）
    """
    if workspace is None:
        return DEFAULT_INDEX_SUFFIXES
    from llgraph.code_index.embedding_config import load_embedding_config

    cfg = load_embedding_config(workspace.expanduser().resolve())
    index_cfg = cfg.get("index") if isinstance(cfg.get("index"), dict) else {}
    return _parse_suffix_list(index_cfg.get("include_suffixes"), DEFAULT_INDEX_SUFFIXES)


def resolve_grep_suffixes(workspace: Path | None) -> tuple[str, ...]:
    """
    解析 ripgrep / grep_files 后缀白名单（embedding.json → index.grep.include_suffixes）。

    @param workspace 工作区根；None 时用默认
    @return 小写后缀元组
    """
    if workspace is None:
        return DEFAULT_GREP_SUFFIXES
    from llgraph.code_index.embedding_config import load_embedding_config

    cfg = load_embedding_config(workspace.expanduser().resolve())
    index_cfg = cfg.get("index") if isinstance(cfg.get("index"), dict) else {}
    grep_cfg = index_cfg.get("grep")
    if isinstance(grep_cfg, dict) and grep_cfg.get("include_suffixes") is not None:
        return _parse_suffix_list(grep_cfg.get("include_suffixes"), DEFAULT_GREP_SUFFIXES)
    return _parse_suffix_list(index_cfg.get("include_suffixes"), DEFAULT_GREP_SUFFIXES)


def suffix_set(workspace: Path | None, *, for_index: bool) -> frozenset[str]:
    """
    @param workspace 工作区根
    @param for_index True=索引后缀，False=grep 后缀
    @return 后缀集合
    """
    if for_index:
        return frozenset(resolve_index_suffixes(workspace))
    return frozenset(resolve_grep_suffixes(workspace))


def is_probably_text_path(path: Path, workspace: Path | None = None) -> bool:
    """
    是否视为文本文件（grep 路径）。

    @param path 文件路径
    @param workspace 工作区根（解析 grep 后缀配置）
    @return 是否文本
    """
    if path.name in _SPECIAL_FILENAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in suffix_set(workspace, for_index=False):
        return True
    return suffix == ""


def read_path_rejection_reason(path: Path, workspace: Path | None = None) -> str | None:
    """
    read_file/read_files 是否应拒绝该路径。

    拒绝二进制、lib/ 原生库、target 等构建目录，以及无文本后缀的扩展名文件。

    @param path 绝对或相对文件路径
    @param workspace 工作区根（解析后缀白名单）
    @return 拒绝原因；可读时 None
    """
    suffix = path.suffix.lower()
    if suffix in _BINARY_READ_SUFFIXES:
        return (
            f"不支持读取库/二进制文件 ({suffix or path.name})；"
            "请读 src/ 下 .java/.py 等源码，或用 search_code_parallel/grep_files 定位。"
        )

    lowered_parts = [part.lower() for part in path.parts]
    for part in lowered_parts:
        if part in _READ_SKIP_PATH_SEGMENTS:
            return (
                f"不支持读取 {part}/ 下文件（构建产物或依赖目录）；"
                "请读 src/main/java 等源码路径。"
            )

    for idx, part in enumerate(lowered_parts):
        if part in _LIB_DIR_SEGMENTS and idx < len(lowered_parts) - 1:
            if path.name in _SPECIAL_FILENAMES:
                break
            if suffix not in suffix_set(workspace, for_index=False):
                return (
                    "不支持读取 lib/ 下库文件（.so/.dylib 或无文本后缀）；"
                    "请读业务源码路径，勿读 native/第三方库。"
                )

    if path.name in _SPECIAL_FILENAMES:
        return None
    if suffix in suffix_set(workspace, for_index=False):
        return None
    if not suffix:
        return (
            "不支持读取无后缀文件（可能为二进制/framework）；"
            "请使用带 .java/.py/.md 等后缀的完整源码路径。"
        )
    return (
        f"不支持读取该类型文件 ({suffix})；"
        "read_file 仅用于源码/配置/文档（见 embedding.json grep 后缀白名单）。"
    )


def is_readable_text_path(path: Path, workspace: Path | None = None) -> bool:
    """
    read_file 是否允许读取。

    @param path 文件路径
    @param workspace 工作区根
    @return 是否允许
    """
    return read_path_rejection_reason(path, workspace) is None


def is_indexable_path(path: Path, workspace: Path | None = None) -> bool:
    """
    是否应纳入向量索引扫描。

    @param path 文件路径
    @param workspace 工作区根
    @return 是否可索引
    """
    if path.name in _SPECIAL_FILENAMES:
        return True
    return path.suffix.lower() in suffix_set(workspace, for_index=True)
