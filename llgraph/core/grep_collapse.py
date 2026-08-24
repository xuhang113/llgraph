"""Grep 结果折叠：命中过多时改为文件统计 + 样例（对齐 Cursor / Claude Code）。

宽 pattern 全仓 grep 时，rg -m 是「每文件」上限，不是全局上限；再叠 ±N 行上下文，
单次工具结果可达数万字符，下一轮首 token 变慢，模型还以为 80 条就是全部。
本模块：先按文件计数，再决定 content / files / count，并始终给出真实总数。
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CONTENT_HEAD_LIMIT = 40
MAX_FILES_IN_TABLE = 25
MAX_SAMPLE_HITS = 8
MAX_SAMPLE_FILES = 6
MAX_PER_FILE_CONTENT = 8
AUTO_COLLAPSE_MATCHES = 28
AUTO_COLLAPSE_FILES = 12
AUTO_COLLAPSE_ESTIMATED_CHARS = 12_000

_MODE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "content": "content",
    "files": "files",
    "files_with_matches": "files",
    "count": "count",
    "count_matches": "count",
}


@dataclass(frozen=True)
class FileMatchCount:
    """单个文件的命中行数。"""

    path: str
    count: int


@dataclass(frozen=True)
class GrepPlan:
    """一次 grep 的展示计划。"""

    mode: str
    auto_collapsed: bool
    total_matches: int
    total_files: int
    head_limit: int
    file_counts: tuple[FileMatchCount, ...]
    hidden_files: int
    content_paths: tuple[str, ...]
    content_limit: int
    max_per_file: int
    needs_content: bool


def normalize_output_mode(raw: str | None) -> str:
    """
    规范化 output_mode（兼容 Claude Code 的 files_with_matches / count）。

    @param raw 模型传入值
    @return auto | content | files | count
    """
    key = (raw or "auto").strip().lower()
    return _MODE_ALIASES.get(key, "auto")


def estimate_content_chars(match_count: int, context_lines: int) -> int:
    """
    估算把全部命中展开成 content 时的字符数。

    @param match_count 命中行数
    @param context_lines 上下文明细行数
    @return 估算字符数
    """
    lines_per_hit = (2 * max(0, context_lines) + 1) + 2
    return max(0, match_count) * lines_per_hit * 70


def plan_grep(
    file_counts: list[FileMatchCount],
    *,
    output_mode: str = "auto",
    head_limit: int = 0,
    context_lines: int = 5,
) -> GrepPlan:
    """
    根据文件计数决定展示形态。

    @param file_counts 各文件命中数
    @param output_mode auto/content/files/count
    @param head_limit content 最多展示条数；≤0 用默认
    @param context_lines 上下文行数（影响 auto 折叠阈值）
    @return 展示计划
    """
    total_files = len(file_counts)
    total_matches = sum(item.count for item in file_counts)
    mode_in = normalize_output_mode(output_mode)
    head = head_limit if head_limit and head_limit > 0 else DEFAULT_CONTENT_HEAD_LIMIT
    head = max(1, min(500, int(head)))

    sorted_counts = tuple(sorted(file_counts, key=lambda item: (-item.count, item.path)))
    display_counts = sorted_counts[:MAX_FILES_IN_TABLE]
    hidden = max(0, total_files - len(display_counts))

    auto_collapsed = False
    mode = mode_in
    if mode_in == "auto":
        estimated = estimate_content_chars(total_matches, context_lines)
        if (
            total_matches > AUTO_COLLAPSE_MATCHES
            or total_files > AUTO_COLLAPSE_FILES
            or estimated > AUTO_COLLAPSE_ESTIMATED_CHARS
        ):
            mode = "files"
            auto_collapsed = True
        else:
            mode = "content"

    if mode == "count":
        return GrepPlan(
            mode=mode,
            auto_collapsed=auto_collapsed,
            total_matches=total_matches,
            total_files=total_files,
            head_limit=head,
            file_counts=display_counts,
            hidden_files=hidden,
            content_paths=(),
            content_limit=0,
            max_per_file=0,
            needs_content=False,
        )

    if mode == "files":
        return GrepPlan(
            mode=mode,
            auto_collapsed=auto_collapsed,
            total_matches=total_matches,
            total_files=total_files,
            head_limit=head,
            file_counts=display_counts,
            hidden_files=hidden,
            content_paths=tuple(item.path for item in sorted_counts[:MAX_SAMPLE_FILES]),
            content_limit=MAX_SAMPLE_HITS,
            max_per_file=2,
            needs_content=True,
        )

    content_paths: list[str] = []
    accumulated = 0
    for item in sorted_counts:
        content_paths.append(item.path)
        accumulated += item.count
        if accumulated >= head:
            break
    per_file = head if total_files <= 2 else MAX_PER_FILE_CONTENT
    per_file = max(1, min(head, per_file))
    return GrepPlan(
        mode="content",
        auto_collapsed=False,
        total_matches=total_matches,
        total_files=total_files,
        head_limit=head,
        file_counts=display_counts,
        hidden_files=hidden,
        content_paths=tuple(content_paths),
        content_limit=head,
        max_per_file=per_file,
        needs_content=True,
    )


def format_hit_block(
    rel: str,
    line_no: int,
    file_lines: list[str],
    *,
    context_lines: int,
) -> str:
    """
    格式化单条命中（与 ripgrep -C 块风格一致）。

    @param rel 相对路径
    @param line_no 命中行号（从 1 计）
    @param file_lines 文件全部行
    @param context_lines 上下文半径
    @return 文本块
    """
    if line_no < 1 or line_no > len(file_lines):
        return f"{rel}:{line_no}: (行号越界)"
    if context_lines <= 0:
        snippet = file_lines[line_no - 1].strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        return f"{rel}:{line_no}: {snippet}"

    start = max(1, line_no - context_lines)
    end = min(len(file_lines), line_no + context_lines)
    body: list[str] = [f"--- {rel}:{line_no} ---"]
    for ln in range(start, end + 1):
        snippet = file_lines[ln - 1].strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        prefix = ">>>" if ln == line_no else "   "
        body.append(f"{prefix} {ln}| {snippet}")
    return "\n".join(body)


def _format_file_table(file_counts: tuple[FileMatchCount, ...], hidden_files: int) -> list[str]:
    """渲染「命中数  路径」表。"""
    lines = ["按命中数："]
    for item in file_counts:
        lines.append(f"  {item.count:>4}  {item.path}")
    if hidden_files > 0:
        lines.append(f"  （另 {hidden_files} 个文件未列出）")
    return lines


def format_grep_result(
    *,
    plan: GrepPlan,
    pattern: str,
    path: str,
    file_glob: str,
    content_blocks: list[str],
    context_lines: int,
    engine: str,
) -> str:
    """
    渲染 grep 工具返回文案。

    @param plan 展示计划
    @param pattern 搜索模式
    @param path 搜索根
    @param file_glob 文件名限制
    @param content_blocks 已取回的命中块
    @param context_lines 上下文行数
    @param engine ripgrep 或 python
    @return 工具返回文本
    """
    glob_note = f" file_glob={file_glob!r}" if file_glob.strip() else ""
    ctx_note = f"，含上下文 ±{context_lines}" if context_lines > 0 else ""
    shown = len(content_blocks)
    header_bits = [
        f"pattern={pattern!r}",
        f"path={path!r}{glob_note}",
        f"共 {plan.total_matches} 处 / {plan.total_files} 个文件",
        engine,
    ]

    lines: list[str] = []
    if plan.auto_collapsed:
        lines.append(
            "【grep 已折叠】"
            + " ".join(header_bits)
            + f"（auto：完整 content 会淹没上下文{ctx_note}）。"
        )
        lines.extend(_format_file_table(plan.file_counts, plan.hidden_files))
        if content_blocks:
            lines.append(f"样例 {shown} 处{ctx_note}：")
            lines.append("")
            lines.append("\n\n".join(content_blocks))
        lines.append("")
        lines.append(
            "请收窄 path / file_glob / pattern 后再 grep_files；"
            "或对上表路径 read_files。"
            "禁止同一宽 pattern 再全仓 grep。"
        )
        return "\n".join(lines)

    if plan.mode == "count":
        lines.append("【count】" + " ".join(header_bits))
        lines.extend(_format_file_table(plan.file_counts, plan.hidden_files))
        return "\n".join(lines)

    if plan.mode == "files":
        lines.append("【files】" + " ".join(header_bits))
        lines.extend(_format_file_table(plan.file_counts, plan.hidden_files))
        if content_blocks:
            lines.append(f"样例 {shown} 处{ctx_note}：")
            lines.append("")
            lines.append("\n\n".join(content_blocks))
        lines.append("")
        lines.append("需要逐行时请收窄范围后 output_mode=content，或直接 read_files。")
        return "\n".join(lines)

    if plan.total_matches > shown:
        lines.append(
            f"匹配结果（显示 {shown}/{plan.total_matches} 处，"
            f"{plan.total_files} 个文件，{engine}{ctx_note}）："
        )
    else:
        lines.append(
            f"匹配结果 {plan.total_matches} 处 / {plan.total_files} 个文件"
            f"（{engine}{ctx_note}）："
        )
    if content_blocks:
        lines.append("")
        lines.append("\n\n".join(content_blocks))
    if plan.total_matches > shown:
        lines.append("")
        lines.append(
            f"还有 {plan.total_matches - shown} 处未列出。"
            "请收窄 path/file_glob/pattern，或改 output_mode=files。"
        )
    return "\n".join(lines)
