"""工作区文件检索、读写的 LangChain 工具（工厂方法按只读/可写模式注册）。"""

import re
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.core.filesystem_tool_schemas import (
    GlobFilesInput,
    GrepFilesInput,
    ListDirectoryInput,
    ReadFileInput,
    ReadFilesInput,
    SearchReplaceInput,
)
from llgraph.core.edit_apply import (
    EditHunk,
    apply_edit_hunks,
    format_apply_failure,
    format_apply_success,
    format_write_snapshot,
    parse_replacements_arg,
)
from llgraph.core.edit_diagnostics import format_edit_diagnostics

from llgraph.config.catalog_paths import resolve_catalog_read_path
from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.cli.search_terms import build_search_terms
from llgraph.code_index.index_ready import code_index_is_ready
from llgraph.code_index.path_hits import match_paths_by_keyword
from llgraph.core.ripgrep_search import (
    ripgrep_available,
    ripgrep_content_in_files,
    ripgrep_count,
    ripgrep_files,
)
from llgraph.core.grep_collapse import (
    FileMatchCount,
    format_grep_result,
    format_hit_block,
    plan_grep,
)
from llgraph.core.read_focus import (
    format_focus_read,
    format_numbered_slice,
    hit_lines_for_path,
    should_focus_read,
)
from llgraph.core.text_file_types import is_probably_text_path, read_path_rejection_reason
from llgraph.session.session_edits import SessionEditTracker
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.core.workspace import (
    MAX_GREP_MATCHES,
    MAX_LIST_ENTRIES,
    MAX_SEARCH_RESULTS,
    WorkspaceContext,
)

# 多词检索时每个词最多贡献的文件名匹配数
_MAX_MATCHES_PER_TERM = 40

# path 超过该深度且 glob 未命中时，尝试扩大到仓库根（第一段路径）再搜一次
_AUTO_WIDEN_PATH_MIN_DEPTH = 3

# 单次 batch 读取上限
_MAX_BATCH_READ_FILES = 8
# 单次 batch 读取总输出上限（字节）
_MAX_BATCH_READ_TOTAL_BYTES = 800_000


def _normalize_path_prefix(path: str) -> str:
    """规范化相对工作区的 path 前缀。"""
    raw = (path or ".").strip().strip("/")
    return raw if raw else "."


def _repo_root_prefix(path: str) -> str:
    """
    取相对路径的第一段（通常为仓库目录名）。

    @param path 相对工作区 path
    @return 第一段或 .
    """
    norm = _normalize_path_prefix(path)
    if norm == ".":
        return "."
    return norm.split("/")[0]


def _path_depth(path: str) -> int:
    """相对 path 的目录深度（. 为 0）。"""
    norm = _normalize_path_prefix(path)
    if norm == ".":
        return 0
    return len(norm.split("/"))


def _glob_pattern_has_package_segments(glob_pattern: str) -> bool:
    """glob 是否含包路径片段（不仅是末尾 **/*.ext）。"""
    pattern = (glob_pattern or "").strip()
    if "/" not in pattern:
        return False
    trimmed = pattern.replace("**/", "").replace("**\\", "").strip("/")
    return "/" in trimmed


def _format_path_scope_hint(path: str, *, glob_pattern: str = "") -> str:
    """
    glob/grep 未命中时的 path 作用域说明与纠正建议。

    @param path 当前搜索根
    @param glob_pattern 可选 glob，用于判断是否跨包搜索
    @return 多行提示
    """
    lines = [
        f"说明: 仅在 path={path!r} 的**子树**内搜索，不包含兄弟目录。",
        'path 相对工作区根，无 cwd；**禁止** path 含 "../"。',
    ]
    root = _repo_root_prefix(path)
    if _glob_pattern_has_package_segments(glob_pattern) or _path_depth(path) >= 4:
        lines.append(
            f"若 glob/关键词指向其它 feature 包，请改用 path=\".\" 或 path=\"{root}\"，"
            "不要沿用 list_directory 后的深层 path。"
        )
    return "\n".join(lines)


def _format_empty_glob_message(
    path: str,
    glob_pattern: str,
    *,
    index_ready: bool,
) -> str:
    """
    glob 0 命中时的完整提示。

    @param path 搜索根
    @param glob_pattern glob 模式
    @param index_ready 索引是否就绪
    @return 工具返回文案
    """
    scope = _format_path_scope_hint(path, glob_pattern=glob_pattern)
    if index_ready:
        next_tools = "grep_files(pattern=...) 或 search_code_parallel(query=...)"
    else:
        next_tools = "grep_files、search_files 或 search_workspace"
    return (
        f"未找到匹配文件: glob={glob_pattern!r} path={path!r}。\n"
        f"{scope}\n"
        "也可能源文件不在工作区（仅 markdowns/docs/远程路径引用）。\n"
        f"请换更宽的 path 后重试，或改用 {next_tools}。\n"
        "禁止同 path 重复 glob。"
    )


def _format_empty_grep_message(
    path: str,
    pattern: str,
    *,
    index_ready: bool,
) -> str:
    """
    grep 0 命中时的完整提示。

    @param path 搜索根
    @param pattern 搜索模式
    @param index_ready 索引是否就绪
    @return 工具返回文案
    """
    scope = _format_path_scope_hint(path)
    if index_ready:
        tail = "可改用 search_code_parallel(query=...) 做语义检索。"
    else:
        tail = "可扩大 path=\".\" 或换 search_workspace。"
    return (
        f"未找到匹配内容: pattern={pattern!r} path={path!r}。\n"
        f"{scope}\n"
        "若 pattern 是业务概念而非代码符号，请一次 OR 合并实现 API/配置项/注释关键词，"
        "或改用 search_code_parallel(query=行为描述)。\n"
        f"{tail}\n"
        "禁止同 path 重复 grep；须换词表或换工具。"
    )


def _reject_unsafe_relative_path(path: str) -> str | None:
    """
    禁止 read/glob 使用含 ../ 的相对路径（无 cwd，易越界或猜错）。

    @param path 用户传入路径
    @return 错误文案；合法则 None
    """
    raw = (path or "").strip()
    if not raw:
        return "path 不能为空"
    if raw == ".":
        return None
    normalized = raw.replace("\\", "/")
    if normalized.startswith("../") or "/../" in normalized or normalized.endswith("/.."):
        return (
            f"路径非法: {path!r}（禁止 ../；请使用 search_code_parallel/glob_files "
            "返回的完整相对路径，勿拼接猜测）。"
        )
    return None


def _read_file_content(
    ctx: WorkspaceContext,
    path: str,
    *,
    start_line: int = 1,
    end_line: int = 0,
) -> tuple[str | None, str | None]:
    """
    读取单个文件并格式化为带行号正文。

    @param ctx 工作区上下文
    @param path 文件路径
    @param start_line 起始行
    @param end_line 结束行，0 表示到末尾
    @return (正文, 错误文案)；成功时错误为 None
    """
    unsafe = _reject_unsafe_relative_path(path)
    if unsafe:
        return None, unsafe
    try:
        target = resolve_catalog_read_path(
            ctx.root, path, sandbox=ctx.sandbox_policy
        )
    except ValueError as exc:
        return None, str(exc)
    if not target.is_file():
        return None, f"文件不存在: {path}"

    read_reject = read_path_rejection_reason(target, ctx.root)
    if read_reject:
        return None, read_reject

    size = target.stat().st_size
    max_read_bytes = ctx.max_read_bytes
    max_read_lines = ctx.max_read_lines
    if size > max_read_bytes:
        return None, (
            f"文件过大 ({size} 字节)，上限 {max_read_bytes}。"
            "请用 start_line/end_line 分段读取，或缩小范围。"
        )

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"读取失败: {exc}"

    lines = text.splitlines()
    if len(lines) == 0:
        return (
            f"--- {path} (空文件，0 行) ---\n"
            "（文件存在但无内容；请改读 src/ 下业务源码或其它非空文件。）",
            None,
        )

    start = max(1, start_line)
    if start > len(lines):
        return None, f"文件共 {len(lines)} 行，start_line 超出范围。"

    if should_focus_read(
        start_line=start,
        end_line=end_line,
        total_lines=len(lines),
    ):
        return (
            format_focus_read(
                path,
                lines,
                hit_lines=hit_lines_for_path(path),
            ),
            None,
        )

    end = len(lines) if end_line <= 0 else min(end_line, len(lines))
    truncated = False
    if (end - start + 1) > max_read_lines:
        end = start + max_read_lines - 1
        truncated = True
    body = format_numbered_slice(path, lines, start, end, total=len(lines))
    if truncated:
        body += (
            f"\n\n（已截断：单次最多 {max_read_lines} 行；"
            f"继续请 read_file(path, start_line={end + 1}, end_line=...)）"
        )
    return body, None


SEARCH_REPLACE_TOOL_DESC = (
    "局部替换工作区文本文件内容（需 -w）。修改已有代码时优先使用。"
    "old_string 尽量与磁盘一致；CRLF、行尾空白、缩进偏差、误贴 read_file 行号前缀会自动对齐；"
    "仍 0 命中且全文件有唯一高相似行窗时会 fuzzy 落地（对齐 Cursor Fast Apply / Codex apply_patch）。"
    "默认同文件多处须唯一，或 replace_all=true。"
    "同一文件多处修改优先 replacements=[{old_string,new_string},...] 一次提交，按顺序应用。"
    "成功后返回带行号的写入后快照；若引入语法错误会附 [语法诊断]，须在本轮修复。"
    "继续改同一文件须用该快照，勿用写入前的 read。"
)

WRITE_FILE_TOOL_DESC = (
    "创建新文件或整文件覆盖（需 -w）。每次调用必须同时提供 path 与 content，禁止只传 path。"
    "长文档（超过约 8000 字符）勿一次写全：先写标题与目录骨架，再用 append_file 或 search_replace 分节追加。"
    "Markdown 路径可直接使用 .md 后缀。"
    "成功后返回文件头部快照；若引入语法错误会附 [语法诊断]。后续局部修改请 search_replace 并引用该快照。"
)

APPEND_FILE_TOOL_DESC = (
    "向工作区文件末尾追加文本（需 -w）；文件不存在则创建。每次必须提供 path 与 content。"
    "长文档分块写入时，第一节用 write_file，后续各节用 append_file（单次 content 建议 <8000 字符）。"
    "成功后返回文件末尾快照；若引入语法错误会附 [语法诊断]。"
)


def _validate_write_content(
    tool_name: str,
    path: str,
    content: str | None,
    *,
    chunk_max_chars: int,
) -> str | None:
    """
    校验写工具 content 参数。

    @param tool_name 工具名
    @param path 相对路径
    @param content 正文
    @param chunk_max_chars 单次建议上限
    @return 错误说明；通过时返回 None
    """
    if not (path or "").strip():
        return (
            f"错误: {tool_name} 缺少有效 path。\n"
            f"示例: {tool_name}(path=\"markdowns/foo.md\", content=\"# 标题\\n\\n正文\")"
        )
    if content is None or not str(content).strip():
        return (
            f"错误: {tool_name} 缺少必填参数 content（禁止只传 path）。\n"
            f"必须一次调用同时提供 path 与 content。\n"
            f"示例: {tool_name}(path=\"{path.strip()}\", content=\"# 标题\\n\\n第一节内容...\")\n"
            f"长文档请分块: write_file 写骨架 → append_file / search_replace 逐节追加（每节 <{chunk_max_chars} 字符）。"
        )
    return None


def _chunk_size_hint(content: str, chunk_max_chars: int) -> str:
    """
    超长 content 时的追加提示。

    @param content 已写入正文
    @param chunk_max_chars 建议上限
    @return 空或提示句
    """
    if len(content) <= chunk_max_chars:
        return ""
    return (
        f" 提示: 本次 content 长度 {len(content)} 超过建议 {chunk_max_chars}，"
        "后续请用 append_file 或 search_replace 分节写入。"
    )


def _count_lines(chunk: str) -> int:
    """估算文本行数。"""
    if not chunk:
        return 0
    return chunk.count("\n") + (1 if not chunk.endswith("\n") else 0)


def create_filesystem_tools(
    ctx: WorkspaceContext,
    *,
    edit_tracker: SessionEditTracker | None = None,
    on_file_changed: Callable[[str], None] | None = None,
    write_failure_tracker: WriteFailureTracker | None = None,
) -> list:
    """
    按工作区上下文创建文件类工具；写工具仅在 ctx.allow_write 为 True 时注册。

    @param ctx 工作区上下文
    @param edit_tracker 会话编辑账本（-w 时传入）
    @param on_file_changed 写成功后回调相对路径（如 watch 通知）
    @param write_failure_tracker 写工具失败计数（用于下一轮提醒）
    @return LangChain Tool 列表
    """
    edit_cfg = resolve_edit_settings(ctx.root)
    chunk_max = (
        write_failure_tracker.chunk_max_chars
        if write_failure_tracker is not None
        else edit_cfg.write_chunk_max_chars
    )

    def _write_error(tool_name: str, path: str, content: str | None) -> str:
        msg = _validate_write_content(
            tool_name, path, content, chunk_max_chars=chunk_max
        )
        if msg is None:
            return ""
        if write_failure_tracker is not None:
            write_failure_tracker.note_failure(tool_name, msg)
        return msg

    def _after_write(rel: str, op: str, *, replacements: int = 1, old_part: str = "", new_part: str = "") -> str:
        """写成功后记账并通知 watch。"""
        if edit_tracker is not None:
            edit_tracker.record(
                rel,
                op,
                replacements=replacements,
                lines_removed=_count_lines(old_part) * replacements,
                lines_added=_count_lines(new_part) * replacements,
            )
        if on_file_changed is not None:
            on_file_changed(rel)
        return rel

    def _with_diagnostics(rel: str, body: str, new_text: str, *, old_text: str = "") -> str:
        if not edit_cfg.syntax_diagnostics:
            return body
        extra = format_edit_diagnostics(rel, new_text, old_text=old_text)
        if not extra:
            return body
        return f"{body.rstrip()}\n{extra}"

    def list_directory(path: str = ".") -> str:
        """
        列出目录下的文件和子目录（相对工作区路径）。

        **用途**：浏览**已知** path 下有什么；工作区顶层有哪些仓库用 path=\".\" **最多列一次**。
        **禁止**用 list 代替 grep 找业务逻辑；跨仓搜内容用 grep_files(path=\".\")。
        **禁止** run_shell_command ls。

        @param path 相对工作区的目录路径，默认 .
        """
        target = ctx.resolve_path(path)
        if not target.exists():
            return f"路径不存在: {path}"
        if not target.is_dir():
            return f"不是目录: {path}"

        lines: list[str] = []
        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            return f"无法列出目录: {exc}"

        for entry in entries[:MAX_LIST_ENTRIES]:
            suffix = "/" if entry.is_dir() else ""
            try:
                rel = entry.relative_to(ctx.root).as_posix()
            except ValueError:
                rel = entry.name
            if entry.is_dir() and ctx.should_skip_dir(entry.name):
                lines.append(f"[skip] {rel}{suffix}")
            else:
                lines.append(f"{rel}{suffix}")

        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"... 仅显示前 {MAX_LIST_ENTRIES} 项")
        return "\n".join(lines) if lines else "(空目录)"

    def _match_paths_by_keyword(
        keyword_lower: str,
        path: str,
        glob_pattern: str,
        limit: int,
    ) -> list[str]:
        """按关键字匹配路径（委托 path_hits 模块）。"""
        return match_paths_by_keyword(
            ctx,
            keyword_lower,
            path,
            glob_pattern,
            limit,
        )

    index_ready = code_index_is_ready(ctx.root)
    skip_dirs = ctx._extra_skip_dirs

    def glob_files(
        glob_pattern: str = "",
        path: str = ".",
        pattern: str = "",
    ) -> str:
        """
        按 glob 快速列出工作区文件（ripgrep --files）。

        path 是**搜索根（子树边界）**，相对工作区根，无 cwd；只在 path 子树内匹配，不含兄弟目录。
        默认 path=\".\" 或 \"<仓库名>\"；**禁止** path 含 \"../\"；**禁止**沿用 list_directory 深层 path
        去 glob 其它 feature 包（如 path=.../activity 却 glob **/basic/**）。

        示例：glob_pattern=\"**/collect_alert.sh\", path=\".\"
        glob_pattern=\"**/infra/dao/*.java\", path=\"auth-api\"

        @param glob_pattern 路径/文件名 glob，如 **/*.sh
        @param path 搜索根相对目录，默认 .（全工作区）
        @param pattern 兼容误传别名，等同 glob_pattern
        """
        effective_pattern = glob_pattern.strip() or pattern.strip()
        if not effective_pattern:
            return (
                "glob_pattern 必填（glob_files 用 glob_pattern，grep_files 才用 pattern 搜内容）。"
            )
        if not ripgrep_available():
            return (
                "错误: 未安装 ripgrep (rg)。请安装后重试，或改用 search_files / search_code_parallel。"
            )
        paths, err = ripgrep_files(
            ctx.root,
            effective_pattern,
            path_prefix=path,
            limit=MAX_SEARCH_RESULTS,
            skip_dirs=skip_dirs,
        )
        if err:
            return f"glob_files 失败: {err}"
        if not paths:
            norm_path = _normalize_path_prefix(path)
            wider = _repo_root_prefix(path)
            if (
                wider != "."
                and wider != norm_path
                and _path_depth(path) >= _AUTO_WIDEN_PATH_MIN_DEPTH
            ):
                wider_paths, widen_err = ripgrep_files(
                    ctx.root,
                    effective_pattern,
                    path_prefix=wider,
                    limit=MAX_SEARCH_RESULTS,
                    skip_dirs=skip_dirs,
                )
                if widen_err:
                    return f"glob_files 失败: {widen_err}"
                if wider_paths:
                    return (
                        f"在 path={path!r} 下未命中；已自动扩大到 path={wider!r}，"
                        f"找到 {len(wider_paths)} 个文件:\n"
                        + "\n".join(wider_paths)
                        + "\n\n"
                        + _format_path_scope_hint(path, glob_pattern=effective_pattern)
                        + "\n下次请直接使用更宽的 path，避免先窄后扩。"
                    )
            return _format_empty_glob_message(
                path, effective_pattern, index_ready=index_ready
            )
        return (
            f"匹配 {len(paths)} 个文件（glob={effective_pattern}, path={path}）:\n"
            + "\n".join(paths)
        )

    def search_files(
        keyword: str,
        path: str = ".",
        glob_pattern: str = "**/*",
    ) -> str:
        """
        按单个关键字在文件名/路径中查找（不读内容）。

        本工作区已向量化索引时**不注册此工具**；请用 search_code_parallel（已含路径匹配）。

        @param keyword 一个关键字（不区分大小写）
        @param path 起始相对目录，默认 .
        @param glob_pattern 文件名匹配模式，如 *.java
        """
        keyword_lower = keyword.strip().lower() if keyword else ""
        matches = _match_paths_by_keyword(
            keyword_lower,
            path,
            glob_pattern,
            MAX_SEARCH_RESULTS,
        )
        if not matches:
            return "未找到匹配文件（路径/文件名）。"
        return "匹配文件（相对工作区）:\n" + "\n".join(matches)

    def search_workspace(
        keywords: str,
        topic: str = "",
        path: str = ".",
        include_content: bool = True,
    ) -> str:
        """
        多关键词并集检索工作区（文件名 + 可选内容 grep）。

        索引已启用时请用 search_code_parallel，勿用本工具做探索性检索。
        同义词、英文路径、项目缩写等须由你在 keywords 中一次给出多个（5～12 个为宜），
        工具不做业务词典映射。topic 仅作补充切分（整句、去「业务/服务」等后缀、提取英文词）。

        @param keywords 必填，逗号分隔，如 auth,login,account,demo-auth,认证
        @param topic 可选主题描述，如「用户认证」
        @param path 起始相对目录
        @param include_content 是否 grep 文件内容（默认 True）
        """
        terms = build_search_terms(topic, keywords)
        if not terms:
            return (
                "keywords 不能为空。请根据用户主题自行列出多个检索词，"
                "例如 keywords='auth,login,account,session,user'。"
            )

        by_term: dict[str, list[str]] = {}
        all_paths: list[str] = []
        path_seen: set[str] = set()

        for term in terms:
            term_lower = term.lower()
            hits = _match_paths_by_keyword(
                term_lower,
                path,
                "**/*",
                _MAX_MATCHES_PER_TERM,
            )
            by_term[term] = hits
            for rel in hits:
                if rel not in path_seen:
                    path_seen.add(rel)
                    all_paths.append(rel)
                if len(all_paths) >= MAX_SEARCH_RESULTS:
                    break
            if len(all_paths) >= MAX_SEARCH_RESULTS:
                break

        lines: list[str] = [
            f"检索主题: {topic or '(未填)'}",
            f"扩展词({len(terms)}): {', '.join(terms[:20])}"
            + (" ..." if len(terms) > 20 else ""),
            f"文件名命中（去重）: {len(all_paths)} 个",
            "---",
        ]

        for term, hits in by_term.items():
            if not hits:
                continue
            lines.append(f"[{term}] 路径/文件名 {len(hits)} 个:")
            for rel in hits[:8]:
                lines.append(f"  {rel}")
            if len(hits) > 8:
                lines.append(f"  ... 该词还有 {len(hits) - 8} 个")

        if not all_paths:
            lines.append("（无文件名匹配，尝试 include_content 或补充 keywords）")

        content_hits: list[str] = []
        if include_content and terms:
            # 内容检索：多词 OR，限制单次上限
            escaped = [re.escape(t) for t in terms if t.strip()]
            if escaped:
                pattern = "|".join(escaped)
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    regex = re.compile(re.escape(terms[0]), re.IGNORECASE)

                for rel in ctx.iter_files(path):
                    full = ctx.resolve_path(rel)
                    if not full.is_file() or not is_probably_text_path(full, ctx.root):
                        continue
                    try:
                        if full.stat().st_size > ctx.max_read_bytes:
                            continue
                        text = full.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    for line_no, line in enumerate(text.splitlines(), start=1):
                        if regex.search(line):
                            snippet = line.strip()
                            if len(snippet) > 160:
                                snippet = snippet[:160] + "..."
                            content_hits.append(f"{rel}:{line_no}: {snippet}")
                            if len(content_hits) >= MAX_GREP_MATCHES:
                                break
                    if len(content_hits) >= MAX_GREP_MATCHES:
                        break

                lines.append("---")
                lines.append(f"内容命中: {len(content_hits)} 条（上限 {MAX_GREP_MATCHES}）")
                if content_hits:
                    lines.extend(content_hits[:MAX_GREP_MATCHES])
                else:
                    lines.append("（无内容匹配）")

        return "\n".join(lines)

    def grep_files(
        pattern: str,
        path: str = ".",
        file_glob: str = "",
        output_mode: str = "auto",
        head_limit: int = 0,
    ) -> str:
        """
        在工作区文本文件中搜索内容（ripgrep）。

        **首选检索工具**。**多个关键字/表名/类名/字段必须合并为一次调用**：
        `pattern="tb_foo|FooDO|FooMapper|biz_id"`，**禁止**多轮各搜一个词（表名一轮、类名一轮是反模式）。
        可与 read_files **同一条 assistant 消息内并行**。

        path 是**搜索根**，须为工作区内**已存在**的目录；**禁止猜**仓库名。
        不确定时用 path=\".\" 搜全工作区。

        默认 output_mode=auto：命中过多时折叠为「真实总数 + 文件统计 + 样例」
        （对齐 Cursor / Claude Code 的 files_with_matches），避免 ±上下文把下一轮打满。
        需要逐行时再收窄范围并设 output_mode=content。

        @param pattern 搜索模式（优先正则；非法则按字面量）
        @param path 搜索根相对目录，默认 .
        @param file_glob 可选文件名 glob 限制，如 *.md
        @param output_mode auto|content|files|count（兼容 files_with_matches / count_matches）
        @param head_limit content 最多条数；0 表示默认 40
        """
        from llgraph.context.context_settings import resolve_context_settings

        grep_ctx = resolve_context_settings(ctx.root).grep_context_lines
        try:
            head = int(head_limit or 0)
        except (TypeError, ValueError):
            head = 0
        if head <= 0:
            head = min(40, MAX_GREP_MATCHES)
        else:
            head = max(1, min(head, MAX_GREP_MATCHES))

        def _empty() -> str:
            return _format_empty_grep_message(
                path, pattern, index_ready=index_ready
            )

        def _path_error(err: str) -> str:
            return (
                f"grep_files 失败: {err}\n"
                "提示: path 须为已存在目录；不确定时用 path=\".\"，"
                "或 list_directory(path=\".\") 核对顶层仓库名后再 grep。"
            )

        def _render(counts: list[FileMatchCount], blocks: list[str], engine: str) -> str:
            if not counts:
                extra = ""
                if engine == "python":
                    extra = "（未安装 rg，遍历范围受限；建议安装 ripgrep。）"
                    if index_ready:
                        return _empty() + extra
                    return _format_empty_grep_message(
                        path, pattern, index_ready=False
                    ) + extra
                return _empty()
            planned = plan_grep(
                counts,
                output_mode=output_mode,
                head_limit=head,
                context_lines=grep_ctx,
            )
            return format_grep_result(
                plan=planned,
                pattern=pattern,
                path=path,
                file_glob=file_glob,
                content_blocks=blocks,
                context_lines=grep_ctx,
                engine=engine,
            )

        def _python_scan() -> tuple[list[FileMatchCount], list[str]]:
            try:
                regex = re.compile(pattern)
            except re.error:
                regex = re.compile(re.escape(pattern))
            counts: list[FileMatchCount] = []
            stored: dict[str, list[str]] = {}
            for rel in ctx.iter_files(path, name_glob=file_glob or None):
                full = ctx.resolve_path(rel)
                if not full.is_file() or not is_probably_text_path(full, ctx.root):
                    continue
                try:
                    if full.stat().st_size > ctx.max_read_bytes:
                        continue
                    text = full.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                file_lines = text.splitlines()
                n = 0
                file_blocks: list[str] = []
                for line_no, line in enumerate(file_lines, start=1):
                    if not regex.search(line):
                        continue
                    n += 1
                    if len(file_blocks) < 8:
                        file_blocks.append(
                            format_hit_block(
                                rel,
                                line_no,
                                file_lines,
                                context_lines=grep_ctx,
                            )
                        )
                if n:
                    counts.append(FileMatchCount(rel, n))
                    stored[rel] = file_blocks
            planned = plan_grep(
                counts,
                output_mode=output_mode,
                head_limit=head,
                context_lines=grep_ctx,
            )
            blocks: list[str] = []
            if planned.needs_content:
                for rel_path in planned.content_paths:
                    for block in stored.get(rel_path, [])[: planned.max_per_file]:
                        blocks.append(block)
                        if len(blocks) >= planned.content_limit:
                            return counts, blocks
            return counts, blocks

        if ripgrep_available():
            try:
                raw_counts, err = ripgrep_count(
                    ctx.root,
                    pattern,
                    path_prefix=path,
                    file_glob=file_glob,
                    skip_dirs=skip_dirs,
                )
            except Exception as exc:
                raw_counts, err = [], f"ripgrep_count 异常: {exc}"
            if err:
                if "路径不存在" in err or "超出工作区" in err:
                    return _path_error(err)
                counts, blocks = _python_scan()
                return _render(counts, blocks, "python")
            if not raw_counts:
                return _empty()
            counts = [FileMatchCount(rel, n) for rel, n in raw_counts]
            planned = plan_grep(
                counts,
                output_mode=output_mode,
                head_limit=head,
                context_lines=grep_ctx,
            )
            blocks: list[str] = []
            if planned.needs_content:
                try:
                    blocks, _c_err = ripgrep_content_in_files(
                        ctx.root,
                        pattern,
                        list(planned.content_paths),
                        limit=planned.content_limit,
                        context_lines=grep_ctx,
                        max_per_file=planned.max_per_file,
                    )
                except Exception:
                    counts_py, blocks = _python_scan()
                    return _render(counts_py, blocks, "python")
            return format_grep_result(
                plan=planned,
                pattern=pattern,
                path=path,
                file_glob=file_glob,
                content_blocks=blocks,
                context_lines=grep_ctx,
                engine="ripgrep",
            )

        counts, blocks = _python_scan()
        return _render(counts, blocks, "python")

    def read_file(
        path: str,
        start_line: int = 1,
        end_line: int = 0,
    ) -> str:
        """
        读取单个文本文件（可指定行号范围）。

        **对齐 Cursor / Claude Code**：
        - 多个路径 → **read_files 一次批量**（勿逐个 read_file 各占一轮）；
        - 大文件未指定行段 → 自动返回符号大纲 + 本问检索命中窗，再对目标符号带行段精读；
        - 已知符号 → **一次取完整函数/类**（约 80–300 行），勿 30 行一段分多轮；
        - 可与 grep_files 在**同一条 assistant 消息内并行** tool_calls。

        path 可为工作区相对路径，或 ~/.llgraph/skills|rules 绝对路径；**禁止**含 ../。
        仅读源码/配置/文档；**不支持** lib/、target/、.so/.jar 等库与二进制文件。

        @param path 文件路径
        @param start_line 起始行号，从 1 开始
        @param end_line 结束行号（含）；0 表示读到文件末尾（大文件会折叠为大纲）
        """
        body, err = _read_file_content(
            ctx, path, start_line=start_line, end_line=end_line
        )
        if err:
            return err
        return body or ""

    def read_files(
        paths: list[str],
        start_line: int = 1,
        end_line: int = 0,
    ) -> str:
        """
        一次批量读取多个文件（对齐 Cursor：规划后一次取数，减少 I/O 与 ReAct 轮次）。

        适用：search_code_parallel / glob_files / grep_files 已给出多个**完整相对路径**，
        需要对比或梳理多个类/模块时。**能批量就批量**，单次最多 8 个路径；**禁止** path 含 ../。
        **禁止**每个路径单独占一轮 LLM；与其它只读工具可在**同一条 assistant 消息中并行** tool_calls。
        大文件未指定行段时每个文件自动折叠为大纲+命中窗；精读请带 start_line/end_line。

        @param paths 工作区相对路径列表（完整路径，从检索结果复制）
        @param start_line 每个文件的起始行号，从 1 开始
        @param end_line 每个文件的结束行号（含）；0 表示到末尾
        """
        if not paths:
            return "paths 不能为空"
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in paths:
            p = (raw or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            cleaned.append(p)
        if not cleaned:
            return "paths 无有效路径"
        if len(cleaned) > _MAX_BATCH_READ_FILES:
            return (
                f"一次最多读取 {_MAX_BATCH_READ_FILES} 个文件（当前 {len(cleaned)} 个）。"
                "请分批调用 read_files，或对大文件用 start_line/end_line。"
            )

        blocks: list[str] = []
        errors: list[str] = []
        total_bytes = 0
        for path in cleaned:
            body, err = _read_file_content(
                ctx, path, start_line=start_line, end_line=end_line
            )
            if err:
                errors.append(f"[失败] {path}: {err}")
                continue
            if body:
                total_bytes += len(body.encode("utf-8", errors="replace"))
                if total_bytes > _MAX_BATCH_READ_TOTAL_BYTES:
                    errors.append(
                        f"[截断] 批量读取总输出已超过 {_MAX_BATCH_READ_TOTAL_BYTES} 字节上限；"
                        f"已读 {len(blocks)} 个文件，剩余未读。请减少 paths 或使用 start_line/end_line。"
                    )
                    break
                blocks.append(body)

        if not blocks and errors:
            return "批量读取全部失败:\n" + "\n".join(errors)
        header = f"批量读取 {len(blocks)}/{len(cleaned)} 个文件:\n\n"
        out = header + "\n\n".join(blocks)
        if errors:
            out += "\n\n--- 部分失败 ---\n" + "\n".join(errors)
        return out

    def write_file(path: str, content: str = "") -> str:
        """
        创建或覆盖工作区文件（需 -w）。path 与 content 必填；长文请分块 append_file。

        @param path 相对工作区的文件路径
        @param content 完整文件内容
        """
        err = _write_error("write_file", path, content)
        if err:
            return err
        ctx.ensure_write_allowed()
        rel = path.strip().lstrip("/")
        try:
            target = ctx.resolve_path(rel, for_write=True)
        except PermissionError as exc:
            return str(exc)
        old_text = ""
        if target.is_file():
            try:
                old_text = target.read_text(encoding="utf-8")
            except OSError:
                old_text = ""
        if edit_tracker is not None and target.is_file():
            edit_tracker.ensure_snapshot(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _after_write(rel, "write", old_part="", new_part=content)
        if write_failure_tracker is not None:
            write_failure_tracker.note_success()
        hint = _chunk_size_hint(content, chunk_max)
        total_lines = len(content.splitlines())
        snapshot = format_write_snapshot(
            rel,
            content,
            start_line=1,
            end_line=min(total_lines, 40) if total_lines else 0,
        )
        return _with_diagnostics(
            rel,
            f"已写入 {rel}（{len(content)} 字符）{hint}\n{snapshot}",
            content,
            old_text=old_text,
        )

    def append_file(path: str, content: str = "") -> str:
        """
        向文件末尾追加内容（需 -w）；不存在则创建。长文档分节请多次 append。

        @param path 相对工作区的文件路径
        @param content 追加的正文
        """
        err = _write_error("append_file", path, content)
        if err:
            return err
        ctx.ensure_write_allowed()
        rel = path.strip().lstrip("/")
        try:
            target = ctx.resolve_path(rel, for_write=True)
        except PermissionError as exc:
            return str(exc)
        old_text = ""
        if target.is_file():
            if edit_tracker is not None:
                edit_tracker.ensure_snapshot(rel)
            old_text = target.read_text(encoding="utf-8")
            new_text = old_text + content
        else:
            new_text = content
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text, encoding="utf-8")
        _after_write(rel, "append", old_part="", new_part=content)
        if write_failure_tracker is not None:
            write_failure_tracker.note_success()
        hint = _chunk_size_hint(content, chunk_max)
        total_lines = len(new_text.splitlines()) or (1 if new_text else 0)
        tail_start = max(1, total_lines - 39) if total_lines else 1
        snapshot = format_write_snapshot(
            rel, new_text, start_line=tail_start, end_line=total_lines
        )
        return _with_diagnostics(
            rel,
            f"已追加 {rel}（+{len(content)} 字符，共 {len(new_text)} 字符）{hint}\n{snapshot}",
            new_text,
            old_text=old_text,
        )

    def search_replace(
        path: str,
        old_string: str = "",
        new_string: str = "",
        replace_all: bool = False,
        replacements: list | None = None,
    ) -> str:
        """
        局部替换文件内容（需要 -w）；修改已有代码时优先于 write_file。

        @param path 相对工作区的文件路径
        @param old_string 待替换片段（容错匹配 CRLF/缩进/行尾空白）
        @param new_string 替换后的文本
        @param replace_all 是否替换全部匹配项
        @param replacements 同一文件多个 hunk，按顺序应用
        """
        ctx.ensure_write_allowed()
        rel = path.strip().lstrip("/")
        try:
            target = ctx.resolve_path(rel, for_write=True)
        except PermissionError as exc:
            return str(exc)
        if not target.is_file():
            return f"文件不存在: {rel}"
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            return f"读取失败: {exc}"
        hunks: list[EditHunk] = []
        if old_string:
            hunks.append(
                EditHunk(
                    old_string=old_string,
                    new_string=new_string,
                    replace_all=replace_all,
                )
            )
        hunks.extend(parse_replacements_arg(replacements))
        if not hunks:
            return "错误: search_replace 必须提供 old_string 或 replacements"
        applied = apply_edit_hunks(
            text,
            hunks,
            require_unique=edit_cfg.require_unique_match,
            allow_fuzzy=edit_cfg.fuzzy_apply,
        )
        if not applied.ok:
            return format_apply_failure(rel, applied)
        if edit_tracker is not None:
            edit_tracker.ensure_snapshot(rel)
        target.write_text(applied.new_text, encoding="utf-8")
        old_part = "\n".join(h.old_string for h in hunks)
        new_part = "\n".join(h.new_string for h in hunks)
        _after_write(
            rel,
            "search_replace",
            replacements=applied.replacements,
            old_part=old_part,
            new_part=new_part,
        )
        if write_failure_tracker is not None:
            write_failure_tracker.note_success()
        return _with_diagnostics(
            rel,
            format_apply_success(rel, applied, old_text=text),
            applied.new_text,
            old_text=text,
        )

    tools: list = [
        StructuredTool.from_function(
            func=list_directory,
            name="list_directory",
            description=list_directory.__doc__ or "",
            args_schema=ListDirectoryInput,
        ),
        StructuredTool.from_function(
            func=glob_files,
            name="glob_files",
            description=glob_files.__doc__ or "",
            args_schema=GlobFilesInput,
        ),
        StructuredTool.from_function(
            func=search_workspace,
            name="search_workspace",
            description=search_workspace.__doc__ or "",
        ),
    ]
    if not index_ready:
        tools.append(
            StructuredTool.from_function(
                func=search_files,
                name="search_files",
                description=search_files.__doc__ or "",
            ),
        )
    tools.extend([
        StructuredTool.from_function(
            func=grep_files,
            name="grep_files",
            description=grep_files.__doc__ or "",
            args_schema=GrepFilesInput,
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=read_file.__doc__ or "",
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            func=read_files,
            name="read_files",
            description=read_files.__doc__ or "",
            args_schema=ReadFilesInput,
        ),
    ])

    if ctx.allow_write:
        tools.extend([
            StructuredTool.from_function(
                func=search_replace,
                name="search_replace",
                description=SEARCH_REPLACE_TOOL_DESC,
                args_schema=SearchReplaceInput,
            ),
            StructuredTool.from_function(
                func=append_file,
                name="append_file",
                description=APPEND_FILE_TOOL_DESC,
            ),
            StructuredTool.from_function(
                func=write_file,
                name="write_file",
                description=WRITE_FILE_TOOL_DESC,
            ),
        ])

    return tools
