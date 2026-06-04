"""工作区文件检索、读写的 LangChain 工具（工厂方法按只读/可写模式注册）。"""

import fnmatch
import re
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.survey.edit_confirm import EditConfirmGate
from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.cli.search_terms import build_search_terms
from llgraph.session.session_edits import SessionEditTracker
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.core.workspace import (
    MAX_GREP_MATCHES,
    MAX_LIST_ENTRIES,
    MAX_READ_BYTES,
    MAX_SEARCH_RESULTS,
    WorkspaceContext,
)

# 多词检索时每个词最多贡献的文件名匹配数
_MAX_MATCHES_PER_TERM = 40

# 参与内容检索的文本扩展名
_TEXT_SUFFIXES = frozenset({
    ".py", ".java", ".kt", ".go", ".rs", ".js", ".ts", ".tsx", ".jsx",
    ".md", ".txt", ".yaml", ".yml", ".json", ".xml", ".properties",
    ".sql", ".sh", ".zsh", ".toml", ".ini", ".cfg", ".html", ".css",
    ".vue", ".gradle", ".pom", ".mdc",
})


def _is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return True
    if path.name in ("Dockerfile", "Makefile", "Jenkinsfile", ".gitignore"):
        return True
    return path.suffix == ""


SEARCH_REPLACE_TOOL_DESC = (
    "局部替换工作区文本文件内容（需 -w）。修改已有代码或 Markdown 某一节时优先使用。"
    "old_string 必须与磁盘完全一致（含缩进）；默认要求唯一匹配。"
)

WRITE_FILE_TOOL_DESC = (
    "创建新文件或整文件覆盖（需 -w）。每次调用必须同时提供 path 与 content，禁止只传 path。"
    "长文档（超过约 8000 字符）勿一次写全：先写标题与目录骨架，再用 append_file 或 search_replace 分节追加。"
    "Markdown 路径可直接使用 .md 后缀。"
)

APPEND_FILE_TOOL_DESC = (
    "向工作区文件末尾追加文本（需 -w）；文件不存在则创建。每次必须提供 path 与 content。"
    "长文档分块写入时，第一节用 write_file，后续各节用 append_file（单次 content 建议 <8000 字符）。"
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


def _hint_lines_for_needle(text: str, needle: str, limit: int = 5) -> str:
    """
    old_string 未命中时给出相近行提示。

    @param text 文件全文
    @param needle 待匹配片段
    @param limit 最多行数
    @return 提示文本
    """
    if not needle.strip():
        return ""
    first_line = needle.split("\n")[0].strip()
    if len(first_line) < 4:
        first_line = needle[:40].strip()
    hints: list[str] = []
    for idx, line in enumerate(text.splitlines(), 1):
        if first_line and first_line in line:
            hints.append(f"  L{idx}: {line[:100]}")
            if len(hints) >= limit:
                break
    if not hints:
        return ""
    return "相近行:\n" + "\n".join(hints)


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
    edit_confirm_gate: EditConfirmGate | None = None,
) -> list:
    """
    按工作区上下文创建文件类工具；写工具仅在 ctx.allow_write 为 True 时注册。

    @param ctx 工作区上下文
    @param edit_tracker 会话编辑账本（-w 时传入）
    @param on_file_changed 写成功后回调相对路径（如 watch 通知）
    @param write_failure_tracker 写工具失败计数（用于下一轮提醒）
    @param edit_confirm_gate 写前终端确认（Claude 风格菜单）
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

    def _confirm_write(rel: str, action_label: str) -> str | None:
        """
        写前确认；拒绝时返回错误文案。

        @param rel 相对路径
        @param action_label 动作描述
        @return 错误说明；允许时 None
        """
        if edit_confirm_gate is None:
            return None
        if edit_confirm_gate.confirm_write(rel, action_label):
            return None
        return f"用户已拒绝写入: {rel}（可在终端重新确认或输入继续说明）"

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

    def list_directory(path: str = ".") -> str:
        """
        列出目录下的文件和子目录（相对工作区路径）。

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
        """
        按关键字匹配路径：先扫工作区顶层目录（monorepo 项目名），再深度遍历文件。
        """
        matches: list[str] = []
        seen: set[str] = set()
        base = ctx.resolve_path(path)

        def add(rel: str) -> bool:
            if rel in seen:
                return len(matches) >= limit
            if glob_pattern and glob_pattern not in ("**/*", "*"):
                name = Path(rel.rstrip("/")).name
                if not fnmatch.fnmatch(name, glob_pattern.lstrip("**/")):
                    if not fnmatch.fnmatch(rel, glob_pattern):
                        return len(matches) >= limit
            seen.add(rel)
            matches.append(rel)
            return len(matches) >= limit

        if keyword_lower and base.is_dir():
            try:
                entries = sorted(
                    base.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except OSError:
                entries = []
            for entry in entries:
                if entry.is_dir() and ctx.should_skip_dir(entry.name):
                    continue
                try:
                    rel = entry.relative_to(ctx.root).as_posix()
                except ValueError:
                    continue
                if entry.is_dir():
                    rel = f"{rel}/"
                if keyword_lower in rel.lower():
                    if add(rel):
                        return matches

        for rel in ctx.iter_files(path):
            if keyword_lower and keyword_lower not in rel.lower():
                continue
            if add(rel):
                return matches
        return matches

    def search_files(
        keyword: str,
        path: str = ".",
        glob_pattern: str = "**/*",
    ) -> str:
        """
        按单个关键字在文件名/路径中查找（不读内容）。多词或业务主题请用 search_workspace。

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
            return "未找到匹配文件。建议改用 search_workspace 并填写 topic/keywords。"
        return "匹配文件（相对工作区）:\n" + "\n".join(matches)

    def search_workspace(
        keywords: str,
        topic: str = "",
        path: str = ".",
        include_content: bool = True,
    ) -> str:
        """
        多关键词并集检索工作区（文件名 + 可选内容 grep）。

        同义词、英文路径、项目缩写等须由你在 keywords 中一次给出多个（5～12 个为宜），
        工具不做业务词典映射。topic 仅作补充切分（整句、去「业务/服务」等后缀、提取英文词）。

        @param keywords 必填，逗号分隔，如 live,livestream,broadcast,acme-live,直播
        @param topic 可选主题描述，如「直播业务」
        @param path 起始相对目录
        @param include_content 是否 grep 文件内容（默认 True）
        """
        terms = build_search_terms(topic, keywords)
        if not terms:
            return (
                "keywords 不能为空。请根据用户主题自行列出多个检索词，"
                "例如 keywords='live,livestream,broadcast,streaming,room'。"
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

        if include_content and terms:
            # 内容检索：多词 OR，限制单次上限
            escaped = [re.escape(t) for t in terms if t.strip()]
            if escaped:
                pattern = "|".join(escaped)
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    regex = re.compile(re.escape(terms[0]), re.IGNORECASE)

                content_hits: list[str] = []
                for rel in ctx.iter_files(path):
                    full = ctx.resolve_path(rel)
                    if not full.is_file() or not _is_probably_text(full):
                        continue
                    try:
                        if full.stat().st_size > MAX_READ_BYTES:
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
    ) -> str:
        """
        在工作区文本文件中搜索内容（正则或普通子串）。

        @param pattern 搜索模式（按正则解析；失败则按子串）
        @param path 起始相对目录
        @param file_glob 可选，限制文件名，如 *.java
        """
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        results: list[str] = []
        for rel in ctx.iter_files(path, name_glob=file_glob or None):
            full = ctx.resolve_path(rel)
            if not full.is_file() or not _is_probably_text(full):
                continue
            try:
                if full.stat().st_size > MAX_READ_BYTES:
                    continue
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    snippet = line.strip()
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    results.append(f"{rel}:{line_no}: {snippet}")
                    if len(results) >= MAX_GREP_MATCHES:
                        return (
                            "匹配结果（已达上限）:\n" + "\n".join(results)
                        )

        if not results:
            return "未找到匹配内容。"
        return "匹配结果:\n" + "\n".join(results)

    def read_file(
        path: str,
        start_line: int = 1,
        end_line: int = 0,
    ) -> str:
        """
        读取工作区内单个文本文件（可指定行号范围）。

        @param path 相对工作区的文件路径
        @param start_line 起始行号，从 1 开始
        @param end_line 结束行号（含）；0 表示读到文件末尾
        """
        target = ctx.resolve_path(path)
        if not target.is_file():
            return f"文件不存在: {path}"

        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return (
                f"文件过大 ({size} 字节)，上限 {MAX_READ_BYTES}。"
                "请用 start_line/end_line 分段读取，或缩小范围。"
            )

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"读取失败: {exc}"

        lines = text.splitlines()
        start = max(1, start_line)
        end = len(lines) if end_line <= 0 else min(end_line, len(lines))
        if start > len(lines):
            return f"文件共 {len(lines)} 行，start_line 超出范围。"

        selected = lines[start - 1 : end]
        header = f"--- {path} (行 {start}-{end} / 共 {len(lines)} 行) ---\n"
        body = "\n".join(f"{start + i}| {line}" for i, line in enumerate(selected))
        return header + body

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
        target = ctx.resolve_path(rel)
        action = "覆盖" if target.is_file() else "创建"
        denied = _confirm_write(rel, f"是否{action}文件")
        if denied:
            return denied
        if edit_tracker is not None and target.is_file():
            edit_tracker.ensure_snapshot(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _after_write(rel, "write", old_part="", new_part=content)
        if write_failure_tracker is not None:
            write_failure_tracker.note_success()
        hint = _chunk_size_hint(content, chunk_max)
        return f"已写入 {rel}（{len(content)} 字符）{hint}"

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
        target = ctx.resolve_path(rel)
        denied = _confirm_write(rel, "是否追加内容到文件")
        if denied:
            return denied
        if target.is_file():
            if edit_tracker is not None:
                edit_tracker.ensure_snapshot(rel)
            existing = target.read_text(encoding="utf-8")
            new_text = existing + content
        else:
            new_text = content
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text, encoding="utf-8")
        _after_write(rel, "append", old_part="", new_part=content)
        if write_failure_tracker is not None:
            write_failure_tracker.note_success()
        hint = _chunk_size_hint(content, chunk_max)
        return f"已追加 {rel}（+{len(content)} 字符，共 {len(new_text)} 字符）{hint}"

    def search_replace(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """
        局部替换文件内容（需要 -w）；修改已有代码时优先于 write_file。

        @param path 相对工作区的文件路径
        @param old_string 必须与文件中片段完全一致（含缩进与换行）
        @param new_string 替换后的文本
        @param replace_all 是否替换全部匹配项
        """
        ctx.ensure_write_allowed()
        rel = path.strip().lstrip("/")
        denied = _confirm_write(rel, "是否编辑文件")
        if denied:
            return denied
        target = ctx.resolve_path(rel)
        if not target.is_file():
            return f"文件不存在: {rel}"
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            return f"读取失败: {exc}"
        count = text.count(old_string)
        if count == 0:
            hint = _hint_lines_for_needle(text, old_string)
            msg = f"未找到 old_string（0 处匹配）: {rel}"
            if hint:
                msg += f"\n{hint}\n请先 read_file 核对缩进与换行。"
            return msg
        if edit_cfg.require_unique_match and not replace_all and count != 1:
            return (
                f"old_string 在 {rel} 中出现 {count} 次，不唯一。"
                "请扩大上下文使片段唯一，或设置 replace_all=true。"
            )
        if edit_tracker is not None:
            edit_tracker.ensure_snapshot(rel)
        if replace_all:
            new_text = text.replace(old_string, new_string)
            replacements = count
        else:
            new_text = text.replace(old_string, new_string, 1)
            replacements = 1
        target.write_text(new_text, encoding="utf-8")
        _after_write(
            rel,
            "search_replace",
            replacements=replacements,
            old_part=old_string,
            new_part=new_string,
        )
        if write_failure_tracker is not None:
            write_failure_tracker.note_success()
        return f"已替换 {rel}（{replacements} 处）"

    tools: list = [
        StructuredTool.from_function(
            func=list_directory,
            name="list_directory",
            description=list_directory.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=search_workspace,
            name="search_workspace",
            description=search_workspace.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=search_files,
            name="search_files",
            description=search_files.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=grep_files,
            name="grep_files",
            description=grep_files.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=read_file.__doc__ or "",
        ),
    ]

    if ctx.allow_write:
        tools.extend([
            StructuredTool.from_function(
                func=search_replace,
                name="search_replace",
                description=SEARCH_REPLACE_TOOL_DESC,
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
