"""终端 Markdown / JSON 渲染（仅经典终端展示层，不用于 API 返回）。

Web / 结构化结果应返回 Markdown 原文（见 display.assistant_content），
由 Web 服务自行渲染；本模块只供 trace 助手回复与显式 opt-in 的终端输出。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from typing import IO, Any

from llgraph.terminal.style import color_enabled, sty

_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.+)$")
_ORDERED_RE = re.compile(r"^(\s*)\d+\.\s+(.+)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")

# Rich Markdown 代码块在窄终端里行尾填充空格；捕获到字符串时需 rstrip
_RICH_JSON_THEME = "monokai"
# crop=True：按内容换行，不 pad 到终端宽度（Cursor 集成终端对行尾空格极不友好）
_RICH_PRINT_OPTS = {"crop": True, "soft_wrap": True, "overflow": "fold"}


def terminal_width(default: int = 100) -> int:
    """
    终端可用列宽。

    @param default 回退宽度
    @return 列数
    """
    try:
        return max(40, shutil.get_terminal_size(fallback=(default, 24)).columns)
    except OSError:
        return default


def markdown_render_enabled() -> bool:
    """
    是否启用 Markdown 渲染（TraceSession 可覆盖；环境变量 LLGRAPH_MD_RENDER）。

    @return 默认 True（交互 TTY）
    """
    raw = __import__("os").environ.get("LLGRAPH_MD_RENDER", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    stdout_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return bool(stdout_tty)


def _rich_import_ok() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def rich_render_enabled(*, session: Any = None) -> bool:
    """
    是否使用 Rich 渲染（默认关）。

    优先级：LLGRAPH_MD_RICH 环境变量 > TraceSession.use_rich（/trace rich）> 默认 False。

    @param session 可选追踪会话（交互 /trace rich 开关）
    @return 是否走 Rich（未安装 rich 时恒为 False）
    """
    if not _rich_import_ok():
        return False
    raw = __import__("os").environ.get("LLGRAPH_MD_RICH", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    if session is not None:
        return bool(getattr(session, "use_rich", False))
    return False


def resolve_rich_from_env() -> bool:
    """
    启动时从环境变量解析 Rich 开关（供 TraceSession 初始值）。

    @return LLGRAPH_MD_RICH=1 且已安装 rich 时为 True
    """
    return rich_render_enabled()


def looks_like_markdown(text: str) -> bool:
    """
    启发式判断内容是否适合 Markdown 渲染。

    @param text 正文
    @return 是否像 Markdown / 结构化文档
    """
    sample = text.strip()
    if not sample:
        return False
    if "```" in sample:
        return True
    if _HEADING_RE.search(sample):
        return True
    if "**" in sample or _INLINE_CODE_RE.search(sample):
        return True
    if sample.startswith("{") and sample.endswith("}"):
        return True
    lines = sample.splitlines()
    md_lines = sum(
        1
        for line in lines
        if line.startswith("#")
        or line.lstrip().startswith("- ")
        or line.lstrip().startswith("* ")
        or _ORDERED_RE.match(line)
    )
    return md_lines >= 2


def _strip_terminal_padding(text: str) -> str:
    """
    去掉 rich 等库为对齐产生的行尾空白。

    @param text 渲染结果
    @return 去空白后的文本
    """
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def _pretty_json_body(body: str) -> str | None:
    """
    尝试格式化 JSON 文本。

    @param body 原始文本
    @return 缩进 JSON 或 None
    """
    stripped = body.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return json.dumps(data, ensure_ascii=False, indent=2)


def _prettify_json_fences(text: str) -> str:
    """
    将 ```json 代码块内容缩进美化（便于 Syntax / 回退渲染）。

    @param text 原文
    @return 替换后的文本
    """
    lines = text.splitlines()
    out: list[str] = []
    in_json = False
    buf: list[str] = []

    def _flush() -> None:
        nonlocal buf
        pretty = _pretty_json_body("\n".join(buf))
        if pretty:
            out.extend(pretty.splitlines())
        else:
            out.extend(buf)
        buf = []

    for line in lines:
        m = _FENCE_RE.match(line.strip())
        if m:
            lang = (m.group(1) or "").lower()
            if not in_json:
                in_json = lang in ("json", "")
                out.append(line)
                buf = []
            else:
                _flush()
                out.append(line)
                in_json = False
            continue
        if in_json:
            buf.append(line)
        else:
            out.append(line)
    if in_json and buf:
        _flush()
    return "\n".join(out)


def _split_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """
    按 fenced code 拆分为 (kind, content) 块；kind 为 text|json|code。

    @param text 已预处理的 Markdown 文本
    @return 块列表
    """
    blocks: list[tuple[str, str]] = []
    in_fence = False
    lang = ""
    buf: list[str] = []
    prose: list[str] = []

    def _flush_prose() -> None:
        if prose:
            blocks.append(("text", "\n".join(prose).strip("\n")))
            prose.clear()

    def _flush_fence() -> None:
        nonlocal buf, lang
        body = "\n".join(buf)
        kind = "json" if lang == "json" or _pretty_json_body(body) else "code"
        blocks.append((kind, body.strip("\n")))
        buf = []
        lang = ""

    for line in text.splitlines():
        m = _FENCE_RE.match(line.strip())
        if m:
            if not in_fence:
                _flush_prose()
                in_fence = True
                lang = (m.group(1) or "").lower()
            else:
                in_fence = False
                _flush_fence()
            continue
        if in_fence:
            buf.append(line)
        else:
            prose.append(line)
    if in_fence and buf:
        _flush_fence()
    _flush_prose()
    return blocks


def _make_rich_console(file: IO[str], *, width: int | None = None) -> Any:
    """
    构建适配 IDE 终端的 Rich Console。

    @param file 输出流
    @param width 列宽
    @return rich.console.Console
    """
    from rich.console import Console

    cols = width or terminal_width()
    is_tty = hasattr(file, "isatty") and file.isatty()
    return Console(
        file=file,
        width=cols,
        soft_wrap=True,
        force_terminal=color_enabled() and is_tty,
        color_system="standard",
        highlight=False,
        tab_size=4,
        legacy_windows=False,
    )


def _needs_syntax_blocks(text: str) -> bool:
    """
    是否含 json/代码 fence，需 Syntax 增强（否则整篇 Markdown 即可）。

    @param text 原文
    @return 是否需要分块 Syntax
    """
    prepared = _prettify_json_fences(text)
    return any(kind in ("json", "code") for kind, _ in _split_fenced_blocks(prepared))


def _is_pure_json_document(text: str) -> bool:
    """
    整段内容是否为 JSON（无 Markdown 结构）。

    @param text 原文
    @return 是否纯 JSON
    """
    stripped = text.strip()
    if not stripped or "```" in stripped or _HEADING_RE.search(stripped):
        return False
    return _pretty_json_body(stripped) is not None


def _print_rich_whole_markdown(
    text: str,
    *,
    file: IO[str],
    indent: int = 0,
    width: int | None = None,
) -> None:
    """
    整篇交给 Rich Markdown 渲染（标题、列表、表格、代码块等统一处理）。

    @param text 原文
    @param file 输出流
    @param indent 左缩进列数
    @param width 内容区列宽
    """
    from rich.markdown import Markdown
    from rich.padding import Padding

    cols = width or max(40, terminal_width() - indent)
    console = _make_rich_console(file, width=cols)
    pad = (0, 0, 0, indent)
    console.print(
        Padding(Markdown(text), pad),
        **_RICH_PRINT_OPTS,
    )


def _print_rich_hybrid(
    text: str,
    *,
    file: IO[str],
    indent: int = 0,
    width: int | None = None,
) -> None:
    """
    Rich 分块渲染：正文 Markdown；json/代码块用 Syntax（比 Markdown 内嵌 code fence 高亮更好）。

    @param text 原文
    @param file 输出流
    @param indent 左缩进列数
    @param width 内容区列宽
    """
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.syntax import Syntax

    cols = width or max(40, terminal_width() - indent)
    console = _make_rich_console(file, width=cols)
    pad = (0, 0, 0, indent)
    prepared = _prettify_json_fences(text)

    for kind, content in _split_fenced_blocks(prepared):
        if not content.strip():
            continue
        if kind == "json":
            pretty = _pretty_json_body(content) or content
            console.print(
                Padding(
                    Syntax(
                        pretty,
                        "json",
                        theme=_RICH_JSON_THEME,
                        line_numbers=False,
                        word_wrap=True,
                    ),
                    pad,
                ),
                **_RICH_PRINT_OPTS,
            )
            continue
        if kind == "code":
            console.print(
                Padding(
                    Syntax(content, "text", word_wrap=True, theme=_RICH_JSON_THEME),
                    pad,
                ),
                **_RICH_PRINT_OPTS,
            )
            continue
        # 有 Rich 时 prose 一律走 Markdown（含纯文本段落）
        console.print(
            Padding(Markdown(content), pad),
            **_RICH_PRINT_OPTS,
        )


def _print_with_rich(
    text: str,
    *,
    file: IO[str],
    indent: int = 0,
    width: int | None = None,
) -> None:
    """
    Rich 统一入口：纯 JSON → Syntax；含 json fence → 分块；其余整篇 Markdown。

    @param text 原文
    @param file 输出流
    @param indent 左缩进列数
    @param width 内容区列宽
    """
    from rich.padding import Padding
    from rich.syntax import Syntax

    prepared = _prettify_json_fences(text)

    # 整段 JSON（Plan worker 结果等）
    if _is_pure_json_document(prepared):
        pretty_json = _pretty_json_body(prepared.strip())
        assert pretty_json is not None
        cols = width or max(40, terminal_width() - indent)
        console = _make_rich_console(file, width=cols)
        pad = (0, 0, 0, indent)
        console.print(
            Padding(
                Syntax(
                    pretty_json,
                    "json",
                    theme=_RICH_JSON_THEME,
                    line_numbers=False,
                    word_wrap=True,
                ),
                pad,
            ),
            **_RICH_PRINT_OPTS,
        )
        return

    if _needs_syntax_blocks(prepared):
        _print_rich_hybrid(prepared, file=file, indent=indent, width=width)
        return

    _print_rich_whole_markdown(prepared, file=file, indent=indent, width=width)


def print_terminal_formatted(
    text: str,
    *,
    file: IO[str] | None = None,
    indent: int = 2,
) -> bool:
    """
    直接渲染并打印到终端（推荐路径，避免 StringIO 填充问题）。

    @param text 助手正文
    @param file 输出流，默认 stdout
    @param indent 左缩进
    @return 是否已用 Rich 输出
    """
    if not rich_render_enabled() or not text.strip():
        return False
    try:
        _print_with_rich(text, file=file or sys.stdout, indent=indent)
        return True
    except Exception:
        return False


def _render_inline(text: str) -> str:
    """
    行内 **bold** 与 `code`。

    @param text 单行文本
    @return 着色文本
    """

    def _code(m: re.Match[str]) -> str:
        return sty(m.group(1), "path")

    def _bold(m: re.Match[str]) -> str:
        return sty(m.group(1), "bold")

    out = _INLINE_CODE_RE.sub(_code, text)
    out = _BOLD_RE.sub(_bold, out)
    return out


def _render_fallback(text: str) -> str:
    """
    无 rich 时的轻量 Markdown/JSON → ANSI。

    @param text 原文
    @return 渲染后文本
    """
    lines_out: list[str] = []
    in_fence = False
    fence_lang = ""
    fence_buf: list[str] = []

    def _flush_fence() -> None:
        nonlocal fence_buf, fence_lang
        body = "\n".join(fence_buf)
        pretty = _pretty_json_body(body) if fence_lang in ("json", "") else None
        payload = pretty if pretty is not None else body
        for line in payload.splitlines():
            lines_out.append(sty(line, "hint") if color_enabled() else line)
        fence_buf = []
        fence_lang = ""

    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line.strip())
        if fence_match:
            if not in_fence:
                in_fence = True
                fence_lang = (fence_match.group(1) or "").lower()
                fence_buf = []
            else:
                in_fence = False
                _flush_fence()
            continue
        if in_fence:
            fence_buf.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            style = "brand" if level <= 2 else "title"
            lines_out.append(sty(title, style) if color_enabled() else title)
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            indent = bullet.group(1)
            body = _render_inline(bullet.group(2))
            prefix = sty("• ", "label") if color_enabled() else "- "
            lines_out.append(f"{indent}{prefix}{body}")
            continue

        ordered = _ORDERED_RE.match(line)
        if ordered:
            indent = ordered.group(1)
            body = _render_inline(ordered.group(2))
            lines_out.append(f"{indent}· {body}")
            continue

        if line.strip().startswith("{") and line.strip().endswith("}"):
            pretty = _pretty_json_body(line)
            if pretty:
                for pline in pretty.splitlines():
                    lines_out.append(sty(pline, "hint") if color_enabled() else pline)
                continue

        lines_out.append(_render_inline(line))

    if in_fence and fence_buf:
        _flush_fence()

    return "\n".join(lines_out)


def render_for_terminal(
    text: str,
    *,
    force: bool = False,
    width: int | None = None,
    use_rich: bool | None = None,
    session: Any = None,
) -> str:
    """
    将 Markdown/JSON 转为终端 ANSI 文本（**仅终端展示**，勿写入 API 响应）。

    @param text Markdown 原文
    @param force 即使不像 Markdown 也尝试 JSON 美化
    @param width 终端宽度
    @param use_rich 是否用 Rich；None 时按 rich_render_enabled(session=session)
    @param session 追踪会话（/trace rich 开关）
    @return 带 ANSI 的终端字符串（非 Markdown 原文）
    """
    if not text.strip():
        return text
    if not force and not looks_like_markdown(text):
        return text

    prepared = _prettify_json_fences(text)
    rich_on = (
        use_rich
        if use_rich is not None
        else rich_render_enabled(session=session)
    )
    if rich_on:
        try:
            from io import StringIO

            buf = StringIO()
            _print_with_rich(prepared, file=buf, indent=0, width=width)
            captured = _strip_terminal_padding(buf.getvalue())
            if captured.strip():
                return captured
        except Exception:
            pass
    return _render_fallback(prepared)


def print_rendered(text: str, *, force: bool = False) -> None:
    """
    渲染并打印（优先 Rich 直出，否则 ANSI 字符串）。

    @param text 原文
    @param force 强制渲染
    """
    if not text.strip():
        return
    if print_terminal_formatted(text):
        return
    payload = render_for_terminal(text, force=force)
    if payload.strip():
        print(payload, flush=True)
