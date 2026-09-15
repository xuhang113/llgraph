"""Shell 命令轻量解析：切分简单命令、识别重定向、下钻命令替换。

权限判定不能只对整条命令做正则：`ls\\nrm -rf x`、`$(rm -rf x)`、`bash -c 'rm -rf x'`
都能绕过「行首或 ; & | 之后」这类边界，而 `2>&1`、`2>/dev/null`、`'a > b'`
又会被「见 > 就拦」误伤。这里把命令拆成 (词, 重定向) 结构后再判。

只做权限判定需要的语法：引号、转义、分隔符、重定向、`$(...)` / 反引号、heredoc 跳过。
不做变量展开、通配、别名。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 写入类重定向操作符（去掉前置 fd 后）
WRITE_REDIRECT_OPS = frozenset({">", ">>", ">|", "&>", "&>>", "<>"})

# 重定向到这些目标不算落盘
NULL_REDIRECT_TARGETS = frozenset({
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
})

_SEPARATORS = (";", "\n", "&&", "||", "|&", "|", "&")

_MAX_DEPTH = 6


@dataclass(frozen=True)
class Redirect:
    """一处重定向。"""

    op: str
    """去掉 fd 前缀后的操作符，如 ">"、">>"、">&"。"""

    fd: str = ""
    """操作符前的文件描述符字面量，如 `2>` 的 "2"。"""

    target: str = ""
    """目标词（已去引号）；`>&1` 这类为 "1"。"""

    def writes_file(self) -> bool:
        """
        是否把内容写进一个真实文件。

        @return 是否落盘写入
        """
        if self.op not in WRITE_REDIRECT_OPS:
            return False
        target = self.target.strip()
        if not target:
            return False
        if target in NULL_REDIRECT_TARGETS:
            return False
        # `>&2` 解析为 op=">&"，不会走到这里；`> &2` 这种写法按落盘处理
        return True


@dataclass(frozen=True)
class SimpleCommand:
    """一条简单命令：分隔符之间的词与重定向。"""

    words: tuple[str, ...] = ()
    """去引号后的词序列。"""

    quoted: tuple[bool, ...] = ()
    """与 words 等长：该词是否出现过引号（引号内的 > ; 等不再是操作符）。"""

    redirects: tuple[Redirect, ...] = ()
    """本条命令上的重定向。"""

    def head(self) -> str:
        """
        命令名（保留原样，不去路径）。

        @return 首个词；无词则空串
        """
        return self.words[0] if self.words else ""


@dataclass
class _Builder:
    commands: list[SimpleCommand] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    quoted: list[bool] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)
    buf: str | None = None
    buf_quoted: bool = False
    pending_op: str = ""
    pending_fd: str = ""
    heredocs: list[str] = field(default_factory=list)

    def add_char(self, ch: str, *, quoted: bool = False) -> None:
        self.buf = (self.buf or "") + ch
        if quoted:
            self.buf_quoted = True

    def touch_word(self) -> None:
        """标记「这里有一个词」（如 `$(...)` 展开后可能为空串）。"""
        if self.buf is None:
            self.buf = ""

    def flush_word(self) -> None:
        if self.buf is None:
            return
        text = self.buf
        was_quoted = self.buf_quoted
        self.buf = None
        self.buf_quoted = False
        if self.pending_op:
            if self.pending_op == "<<":
                self.heredocs.append(_heredoc_delimiter(text))
            self.redirects.append(
                Redirect(op=self.pending_op, fd=self.pending_fd, target=text)
            )
            self.pending_op = ""
            self.pending_fd = ""
            return
        self.words.append(text)
        self.quoted.append(was_quoted)

    def flush_command(self) -> None:
        self.flush_word()
        if self.pending_op:
            self.redirects.append(Redirect(op=self.pending_op, fd=self.pending_fd))
            self.pending_op = ""
            self.pending_fd = ""
        if self.words or self.redirects:
            self.commands.append(
                SimpleCommand(
                    words=tuple(self.words),
                    quoted=tuple(self.quoted),
                    redirects=tuple(self.redirects),
                )
            )
        self.words = []
        self.quoted = []
        self.redirects = []


def _match_separator(text: str, i: int) -> str:
    for sep in _SEPARATORS:
        if text.startswith(sep, i):
            return sep
    return ""


def _scan_single_quote(text: str, i: int) -> tuple[str, int]:
    """从 `'` 起扫到闭合引号；未闭合则吃到结尾。"""
    end = text.find("'", i + 1)
    if end < 0:
        return text[i + 1 :], len(text)
    return text[i + 1 : end], end + 1


def _scan_balanced(text: str, i: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """
    从 open_ch 起按引号感知地扫到配对的 close_ch。

    @param text 原文
    @param i open_ch 所在下标
    @param open_ch 开括号
    @param close_ch 闭括号
    @return (内部文本, 闭括号之后的下标)
    """
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "'":
            _, j = _scan_single_quote(text, j)
            continue
        if ch == '"':
            j = _skip_double_quote(text, j)
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
        j += 1
    return text[i + 1 :], len(text)


def _skip_double_quote(text: str, i: int) -> int:
    """跳过一段双引号，返回闭合引号之后的下标。"""
    j = i + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == '"':
            return j + 1
        j += 1
    return len(text)


def _scan_backtick(text: str, i: int) -> tuple[str, int]:
    """从反引号起扫到闭合反引号。"""
    j = i + 1
    out: list[str] = []
    while j < len(text):
        ch = text[j]
        if ch == "\\" and j + 1 < len(text):
            out.append(text[j + 1])
            j += 2
            continue
        if ch == "`":
            return "".join(out), j + 1
        out.append(ch)
        j += 1
    return "".join(out), len(text)


def _heredoc_delimiter(word: str) -> str:
    return word.strip().strip("'\"")


def _skip_heredoc_bodies(text: str, i: int, delimiters: list[str]) -> int:
    """
    跳过 heredoc 正文（正文是数据，不是命令，不该参与权限判定）。

    @param text 原文
    @param i 换行符之后的下标
    @param delimiters 本行登记的结束标记
    @return 跳过全部正文后的下标
    """
    pos = i
    for delim in delimiters:
        while pos < len(text):
            end = text.find("\n", pos)
            line = text[pos:end] if end >= 0 else text[pos:]
            consumed = len(line) + (1 if end >= 0 else 0)
            pos += consumed
            if line.strip() == delim or not delim:
                break
            if end < 0:
                break
    delimiters.clear()
    return pos


def _parse_redirect_op(text: str, i: int) -> tuple[str, int]:
    """识别位置 i 起的重定向操作符（不含 fd 前缀）。"""
    for op in (">>", ">|", ">&", "<<<", "<<", "<&", "<>", ">", "<"):
        if text.startswith(op, i):
            return op, i + len(op)
    return "", i


def parse_shell_commands(command: str, *, _depth: int = 0) -> tuple[SimpleCommand, ...]:
    """
    把一条 shell 命令拆成简单命令序列（含命令替换里的命令）。

    @param command 原始命令文本
    @param _depth 递归深度（内部用）
    @return 简单命令元组；顺序为出现顺序，替换内的命令排在宿主命令之前
    """
    if _depth > _MAX_DEPTH:
        return ()
    text = command or ""
    b = _Builder()
    nested: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == "\\":
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt == "\n":
                    i += 2
                    continue
                b.add_char(nxt, quoted=True)
                i += 2
                continue
            i += 1
            continue

        if ch == "'":
            inner, i = _scan_single_quote(text, i)
            b.add_char(inner, quoted=True)
            continue

        if ch == '"':
            i = _consume_double_quoted(text, i, b, nested)
            continue

        if ch == "`":
            inner, i = _scan_backtick(text, i)
            nested.append(inner)
            b.touch_word()
            continue

        if text.startswith("$((", i):
            inner, i = _scan_balanced(text, i + 1, "(", ")")
            b.add_char(f"$(({inner})", quoted=True)
            continue

        if text.startswith("$(", i):
            inner, i = _scan_balanced(text, i + 1, "(", ")")
            nested.append(inner)
            b.touch_word()
            continue

        if ch in " \t\r":
            b.flush_word()
            i += 1
            continue

        if ch == "\n":
            b.flush_command()
            i += 1
            if b.heredocs:
                i = _skip_heredoc_bodies(text, i, b.heredocs)
            continue

        if ch == "&" and text.startswith("&>", i):
            b.flush_word()
            op = "&>>" if text.startswith("&>>", i) else "&>"
            b.pending_op = op
            b.pending_fd = ""
            i += len(op)
            continue

        sep = _match_separator(text, i)
        if sep:
            b.flush_command()
            i += len(sep)
            continue

        if ch in "()":
            b.flush_command()
            i += 1
            continue

        if ch in "{}" and b.buf is None:
            nxt = text[i + 1 : i + 2]
            if nxt in ("", " ", "\t", "\n", ";"):
                b.flush_command()
                i += 1
                continue

        if ch in "<>":
            fd = ""
            if b.buf is not None and not b.buf_quoted and b.buf.isdigit():
                fd = b.buf
                b.buf = None
                b.buf_quoted = False
            else:
                b.flush_word()
            op, i = _parse_redirect_op(text, i)
            if not op:
                b.add_char(ch)
                i += 1
                continue
            b.pending_op = op
            b.pending_fd = fd
            continue

        b.add_char(ch)
        i += 1

    b.flush_command()

    out: list[SimpleCommand] = []
    for chunk in nested:
        out.extend(parse_shell_commands(chunk, _depth=_depth + 1))
    out.extend(b.commands)
    return tuple(out)


def _consume_double_quoted(
    text: str,
    i: int,
    b: _Builder,
    nested: list[str],
) -> int:
    """
    消费一段双引号：内部仍需下钻 `$(...)` 与反引号。

    @param text 原文
    @param i 起始双引号下标
    @param b 词构造器
    @param nested 命令替换收集器
    @return 闭合引号之后的下标
    """
    b.touch_word()
    b.buf_quoted = True
    j = i + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\" and j + 1 < len(text):
            b.add_char(text[j + 1], quoted=True)
            j += 2
            continue
        if ch == '"':
            return j + 1
        if text.startswith("$((", j):
            inner, j = _scan_balanced(text, j + 1, "(", ")")
            b.add_char(f"$(({inner})", quoted=True)
            continue
        if text.startswith("$(", j):
            inner, j = _scan_balanced(text, j + 1, "(", ")")
            nested.append(inner)
            continue
        if ch == "`":
            inner, j = _scan_backtick(text, j)
            nested.append(inner)
            continue
        b.add_char(ch, quoted=True)
        j += 1
    return len(text)
