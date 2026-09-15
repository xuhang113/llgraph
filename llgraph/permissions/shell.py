"""Shell 命令权限：按简单命令逐条判定，只读黑名单 + 全局高危拦截。

判定基于 `shell_lex` 的结构化解析，而不是整条命令的正则：

- 分隔符不再只认 `; & |`：换行、`(...)`、`$(...)`、反引号里的命令一样要过闸。
- 只读模式只拦「写入类重定向到真实文件」，`2>&1`、`2>/dev/null`、`< in`、
  以及引号里的 `>` 都放行。
- `sudo` / `env` / `xargs` / `timeout` / `bash -c` 这类包装会剥到真正的命令名。
"""

from __future__ import annotations

import re

from llgraph.permissions.shell_lex import Redirect, SimpleCommand, parse_shell_commands

# 只读模式禁止的命令名（写盘 / 改权限 / 提权）
_READ_ONLY_COMMANDS = frozenset({
    "rm",
    "rmdir",
    "mv",
    "cp",
    "ln",
    "chmod",
    "chown",
    "chgrp",
    "tee",
    "dd",
    "truncate",
    "shred",
    "mkdir",
    "touch",
    "install",
    "patch",
    "mkfs",
})

# 只读模式禁止的 `<cmd> <subcmd>` 组合
_READ_ONLY_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "git": frozenset({
        "commit",
        "push",
        "pull",
        "reset",
        "checkout",
        "switch",
        "restore",
        "merge",
        "rebase",
        "cherry-pick",
        "revert",
        "clean",
        "apply",
        "am",
        "rm",
        "mv",
    }),
    "npm": frozenset({"install", "uninstall", "publish", "ci"}),
    "pnpm": frozenset({"install", "uninstall", "add", "remove", "publish"}),
    "yarn": frozenset({"install", "add", "remove", "publish"}),
    "bun": frozenset({"install", "add", "remove"}),
    "pip": frozenset({"install", "uninstall"}),
    "pip3": frozenset({"install", "uninstall"}),
    "uv": frozenset({"add", "remove", "sync", "pip"}),
    "mvn": frozenset({"install", "deploy"}),
    "cargo": frozenset({"install", "publish"}),
    "go": frozenset({"install"}),
    "apt": frozenset({"install", "remove", "purge"}),
    "apt-get": frozenset({"install", "remove", "purge"}),
    "yum": frozenset({"install", "remove"}),
    "dnf": frozenset({"install", "remove"}),
    "brew": frozenset({"install", "uninstall"}),
}

# `git stash push/drop/clear` 才算写；`git stash list` 放行
_READ_ONLY_SUBSUB: dict[tuple[str, str], frozenset[str]] = {
    ("git", "stash"): frozenset({"push", "drop", "clear", "pop", "apply", "save"}),
    ("uv", "pip"): frozenset({"install", "uninstall", "sync"}),
}

# 剥壳命令：真正要判的是后面那条
_WRAPPERS = frozenset({
    "sudo",
    "doas",
    "env",
    "nohup",
    "nice",
    "ionice",
    "stdbuf",
    "setsid",
    "time",
    "timeout",
    "command",
    "builtin",
    "exec",
    "xargs",
})

# `-c` 后面整串是子命令，需递归判定
_SHELL_RUNNERS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "ash"})

# 就地改文件的解释器开关（`sed -i`、`perl -pi -e`）
_IN_PLACE_TOOLS = frozenset({"sed", "perl", "ruby", "gsed"})

# 高危删除的目标根（含通配写法）
_DANGEROUS_ROOTS = frozenset({
    "/",
    "/*",
    "/.",
    "~",
    "~/",
    "~/*",
    "$HOME",
    "$HOME/",
    "$HOME/*",
    "/usr",
    "/etc",
    "/var",
    "/bin",
    "/sbin",
    "/lib",
    "/boot",
    "/dev",
    "/opt",
    "/home",
    "/root",
    "/System",
    "/Users",
    "/Library",
})

_FORK_BOMB_RE = re.compile(r":\s*\(\s*\)\s*\{.*\|\s*:\s*&.*\}\s*;?\s*:", re.DOTALL)

_BLOCK_DEVICE_RE = re.compile(r"^/dev/(?:sd|hd|nvme|vd|disk|rdisk|mmcblk|loop)", re.IGNORECASE)

_READ_ONLY_HINT = (
    "请使用 llgraph -w 启动，或改用 read_file/grep_files 等只读工具。"
)


def _basename(word: str) -> str:
    """
    取命令名（去掉路径与 Windows 反斜杠）。

    @param word 原始词
    @return 命令名
    """
    text = word.strip()
    if not text:
        return ""
    for sep in ("/", "\\"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text


def _is_assignment(word: str) -> bool:
    head, sep, _ = word.partition("=")
    if not sep or not head:
        return False
    return head.replace("_", "a").isalnum() and not head[0].isdigit()


def _peel(words: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """
    剥掉环境赋值与包装命令，得到真正执行的命令。

    @param words 简单命令的词序列
    @return (命令名, 该命令的参数)
    """
    rest = list(words)
    seen_wrapper = 0
    while rest:
        while rest and _is_assignment(rest[0]):
            rest.pop(0)
        if not rest:
            return "", ()
        name = _basename(rest[0])
        if name not in _WRAPPERS or seen_wrapper > 4:
            return name, tuple(rest[1:])
        seen_wrapper += 1
        rest.pop(0)
        while rest and rest[0].startswith("-"):
            rest.pop(0)
        if name == "timeout":
            while rest and re.fullmatch(r"[0-9.]+[smhd]?", rest[0]):
                rest.pop(0)
    return "", ()


def _short_flags(args: tuple[str, ...]) -> str:
    """
    合并所有短参数字母（`-rf`、`-r -f` 都算）。

    @param args 参数
    @return 字母串
    """
    letters: list[str] = []
    for arg in args:
        if arg.startswith("--") or not arg.startswith("-"):
            continue
        letters.append(arg[1:])
    return "".join(letters)


def _positional(args: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(a for a in args if not a.startswith("-"))


def _normalize_target(target: str) -> str:
    text = target.strip().rstrip("/")
    return text or "/"


def _is_dangerous_root(target: str) -> bool:
    raw = target.strip()
    if not raw:
        return False
    if raw in _DANGEROUS_ROOTS:
        return True
    return _normalize_target(raw) in {_normalize_target(x) for x in _DANGEROUS_ROOTS}


def _check_always_blocked(cmd: SimpleCommand) -> str | None:
    """
    任意模式（含 -w）都要拦的高危命令。

    @param cmd 简单命令
    @return 拒绝原因；允许则 None
    """
    name, args = _peel(cmd.words)
    if not name:
        return None

    if name in ("rm", "rmdir"):
        flags = _short_flags(args)
        recursive = "r" in flags.lower() or any(
            a in ("--recursive", "--dir") for a in args
        )
        no_preserve = any(a == "--no-preserve-root" for a in args)
        for target in _positional(args):
            if _is_dangerous_root(target) and (recursive or no_preserve or name == "rmdir"):
                return (
                    f"该命令被安全策略禁止（高危删除）：{name} 递归删除系统根目录 {target}"
                )

    if name.startswith("mkfs"):
        return f"该命令被安全策略禁止（格式化磁盘）：{name}"

    if name in ("dd", "shred"):
        for arg in args:
            value = arg.split("=", 1)[1] if arg.startswith("of=") else arg
            if _BLOCK_DEVICE_RE.match(value.strip()):
                return f"该命令被安全策略禁止（写块设备）：{name} → {value}"

    for redirect in cmd.redirects:
        if redirect.writes_file() and _BLOCK_DEVICE_RE.match(redirect.target.strip()):
            return f"该命令被安全策略禁止（写块设备）：重定向 → {redirect.target}"

    return None


def _write_redirect_reason(redirects: tuple[Redirect, ...]) -> str | None:
    for redirect in redirects:
        if redirect.writes_file():
            return (
                f"只读模式禁止写入重定向（{redirect.fd}{redirect.op} {redirect.target}）。"
                f"2>&1、2>/dev/null、管道可用；要落盘请使用 llgraph -w。"
            )
    return None


def _find_bypass_reason(name: str, args: tuple[str, ...]) -> str | None:
    """
    `find -delete` / `-exec rm` 这类经 find 落盘的写操作。

    @param name 命令名
    @param args 参数
    @return 拒绝原因；允许则 None
    """
    if name != "find":
        return None
    for idx, arg in enumerate(args):
        if arg == "-delete":
            return f"只读模式禁止 find -delete。{_READ_ONLY_HINT}"
        if arg in ("-exec", "-execdir", "-ok", "-okdir"):
            inner = args[idx + 1 : idx + 2]
            inner_name = _basename(inner[0]) if inner else ""
            if inner_name in _READ_ONLY_COMMANDS:
                return (
                    f"只读模式禁止 find {arg} {inner_name}（写盘）。{_READ_ONLY_HINT}"
                )
    return None


def _in_place_reason(name: str, args: tuple[str, ...]) -> str | None:
    """`sed -i` / `perl -i` 就地改文件。"""
    if name not in _IN_PLACE_TOOLS:
        return None
    for arg in args:
        if arg == "--in-place" or arg.startswith("--in-place="):
            return f"只读模式禁止 {name} 就地改写（--in-place）。{_READ_ONLY_HINT}"
        if arg.startswith("-") and not arg.startswith("--") and "i" in arg[1:]:
            return f"只读模式禁止 {name} 就地改写（-i）。{_READ_ONLY_HINT}"
    return None


def _subcommand_reason(name: str, args: tuple[str, ...]) -> str | None:
    blocked = _READ_ONLY_SUBCOMMANDS.get(name)
    if not blocked:
        return None
    positional = _positional(args)
    if not positional:
        return None
    sub = positional[0]
    deeper = _READ_ONLY_SUBSUB.get((name, sub))
    if deeper is not None:
        if len(positional) > 1 and positional[1] in deeper:
            return (
                f"只读模式禁止该命令（写盘/git 提交/安装依赖等）：{name} {sub} {positional[1]}。"
                f"{_READ_ONLY_HINT}"
            )
        return None
    if sub in blocked:
        return (
            f"只读模式禁止该命令（写盘/git 提交/安装依赖等）：{name} {sub}。"
            f"{_READ_ONLY_HINT}"
        )
    return None


def _escalation_name(words: tuple[str, ...]) -> str:
    """
    命令是否经 sudo / doas 提权（提权本身在只读模式即拦，不看后面是什么）。

    @param words 词序列
    @return 提权命令名；未提权则空串
    """
    for word in words:
        if _is_assignment(word):
            continue
        name = _basename(word)
        if name in ("sudo", "doas"):
            return name
        if name in _WRAPPERS or word.startswith("-"):
            continue
        return ""
    return ""


def _check_read_only(cmd: SimpleCommand) -> str | None:
    """
    只读模式下的拦截。

    @param cmd 简单命令
    @return 拒绝原因；允许则 None
    """
    reason = _write_redirect_reason(cmd.redirects)
    if reason:
        return reason

    escalation = _escalation_name(cmd.words)
    if escalation:
        return (
            f"只读模式禁止提权命令：{escalation}。{_READ_ONLY_HINT}"
        )

    name, args = _peel(cmd.words)
    if not name:
        return None

    if name in _READ_ONLY_COMMANDS:
        return (
            f"只读模式禁止该命令（写盘/git 提交/安装依赖等）：{name}。{_READ_ONLY_HINT}"
        )
    for probe in (
        _subcommand_reason(name, args),
        _find_bypass_reason(name, args),
        _in_place_reason(name, args),
    ):
        if probe:
            return probe
    return None


def _expand_shell_runners(commands: tuple[SimpleCommand, ...]) -> tuple[SimpleCommand, ...]:
    """
    把 `bash -c '<cmd>'` 的内层命令展开进待判列表。

    @param commands 解析出的简单命令
    @return 含内层命令的列表
    """
    out: list[SimpleCommand] = []
    for cmd in commands:
        out.append(cmd)
        name, args = _peel(cmd.words)
        if name not in _SHELL_RUNNERS:
            continue
        for idx, arg in enumerate(args):
            if arg in ("-c", "-lc", "-cl") or (arg.startswith("-") and arg.endswith("c")):
                inner = args[idx + 1 : idx + 2]
                if inner:
                    out.extend(parse_shell_commands(inner[0]))
                break
    return tuple(out)


def check_shell_command(command: str, *, allow_write: bool) -> str | None:
    """
    校验 shell 命令是否允许执行。

    @param command shell 命令
    @param allow_write 是否 -w 模式
    @return 拒绝原因；允许则返回 None
    """
    stripped = (command or "").strip()
    if not stripped:
        return "命令不能为空"

    if _FORK_BOMB_RE.search(stripped):
        return "该命令被安全策略禁止（fork 炸弹）"

    commands = _expand_shell_runners(parse_shell_commands(stripped))
    for cmd in commands:
        reason = _check_always_blocked(cmd)
        if reason:
            return reason
    if allow_write:
        return None
    for cmd in commands:
        reason = _check_read_only(cmd)
        if reason:
            return reason
    return None
