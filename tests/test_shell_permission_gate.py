"""Shell 权限闸门：只读放行常见只读命令、拦住绕过写法；高危写法任意模式都拦。"""

from __future__ import annotations

import pytest

from llgraph.permissions.shell import check_shell_command
from llgraph.permissions.shell_lex import parse_shell_commands

READ_ONLY_ALLOWED = [
    "ls -la",
    "grep -rn foo . 2>/dev/null",
    "python3 -m pytest -q 2>&1 | tail -20",
    "cat < README.md",
    "echo 'a > b'",
    "git log --format='%h -> %s' | head",
    "git status && git diff",
    "git stash list",
    "uv pip list",
    "uv run pytest -q",
    "npm ls --depth=0",
    "go test ./... 2>&1 | tail -30",
    "echo $((1 > 2))",
    "diff <(sort a) <(sort b)",
    "for f in *.py; do echo $f; done",
    "if [ -f x ]; then cat x; fi",
    "find . -name '*.py' | head",
    "sed -n '1,20p' README.md",
    "cd web-ui && npx tsc --noEmit",
]

READ_ONLY_BLOCKED = [
    "ls\nrm -rf build",
    "echo $(rm -rf build)",
    "echo `rm -rf build`",
    "true && rm -rf build",
    "xargs rm -rf < list.txt",
    "find . -name x -exec rm {} \\;",
    "find . -name x -delete",
    "bash -c 'rm -rf build'",
    "echo hi > out.txt",
    "echo hi >> out.txt",
    "cat a | tee b",
    "sed -i 's/a/b/' f.py",
    "git commit -m x",
    "git stash push",
    "pip install requests",
    "uv pip install requests",
    "sudo ls",
    "mkdir -p tmp",
    "( cd x && rm -rf y )",
    "dd if=a of=b",
]

ALWAYS_BLOCKED = [
    "rm -rf /",
    "rm -r -f /",
    "rm --recursive --force /",
    "rm -rf /*",
    "rm -rf ~",
    "sudo rm -rf /",
    "ls; rm -rf ~/",
    "dd if=/dev/zero of=/dev/sda",
    "cat x > /dev/sda",
    "mkfs.ext4 /dev/sdb1",
    ":(){ :|:& };:",
]

WRITE_MODE_ALLOWED = [
    "rm -rf build",
    "rm -rf /tmp/llgraph-test",
    "git commit -m x",
    "pip install -e .",
    "mkdir -p a/b",
    "echo x > out.txt",
    "sed -i 's/a/b/' f.py",
    "chmod +x scripts/x.sh",
]


@pytest.mark.parametrize("command", READ_ONLY_ALLOWED)
def test_read_only_allows_common_inspection(command: str) -> None:
    """只读模式不该误伤 2>&1 / 2>/dev/null / 引号里的 > / 输入重定向。"""
    assert check_shell_command(command, allow_write=False) is None


@pytest.mark.parametrize("command", READ_ONLY_BLOCKED)
def test_read_only_blocks_writes_and_bypasses(command: str) -> None:
    """换行、命令替换、bash -c、xargs、find -exec 都不能绕过只读闸门。"""
    assert check_shell_command(command, allow_write=False)


@pytest.mark.parametrize("command", ALWAYS_BLOCKED)
def test_always_blocked_even_in_write_mode(command: str) -> None:
    assert check_shell_command(command, allow_write=True)


@pytest.mark.parametrize("command", WRITE_MODE_ALLOWED)
def test_write_mode_allows_scoped_writes(command: str) -> None:
    assert check_shell_command(command, allow_write=True) is None


def test_empty_command_rejected() -> None:
    assert check_shell_command("   ", allow_write=True) == "命令不能为空"


def test_heredoc_body_is_data_not_command() -> None:
    """heredoc 正文里的 rm 是数据，不参与判定。"""
    command = "python3 - <<'PY'\nimport shutil\n# rm -rf /etc\nPY\necho done"
    assert check_shell_command(command, allow_write=False) is None


def test_heredoc_redirect_to_file_still_blocked() -> None:
    command = "cat <<EOF > f.txt\nhello\nEOF"
    assert check_shell_command(command, allow_write=False)


def test_parse_splits_newline_and_substitution() -> None:
    cmds = parse_shell_commands("ls -l\necho $(git rev-parse HEAD)")
    heads = [cmd.head() for cmd in cmds]
    assert "git" in heads
    assert heads[-2:] == ["ls", "echo"]


def test_parse_keeps_fd_prefix_and_dup_target() -> None:
    (cmd,) = parse_shell_commands("pytest 2>&1")
    (redirect,) = cmd.redirects
    assert (redirect.fd, redirect.op, redirect.target) == ("2", ">&", "1")
    assert not redirect.writes_file()


def test_parse_dev_null_is_not_file_write() -> None:
    (cmd,) = parse_shell_commands("make 2>/dev/null")
    assert not cmd.redirects[0].writes_file()


def test_parse_quoted_redirect_is_plain_text() -> None:
    (cmd,) = parse_shell_commands("echo 'a > b'")
    assert cmd.redirects == ()
    assert cmd.words == ("echo", "a > b")
