"""Shell 头尾截断、cd 解析、超时保输出、后台 await。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from llgraph.config.shell_settings import ShellSettings
from llgraph.context.runtime_context import set_active_thread_id
from llgraph.core.shell_cwd import peel_all_leading_cd, peel_leading_cd
from llgraph.core.shell_jobs import reset_shell_runtime_for_tests
from llgraph.core.shell_output import clip_shell_output, combine_stdio
from llgraph.core.shell_tools import create_shell_tools
from llgraph.core.workspace import WorkspaceContext
from llgraph.display.trace_display import _short_tool_target
from llgraph.sandbox.exec import CappedByteBuffer
from llgraph.sandbox.runner import run_sandboxed_shell


@pytest.fixture(autouse=True)
def _reset_shell_runtime():
    reset_shell_runtime_for_tests()
    set_active_thread_id("test-shell")
    yield
    reset_shell_runtime_for_tests()
    set_active_thread_id(None)


def _settings(**overrides: object) -> ShellSettings:
    data: dict[str, object] = {
        "enabled": True,
        "timeout_sec": 5.0,
        "background_timeout_sec": 20.0,
        "max_output_chars": 4000,
        "max_jobs": 4,
        "terminal_log_dir": ".llgraph/context/terminals",
        "log_commands": False,
    }
    data.update(overrides)
    return ShellSettings(**data)  # type: ignore[arg-type]


def _tools(root: Path, **overrides: object):
    ctx = WorkspaceContext(root, allow_write=False)
    return create_shell_tools(ctx, allow_write=False, settings=_settings(**overrides))


def _py(script: str) -> str:
    return f"{sys.executable} -c {script!r}"


def test_clip_shell_output_keeps_head_and_tail() -> None:
    text = "HEAD\n" + ("x" * 4000) + "\nTAIL-FAIL\n"
    clipped, omitted = clip_shell_output(text, 200)
    assert omitted > 0
    assert clipped.startswith("HEAD")
    assert "TAIL-FAIL" in clipped
    assert "省略" in clipped
    assert "xxxx" not in clipped or clipped.count("x") < 4000


def test_clip_short_text_unchanged() -> None:
    assert clip_shell_output("ok\n", 1000) == ("ok\n", 0)


def test_combine_stdio_appends_stderr() -> None:
    assert combine_stdio("out", "err") == "out\nerr"


def test_peel_cd_chain() -> None:
    hops, rest = peel_all_leading_cd("cd src && cd pkg && pytest -q")
    assert hops == ["src", "pkg"]
    assert rest == "pytest -q"
    assert peel_leading_cd("echo cd foo")[0] is None
    assert peel_leading_cd("(cd foo && make)")[0] is None


def test_capped_buffer_drops_middle() -> None:
    buf = CappedByteBuffer(cap=100, head_keep=40)
    buf.write(b"A" * 40)
    buf.write(b"M" * 80)
    buf.write(b"Z" * 40)
    snap = buf.snapshot()
    assert snap.startswith(b"A" * 40)
    assert snap.endswith(b"Z" * 40)
    assert buf.total == 160


def test_timeout_keeps_partial_output(tmp_path: Path) -> None:
    tools = _tools(tmp_path, timeout_sec=0.4)
    run = next(t for t in tools if t.name == "run_shell_command")
    out = run.invoke(
        {
            "command": _py(
                "import sys,time; print('STARTED', flush=True); time.sleep(8); print('NEVER')"
            )
        }
    )
    assert "STARTED" in out
    assert "NEVER" not in out
    assert "超时" in out


def test_runner_timeout_keeps_stdout(tmp_path: Path) -> None:
    from llgraph.config.sandbox_settings import resolve_sandbox_settings
    from llgraph.sandbox.policy import build_sandbox_policy

    policy = build_sandbox_policy(
        tmp_path,
        resolve_sandbox_settings(tmp_path),
        cli_enabled=False,
        allow_write=False,
    )
    result = run_sandboxed_shell(
        policy,
        command=_py("import sys,time; print('PARTIAL', flush=True); time.sleep(8)"),
        cwd=tmp_path,
        timeout_sec=0.4,
    )
    assert result.error == "timeout"
    assert "PARTIAL" in result.stdout


def test_head_tail_on_tool_result(tmp_path: Path) -> None:
    tools = _tools(tmp_path, timeout_sec=5.0, max_output_chars=180)
    run = next(t for t in tools if t.name == "run_shell_command")
    out = run.invoke(
        {
            "command": _py(
                "print('BEGIN-MARKER'); print('m'*5000); print('END-FAIL: boom')"
            )
        }
    )
    assert "BEGIN-MARKER" in out
    assert "END-FAIL: boom" in out
    assert "省略" in out


def test_invalid_utf8_does_not_crash(tmp_path: Path) -> None:
    script = tmp_path / "emit_bad.py"
    script.write_text(
        "import sys\nsys.stdout.buffer.write(b'\\xff\\xfe hello\\n')\n",
        encoding="utf-8",
    )
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    out = run.invoke({"command": f"{sys.executable} {script}"})
    assert "hello" in out
    assert "执行失败" not in out


def test_cd_persists_across_calls(tmp_path: Path) -> None:
    sub = tmp_path / "pkg"
    sub.mkdir()
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    first = run.invoke({"command": "cd pkg"})
    assert "pkg" in first
    second = run.invoke({"command": "pwd"})
    assert str(sub.resolve()) in second or second.rstrip().endswith("pkg")


def test_cd_and_command_same_invocation(tmp_path: Path) -> None:
    sub = tmp_path / "inner"
    sub.mkdir()
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    out = run.invoke({"command": "cd inner && pwd"})
    assert "inner" in out


def test_background_and_await(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    await_tool = next(t for t in tools if t.name == "await_shell")
    started = run.invoke(
        {
            "command": _py(
                "import sys,time; print('GO', flush=True); time.sleep(0.6); print('DONE', flush=True)"
            ),
            "block_until_ms": 0,
        }
    )
    assert "running" in started
    assert "job=sh-" in started
    job_id = None
    for part in started.split():
        if part.startswith("job=sh-"):
            job_id = part.split("=", 1)[1].rstrip(",)")
            break
    assert job_id
    final = await_tool.invoke({"job_id": job_id, "block_until_ms": 5000})
    assert "DONE" in final
    assert "exit=0" in final or "[exit" not in final


def test_await_pattern_returns_early(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    await_tool = next(t for t in tools if t.name == "await_shell")
    started = run.invoke(
        {
            "command": _py(
                "import sys,time; print('READY-NOW', flush=True); time.sleep(8)"
            ),
            "block_until_ms": 0,
        }
    )
    job_id = next(
        part.split("=", 1)[1].rstrip(",)")
        for part in started.split()
        if part.startswith("job=sh-")
    )
    t0 = time.perf_counter()
    hit = await_tool.invoke(
        {"job_id": job_id, "block_until_ms": 5000, "pattern": "READY-NOW"}
    )
    elapsed = time.perf_counter() - t0
    assert "READY-NOW" in hit
    assert "仍在运行" in hit or "running" in hit
    assert elapsed < 3.0


def test_duplicate_running_command_reuses_job(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    cmd = _py("import time; time.sleep(8)")
    first = run.invoke({"command": cmd, "block_until_ms": 0})
    second = run.invoke({"command": cmd, "block_until_ms": 0})
    assert "未再启动新进程" in second
    assert "job=sh-" in first
    job1 = next(p.split("=", 1)[1].rstrip(",)") for p in first.split() if p.startswith("job=sh-"))
    job2 = next(p.split("=", 1)[1].rstrip(",)") for p in second.split() if p.startswith("job=sh-"))
    assert job1 == job2


def test_schema_includes_block_until_ms_and_await_shell(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    await_tool = next(t for t in tools if t.name == "await_shell")
    run_props = convert_to_openai_tool(run)["function"]["parameters"]["properties"]
    await_props = convert_to_openai_tool(await_tool)["function"]["parameters"]["properties"]
    assert "block_until_ms" in run_props
    assert "working_directory" in run_props
    assert "job_id" in await_props
    assert "pattern" in await_props


def test_trace_summary_prefers_job_id() -> None:
    assert _short_tool_target("await_shell", {"job_id": "sh-abc123"}) == "sh-abc123"
    assert _short_tool_target(
        "await_shell", {"job_id": "sh-abc123", "pattern": "READY"}
    ) == "sh-abc123"


def test_cancel_kills_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llgraph.core.react_invoke.agent_cancel_requested",
        lambda: True,
    )
    tools = _tools(tmp_path, timeout_sec=8.0)
    run = next(t for t in tools if t.name == "run_shell_command")
    t0 = time.perf_counter()
    out = run.invoke(
        {"command": _py("import time; time.sleep(6); print('TOO-LATE')")}
    )
    assert time.perf_counter() - t0 < 4.0
    assert "停止" in out or "cancelled" in out
    assert "TOO-LATE" not in out


def test_resolve_shell_settings_defaults(tmp_path: Path) -> None:
    from llgraph.config.shell_settings import resolve_shell_settings

    settings = resolve_shell_settings(tmp_path)
    assert settings.max_output_chars == 32_000
    assert settings.background_timeout_sec == 1800.0
    assert settings.max_jobs == 4


def test_await_unknown_job(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    await_tool = next(t for t in tools if t.name == "await_shell")
    out = await_tool.invoke({"job_id": "sh-missing"})
    assert "找不到 job" in out


def test_explicit_dot_resets_cwd(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    tools = _tools(tmp_path)
    run = next(t for t in tools if t.name == "run_shell_command")
    run.invoke({"command": "cd pkg"})
    out = run.invoke({"command": "pwd", "working_directory": "."})
    assert str(tmp_path.resolve()) in out
    assert not out.rstrip().endswith("pkg")
