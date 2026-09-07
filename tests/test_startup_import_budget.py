"""冷启动 import 预算：出提示符之前不许拉 LLM SDK。

llgraph 启动 90% 的墙钟是 import：`langgraph` + `langchain_anthropic` +
`anthropic` 一条链约 0.8s。这些只有真正建 Agent 时才需要，
所以「打印 --help」「列会话」「跑 llgraph index」「渲染 banner」这些路径
都必须在不 import 它们的前提下完成。

断言写成「sys.modules 里没有这几个名字」而不是墙钟上界：
墙钟随机器抖，import 图不抖，回归时报错也直接指到是谁把链拉回来的。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# 建 Agent 才需要的重模块；任何一个出现在启动早期路径里都是回归
HEAVY_MODULES = ("anthropic", "langchain_anthropic", "langgraph")

_PROBE = """
import importlib, json, sys

names = json.loads(sys.argv[1])
for name in names:
    importlib.import_module(name)
heavy = json.loads(sys.argv[2])
print(json.dumps(sorted(m for m in heavy if m in sys.modules)))
"""

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _heavy_after_import(*modules: str) -> list[str]:
    """在干净子解释器里 import 指定模块，返回被顺带拉进来的重模块。"""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(list(modules)), json.dumps(list(HEAVY_MODULES))],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "module",
    [
        # CLI 入口：--help / --list-sessions / index / search / web 都只走到这里
        "llgraph.main",
        # 子命令入口
        "llgraph.cli.index_cli",
        "llgraph.cli.search_cli",
        "llgraph.cli.web_cli",
        # 会话列举与删除
        "llgraph.session.session_registry",
        "llgraph.session.session_delete",
        # banner 与过程展示：交互模式提示符之前会走
        "llgraph.terminal.session",
        "llgraph.terminal.banner",
        "llgraph.display.trace_display",
        # 启动阶段被调用，但都不该拉起 Agent 那条链
        "llgraph.core.mcp_bundle",
        "llgraph.core.checkpointer_factory",
        "llgraph.session.session_web_search",
    ],
)
def test_startup_module_does_not_import_llm_stack(module: str) -> None:
    assert _heavy_after_import(module) == []


def test_agent_module_still_imports_llm_stack() -> None:
    """反向断言：真正建 Agent 的模块当然要拉链，否则上面的用例是假绿。"""
    assert _heavy_after_import("llgraph.core.agent") == sorted(HEAVY_MODULES)


def test_trace_mode_is_shared_not_duplicated() -> None:
    """trace_display 必须 re-export trace_mode 的同一个 Enum，不能各自定义一份。"""
    from llgraph.display import trace_display, trace_mode

    assert trace_display.TraceMode is trace_mode.TraceMode
    assert trace_display.parse_trace_mode is trace_mode.parse_trace_mode
    assert trace_display.TRACE_MODE_LABELS is trace_mode.TRACE_MODE_LABELS
    assert trace_display.parse_trace_mode("完整") is trace_mode.TraceMode.ALL


def test_mcp_bundle_reexported_from_core_tools() -> None:
    """老 import 路径 llgraph.core.tools.load_mcp_tool_bundle 不能断。"""
    from llgraph.core import mcp_bundle, tools

    assert tools.load_mcp_tool_bundle is mcp_bundle.load_mcp_tool_bundle


def test_every_llgraph_subpackage_has_init() -> None:
    """缺 __init__.py 的目录会被 setuptools 的 packages.find 整个丢掉（非 editable 安装即缺文件）。

    `llgraph/prompts` 是例外：它按 package-data 打进 `llgraph`，本身不是包。
    """
    pkg_root = _REPO_ROOT / "llgraph"
    missing = sorted(
        str(d.relative_to(_REPO_ROOT))
        for d in pkg_root.rglob("*")
        if d.is_dir()
        and d.name != "__pycache__"
        and "prompts" not in d.relative_to(pkg_root).parts
        and not (d / "__init__.py").exists()
    )
    assert missing == []


def test_mcp_bundle_skips_tool_stack_without_servers(tmp_path: Path) -> None:
    """没配 MCP Server 时不该 import mcp_tools（那条链要 langchain_core.tools）。"""
    from llgraph.core.mcp_bundle import load_mcp_tool_bundle

    tools, registry, summary = load_mcp_tool_bundle(tmp_path, allow_write=False)
    assert tools == []
    assert registry is None
    assert "MCP" in summary
