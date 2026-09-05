"""首 token 前奏的耗时回归：只测「不该做的事没做」，不测绝对墙钟。

三个被修掉的形态都是纯性能问题（结果一模一样，只是慢），所以断言分两层：
等价性（输出不变）+ 上界（不再随无空格长行二次方增长 / 空库不加载模型）。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.context_continuity import (
    _PATH_IN_TOOL_RE,
    _recent_read_paths,
    build_continuity_context_hint,
)

# 修复前的写法：起点不锚定分隔符，每个位置都贪婪吃到行尾再回溯
_QUADRATIC_PATH_RE = re.compile(
    r"[`'\"]?([^\s`'\"><|]+\.(?:md|mdc|txt|json|yaml|yml))[`'\"]?", re.I
)


def _read_tool_message(body: str, cid: str = "c1"):
    return ToolMessage(content=body, tool_call_id=cid, name="read_file")


def _nospace_blob(n: int = 4000) -> str:
    """无空格长行：压缩 JSON / base64 / CSV / lock 文件读出来就是这个形态。"""
    return ("0123456789abcdef" * (n // 16 + 1))[:n]


def test_path_regex_matches_same_paths_as_before() -> None:
    """锚定分隔符后 group(1) 必须与旧写法逐个相同（纯性能修复，不改行为）。"""
    samples = [
        "docs/cursor-agent.md",
        "见 `docs/项目结构.md` 与 docs/模块说明.md",
        "--- llgraph/core/agent.py (1-40) ---\n配置在 .llgraph/agent.json 里",
        "'a.md' \"b.json\" <c.yml> |d.yaml| e.txt",
        "no path here at all",
        "package-lock.json 和 uv.lock",
        _nospace_blob(500),
        "结尾就是路径 tail/x.mdc",
        "开头.md 也要能取到",
        Path(__file__).read_text(encoding="utf-8")[:3000],
    ]
    for text in samples:
        expected = [m.group(1) for m in _QUADRATIC_PATH_RE.finditer(text)]
        actual = [m.group(1) for m in _PATH_IN_TOOL_RE.finditer(text)]
        assert actual == expected, text[:80]


def test_path_regex_does_not_blow_up_on_nospace_lines() -> None:
    """旧写法在 4000 字符无空格正文上约 45ms；新写法必须落在 1ms 量级。"""
    blob = _nospace_blob(4000)

    slow = float("inf")
    fast = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        list(_QUADRATIC_PATH_RE.finditer(blob))
        slow = min(slow, time.perf_counter() - t0)
        t0 = time.perf_counter()
        list(_PATH_IN_TOOL_RE.finditer(blob))
        fast = min(fast, time.perf_counter() - t0)

    assert fast < 0.005, f"新写法 {fast * 1000:.2f}ms，疑似又退回逐位置回溯"
    assert fast * 20 < slow, f"新 {fast * 1000:.2f}ms vs 旧 {slow * 1000:.2f}ms，没拉开差距"


def test_recent_read_paths_scales_with_long_nospace_history() -> None:
    """整段前奏的实际形态：60 条 read_file 结果全是无空格长行。"""
    messages: list = []
    for i in range(60):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"id": f"c{i}", "name": "read_file", "args": {}}],
            )
        )
        messages.append(_read_tool_message(_nospace_blob(6000), f"c{i}"))

    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        paths = _recent_read_paths(messages)
        best = min(best, time.perf_counter() - t0)

    assert paths == []
    assert best < 0.2, f"_recent_read_paths {best * 1000:.0f}ms，60 条无空格正文又退化了"


def test_recent_read_paths_still_finds_paths() -> None:
    messages = [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "read_file", "args": {}}]),
        _read_tool_message("--- docs/项目结构.md ---\n目录说明见 docs/模块说明.md", "c1"),
    ]
    assert _recent_read_paths(messages) == ["docs/项目结构.md", "docs/模块说明.md"]


def test_continuity_hint_lists_recent_reads() -> None:
    messages = [
        HumanMessage(content="看下文档"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "read_file", "args": {}}]),
        _read_tool_message("正文见 docs/cursor-agent.md", "c1"),
        AIMessage(content="文档说明如上。"),
    ]
    hint = build_continuity_context_hint(messages, user_message="接着上面的分析继续")
    assert "docs/cursor-agent.md" in hint
