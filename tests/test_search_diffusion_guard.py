"""扩散式检索治理：同 pattern 只换目录时扩根一次，后续子目录搜索走覆盖短路。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.react_limits import parse_grep_widen_after, resolve_grep_widen_after
from llgraph.core.search_diffusion_guard import (
    MERGED_MARKER,
    WIDENED_MARKER,
    GrepScope,
    PendingGrep,
    WidenDecision,
    apply_widened_note,
    common_root,
    plan_grep_widenings,
    widened_root_from_result,
)
from llgraph.core.tool_loop_guard import (
    IDENTICAL_BLOCK_MARKER,
    build_history_index,
    compute_tool_loop_plan,
    install_tool_loop_guard,
    is_llgraph_placeholder,
    tool_result_failed,
)
from llgraph.core.workspace import WorkspaceContext

PATTERN = "resolve_context_settings"


def _ai(calls: list[dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=calls)


def _call(cid: str, path: str, pattern: str = PATTERN, **args: object) -> dict:
    return {
        "id": cid,
        "name": "grep_files",
        "args": {"pattern": pattern, "path": path, **args},
        "type": "tool_call",
    }


def _grep_result(path: str, pattern: str = PATTERN) -> str:
    return f"匹配结果 2 处 / 1 个文件（ripgrep）：\n--- {path}/mod.py:12 ---\n>>> 12| {pattern}()"


def _pending(cid: str, path: str, pattern: str = PATTERN, file_glob: str = "") -> PendingGrep:
    return PendingGrep(call_id=cid, pattern=pattern, file_glob=file_glob, path=path)


def _prior(path: str, pattern: str = PATTERN, file_glob: str = "") -> GrepScope:
    return GrepScope(pattern=pattern, file_glob=file_glob, path=path)


# --- 决策层：什么时候才许扩根 ------------------------------------------------


def test_two_disjoint_dirs_then_third_is_widened() -> None:
    plan = plan_grep_widenings(
        [_prior("llgraph/core"), _prior("llgraph/context")],
        [_pending("c3", "llgraph/session")],
        widen_after=2,
    )
    assert plan.widen["c3"].root == "llgraph"
    assert set(plan.widen["c3"].sources) == {
        "llgraph/core",
        "llgraph/context",
        "llgraph/session",
    }


def test_below_threshold_is_not_widened() -> None:
    plan = plan_grep_widenings(
        [_prior("llgraph/core")],
        [_pending("c2", "llgraph/context")],
        widen_after=2,
    )
    assert plan.is_empty()


def test_zero_threshold_disables_layer() -> None:
    plan = plan_grep_widenings(
        [_prior("a/b"), _prior("a/c"), _prior("a/d")],
        [_pending("c9", "a/e")],
        widen_after=0,
    )
    assert plan.is_empty()


def test_already_covered_path_is_left_to_existing_dedupe() -> None:
    """历史已在 `.` 搜过：本层不接手，交给 tool_loop_guard 的覆盖短路。"""
    plan = plan_grep_widenings(
        [_prior("."), _prior("a/b")],
        [_pending("c3", "a/c")],
        widen_after=2,
    )
    assert plan.is_empty()


def test_pending_inside_a_searched_root_is_not_widened() -> None:
    """本次根已被某个旧根包住：该由覆盖短路拦下，不该再扩根多搜一遍。"""
    plan = plan_grep_widenings(
        [_prior("x/1"), _prior("x/2")],
        [_pending("c3", "x/1/deep")],
        widen_after=2,
    )
    assert plan.is_empty()


def test_model_widening_itself_is_not_rewritten() -> None:
    """模型自己已经改成搜父目录，不许再动它的参数。"""
    plan = plan_grep_widenings(
        [_prior("a/b"), _prior("a/c")],
        [_pending("c3", "a")],
        widen_after=2,
    )
    assert plan.is_empty()


def test_partial_cover_keeps_count_below_threshold() -> None:
    """新根盖住了一个旧根，剩下互不覆盖的只有 1 个 → 不到阈值。"""
    plan = plan_grep_widenings(
        [_prior("a/b/x"), _prior("c")],
        [_pending("c3", "a/b")],
        widen_after=2,
    )
    assert plan.is_empty()


def test_different_pattern_counts_separately() -> None:
    plan = plan_grep_widenings(
        [_prior("a/b", pattern="alpha"), _prior("a/c", pattern="beta")],
        [_pending("c3", "a/d", pattern="alpha")],
        widen_after=2,
    )
    assert plan.is_empty()


def test_different_file_glob_counts_separately() -> None:
    plan = plan_grep_widenings(
        [_prior("a/b", file_glob="*.py"), _prior("a/c")],
        [_pending("c3", "a/d", file_glob="*.py")],
        widen_after=2,
    )
    assert plan.is_empty()


def test_no_common_prefix_widens_to_workspace_root() -> None:
    plan = plan_grep_widenings(
        [_prior("llgraph/core"), _prior("docs")],
        [_pending("c3", "web-ui/src")],
        widen_after=2,
    )
    assert plan.widen["c3"].root == "."


def test_root_must_be_a_real_directory_when_workspace_given(tmp_path: Path) -> None:
    """模型传的相对路径可能被工具内部改写过，算出的祖先未必存在。"""
    (tmp_path / "real").mkdir()
    plan = plan_grep_widenings(
        [_prior("ghost/core"), _prior("ghost/context")],
        [_pending("c3", "ghost/session")],
        widen_after=2,
        workspace=tmp_path,
    )
    assert plan.is_empty()

    (tmp_path / "ghost").mkdir()
    plan_ok = plan_grep_widenings(
        [_prior("ghost/core"), _prior("ghost/context")],
        [_pending("c3", "ghost/session")],
        widen_after=2,
        workspace=tmp_path,
    )
    assert plan_ok.widen["c3"].root == "ghost"


def test_path_normalization_treats_dot_prefixed_as_same_root() -> None:
    plan = plan_grep_widenings(
        [_prior("./a/b"), _prior("a/b/")],
        [_pending("c3", "a/b")],
        widen_after=1,
    )
    assert plan.is_empty()


def test_batch_fanout_widens_once_and_merges_the_rest() -> None:
    """同一批 4 个只换目录的 grep：只执行一次扩根，其余合并到它。"""
    plan = plan_grep_widenings(
        [],
        [
            _pending("b1", "llgraph/core"),
            _pending("b2", "llgraph/context"),
            _pending("b3", "llgraph/session"),
            _pending("b4", "llgraph/display"),
        ],
        widen_after=2,
    )
    assert list(plan.widen) == ["b3"]
    assert plan.widen["b3"].root == "llgraph"
    assert plan.merged == {"b4": "b3"}


def test_common_root_of_files_is_their_directory() -> None:
    assert common_root(["pkg/a.py", "pkg/b.py"]) == "pkg"
    assert common_root(["a/b", "."]) == "."


# --- 结果尾注：既给模型看，也是后续覆盖判定的依据 ----------------------------


def test_widened_note_roundtrip() -> None:
    decision = WidenDecision(root="llgraph", pattern=PATTERN, sources=("llgraph/core",))
    body = apply_widened_note(_grep_result("llgraph/core"), decision)
    assert widened_root_from_result(body) == "llgraph"
    assert widened_root_from_result(_grep_result("llgraph/core")) == ""


def test_widened_note_is_appended_not_prefixed() -> None:
    """前置会被判成 llgraph 占位，整条结果就不进历史索引，覆盖短路也随之失效。"""
    decision = WidenDecision(root="llgraph", pattern=PATTERN, sources=("llgraph/core",))
    body = apply_widened_note(_grep_result("llgraph/core"), decision)
    assert body.startswith("匹配结果")
    assert is_llgraph_placeholder(body) is False
    assert tool_result_failed("grep_files", body) is False
    assert WIDENED_MARKER in body


def test_widened_note_is_not_appended_twice() -> None:
    decision = WidenDecision(root="llgraph", pattern=PATTERN, sources=("llgraph/core",))
    once = apply_widened_note(_grep_result("llgraph/core"), decision)
    assert apply_widened_note(once, decision) == once


# --- 与既有拦截层的接线 ------------------------------------------------------


def _history(paths: list[str]) -> list[BaseMessage]:
    msgs: list[BaseMessage] = [HumanMessage(content="哪些模块用了这个配置")]
    for idx, path in enumerate(paths, start=1):
        cid = f"h{idx}"
        msgs.append(_ai([_call(cid, path)]))
        msgs.append(
            ToolMessage(content=_grep_result(path), tool_call_id=cid, name="grep_files")
        )
    return msgs


def test_third_turn_grep_is_widened_and_fourth_is_blocked() -> None:
    msgs = _history(["llgraph/core", "llgraph/context"])
    third = [_call("t3", "llgraph/session")]
    plan = compute_tool_loop_plan([*msgs, _ai(third)], third, grep_widen_after=2)
    assert plan.blocked == {}
    assert plan.widenings["t3"].root == "llgraph"

    widened = ToolMessage(
        content=apply_widened_note(_grep_result("llgraph"), plan.widenings["t3"]),
        tool_call_id="t3",
        name="grep_files",
    )
    fourth = [_call("t4", "llgraph/display")]
    later = [*msgs, _ai(third), widened, _ai(fourth)]
    plan_4 = compute_tool_loop_plan(later, fourth, grep_widen_after=2)
    assert plan_4.widenings == {}
    assert IDENTICAL_BLOCK_MARKER in str(plan_4.blocked["t4"].content)


def test_history_index_records_the_root_actually_searched() -> None:
    decision = WidenDecision(root="llgraph", pattern=PATTERN, sources=("llgraph/core",))
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        _ai([_call("h1", "llgraph/core")]),
        ToolMessage(
            content=apply_widened_note(_grep_result("llgraph"), decision),
            tool_call_id="h1",
            name="grep_files",
        ),
    ]
    index = build_history_index(msgs)
    assert [rec.shape.path for rec in index.greps] == ["llgraph"]
    # fp 仍按模型请求的参数建，同参数重复请求照旧精确命中
    assert any(fp.key[2] == "llgraph/core" for fp in index.exact)


def test_failed_widened_result_grants_no_coverage() -> None:
    decision = WidenDecision(root="llgraph", pattern=PATTERN, sources=("llgraph/core",))
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        _ai([_call("h1", "llgraph/core")]),
        ToolMessage(
            content=apply_widened_note("grep_files 失败: 路径不存在", decision),
            tool_call_id="h1",
            name="grep_files",
        ),
        _ai([_call("h2", "llgraph/context")]),
    ]
    plan = compute_tool_loop_plan(msgs, [_call("h2", "llgraph/context")], grep_widen_after=2)
    assert plan.blocked == {}


def test_blocked_widening_call_drops_the_plan() -> None:
    """要扩根的那次调用自己被拦下（同参数已失败过）时，别留下指向空结果的合并占位。"""
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        _ai([_call("h1", "a/b")]),
        ToolMessage(content=_grep_result("a/b"), tool_call_id="h1", name="grep_files"),
        _ai([_call("h2", "a/c")]),
        ToolMessage(content=_grep_result("a/c"), tool_call_id="h2", name="grep_files"),
        _ai([_call("h3", "a/d")]),
        ToolMessage(
            content="grep_files 失败: 路径不存在: a/d", tool_call_id="h3", name="grep_files"
        ),
    ]
    calls = [_call("n1", "a/d"), _call("n2", "a/e")]
    plan = compute_tool_loop_plan([*msgs, _ai(calls)], calls, grep_widen_after=2)
    assert plan.widenings == {}
    assert "n1" in plan.blocked
    assert "n2" not in plan.blocked


def test_batch_merge_placeholder_is_not_indexed() -> None:
    calls = [
        _call("b1", "llgraph/core"),
        _call("b2", "llgraph/context"),
        _call("b3", "llgraph/session"),
        _call("b4", "llgraph/display"),
    ]
    msgs: list[BaseMessage] = [HumanMessage(content="q"), _ai(calls)]
    plan = compute_tool_loop_plan(msgs, calls, grep_widen_after=2)
    assert list(plan.widenings) == ["b3"]
    body = str(plan.blocked["b4"].content)
    assert body.startswith(MERGED_MARKER)
    assert is_llgraph_placeholder(body) is True


def test_identical_narrow_call_in_same_batch_still_blocked() -> None:
    """第三个调用被扩根后，同参数的第四个仍不执行（由扩根覆盖接住）。"""
    calls = [
        _call("b1", "llgraph/core"),
        _call("b2", "llgraph/context"),
        _call("b3", "llgraph/session"),
        _call("b4", "llgraph/session"),
    ]
    msgs: list[BaseMessage] = [HumanMessage(content="q"), _ai(calls)]
    plan = compute_tool_loop_plan(msgs, calls, grep_widen_after=2)
    assert list(plan.widenings) == ["b3"]
    body = str(plan.blocked["b4"].content)
    assert body.startswith(MERGED_MARKER)
    assert "tool_call_id=b3" in body


def test_layer_off_by_default_argument() -> None:
    msgs = _history(["a/b", "a/c"])
    calls = [_call("t3", "a/d")]
    plan = compute_tool_loop_plan([*msgs, _ai(calls)], calls)
    assert plan.widenings == {}


def test_identical_tool_guard_off_clears_widenings() -> None:
    class _Node:
        pass

    node = _Node()
    msgs = _history(["a/b", "a/c"])
    calls = [_call("t3", "a/d")]
    install_tool_loop_guard(node, [*msgs, _ai(calls)], calls, enabled=False, grep_widen_after=2)
    assert node._llgraph_grep_widenings == {}


def test_read_and_glob_are_untouched() -> None:
    calls = [
        {"id": "g1", "name": "glob_files", "args": {"glob_pattern": "*.py", "path": "a/b"}},
        {"id": "g2", "name": "glob_files", "args": {"glob_pattern": "*.py", "path": "a/c"}},
        {"id": "g3", "name": "glob_files", "args": {"glob_pattern": "*.py", "path": "a/d"}},
    ]
    msgs: list[BaseMessage] = [HumanMessage(content="q"), _ai(calls)]
    plan = compute_tool_loop_plan(msgs, calls, grep_widen_after=2)
    assert plan.widenings == {}


# --- 配置 -------------------------------------------------------------------


def test_parse_grep_widen_after() -> None:
    assert parse_grep_widen_after(None) == 2
    assert parse_grep_widen_after(False) == 0
    assert parse_grep_widen_after(0) == 0
    assert parse_grep_widen_after(3) == 3
    assert parse_grep_widen_after(99) == 10
    assert parse_grep_widen_after("bad") == 2


def test_resolve_grep_widen_after_from_workspace(tmp_path: Path) -> None:
    (tmp_path / ".llgraph").mkdir()
    (tmp_path / ".llgraph" / "agent.json").write_text(
        json.dumps({"agent": {"grep_widen_after": 0}}), encoding="utf-8"
    )
    assert resolve_grep_widen_after(tmp_path) == 0
    assert resolve_grep_widen_after(None) == 2


# --- 真 ToolNode 端到端 -----------------------------------------------------


def _node_config(thread_id: str) -> object:
    """真 ToolNode 需要 langgraph 注入的 runtime；内部键变动时退回普通 config。"""
    from langchain_core.runnables import ensure_config

    configurable: dict[str, object] = {"thread_id": thread_id}
    try:
        from langgraph._internal._constants import CONFIG_KEY_RUNTIME
        from langgraph.runtime import Runtime

        configurable[CONFIG_KEY_RUNTIME] = Runtime()
    except ImportError:  # pragma: no cover - langgraph 内部结构变动
        pass
    return ensure_config({"configurable": configurable})


def _seed_workspace(ws: Path) -> None:
    for pkg in ("core", "context", "session", "display"):
        target = ws / "pkg" / pkg
        target.mkdir(parents=True, exist_ok=True)
        (target / "mod.py").write_text(
            f"def use_{pkg}():\n    return {PATTERN}()\n",
            encoding="utf-8",
        )


def test_tool_node_end_to_end_widens_once_then_short_circuits(tmp_path: Path) -> None:
    """走生产链路：真 ToolNode + 真 grep_files。第三次执行的是公共祖先，第四次不再执行。"""
    import pytest

    from llgraph.core.react_tools import build_tool_node

    _seed_workspace(tmp_path)
    node = build_tool_node(create_filesystem_tools(WorkspaceContext(tmp_path)), workspace=tmp_path)
    thread = "t-e2e-grep-widen"
    msgs: list[BaseMessage] = [HumanMessage(content=f"哪些模块用了 {PATTERN}")]

    executed: list[str] = []
    for idx, sub in enumerate(("pkg/core", "pkg/context", "pkg/session", "pkg/display"), start=1):
        calls = [_call(f"e{idx}", sub)]
        msgs.append(_ai(calls))
        try:
            out = node.invoke({"messages": list(msgs)}, _node_config(thread))
        except ValueError as exc:  # pragma: no cover - langgraph config 契约变动
            pytest.skip(f"ToolNode 直接调用不可用: {exc}")
        produced = list(out.get("messages") or [])
        assert len(produced) == 1
        body = str(produced[0].content)
        msgs.extend(produced)
        executed.append(body)
        # 原 AIMessage 里模型请求的参数不许被改写
        assert msgs[-2].tool_calls[0]["args"]["path"] == sub

    # 前两次按模型请求的目录各搜一次，互相看不到对方
    assert "pkg/core/mod.py" in executed[0]
    assert "pkg/context/mod.py" not in executed[0]
    assert "pkg/context/mod.py" in executed[1]
    # 第三次：真正搜的是公共祖先 pkg，一次拿齐 4 个目录
    assert WIDENED_MARKER in executed[2]
    assert widened_root_from_result(executed[2]) == "pkg"
    for pkg in ("core", "context", "session", "display"):
        assert f"pkg/{pkg}/mod.py" in executed[2]
    # 第四次：已被扩根结果覆盖，不再执行真实 grep
    assert IDENTICAL_BLOCK_MARKER in executed[3]
    assert "pkg/display/mod.py" not in executed[3].split("上次返回摘录:")[0]
