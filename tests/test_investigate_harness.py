"""ReAct 稳定性：字面量拦 parallel、工具往返上限收口。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.investigate_harness import (
    SOFT_CLOSE_NUDGE_MARKER,
    SOFT_CLOSE_TOOL_BLOCK_MSG,
    append_soft_close_for_dispatch,
    build_soft_close_nudge,
    count_tool_rounds_since_user,
    guard_soft_close_tools,
    last_real_user_text,
    parallel_blocked_for_literals_message,
    should_inject_soft_close,
    soft_close_nudge_pending,
    strip_ephemeral_harness_messages,
    suggest_repo_path_hint,
    user_has_directed_search_literals,
)


def test_literals_detect_project_and_camel() -> None:
    text = (
        "埋点属性删除后滚动条变短，确认后无法删除（dataworkstation项目），是前端bug吗"
    )
    assert user_has_directed_search_literals(text)
    msg = parallel_blocked_for_literals_message(text)
    assert "禁止先调用 search_code_parallel" in msg
    assert "grep_files" in msg


def test_no_literal_vague_question() -> None:
    assert not user_has_directed_search_literals("帮我看看哪里有问题")


def test_suggest_repo_path_hint() -> None:
    q = "埋点属性删除后确认无效（dataworkstation项目），是前端bug吗"
    assert "dataworkstation" in suggest_repo_path_hint(q).lower()


def test_count_tool_rounds_since_user() -> None:
    msgs = [
        HumanMessage(content="<user_query>\n是否前端 bug\n</user_query>"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "grep_files", "args": {}}]),
        ToolMessage(content="a", tool_call_id="1", name="grep_files"),
        AIMessage(content="", tool_calls=[{"id": "2", "name": "grep_files", "args": {}}]),
        ToolMessage(content="b", tool_call_id="2", name="grep_files"),
    ]
    assert count_tool_rounds_since_user(msgs) == 2


def test_soft_close_injects_at_threshold_for_any_query() -> None:
    """软收口不按意图分流：任意问句达轮次即注入。"""
    user = "<user_query>\n帮我改个函数名\n</user_query>"
    msgs: list = [HumanMessage(content=user)]
    for i in range(20):
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"id": f"c{i}", "name": "grep_files", "args": {"pattern": "x"}}],
            )
        )
        msgs.append(ToolMessage(content="hit", tool_call_id=f"c{i}", name="grep_files"))
    assert should_inject_soft_close(msgs, workspace=None) is True
    out = append_soft_close_for_dispatch(msgs, workspace=None)
    assert soft_close_nudge_pending(out)
    assert SOFT_CLOSE_NUDGE_MARKER in str(out[-1].content)
    assert "上限" in str(out[-1].content)
    assert should_inject_soft_close(out, workspace=None) is False
    nudge = build_soft_close_nudge(msgs, workspace=None)
    assert "禁止再开" in nudge
    assert "证据" in nudge


def test_soft_close_blocks_further_tools() -> None:
    user = "<user_query>\n随便查一下\n</user_query>"
    state = {
        "messages": [
            HumanMessage(content=user),
            HumanMessage(content=f"{SOFT_CLOSE_NUDGE_MARKER} 请结案"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "grep_files", "args": {"pattern": "x"}},
                    {"id": "c2", "name": "read_file", "args": {"path": "a.vue"}},
                ],
            ),
        ]
    }
    new_state, blocked = guard_soft_close_tools(state, workspace=None)
    assert len(blocked) == 2
    assert "上限" in blocked[0].content or SOFT_CLOSE_TOOL_BLOCK_MSG[:12] in blocked[0].content
    last = new_state["messages"][-1]
    assert isinstance(last, AIMessage)
    assert not (last.tool_calls or [])


def test_strip_ephemeral_and_last_user() -> None:
    msgs = [
        HumanMessage(content="<user_query>\nhello\n</user_query>"),
        HumanMessage(content=f"{SOFT_CLOSE_NUDGE_MARKER} x"),
        AIMessage(content="ok"),
    ]
    assert last_real_user_text(msgs) == "hello"
    stripped = strip_ephemeral_harness_messages(msgs)
    assert len(stripped) == 2


def test_workspace_context_no_intent_routing(tmp_path) -> None:
    from llgraph.context.context_builder import build_workspace_context_block
    from llgraph.context.context_session import ContextSession

    session = ContextSession()
    q = "优惠券里什么信息可能导致大key？"
    block = build_workspace_context_block(
        tmp_path,
        session,
        q,
        allow_write=False,
        recent_messages=[],
    )
    assert "意图与排查/归因策略由你自行判断" in block
    assert "归因核验" not in block
    assert "排查两段式" not in block
    assert "正确性优先" in block
