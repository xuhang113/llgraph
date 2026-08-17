"""会话锚点：检测、兜底摘要、自动注入上下文。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llgraph.context.conversation_anchor import (
    CONVERSATION_ANCHOR_TAG,
    ensure_anchor_sections_minimum,
    ensure_messages_include_conversation_anchor,
    is_conversation_anchor_message,
    load_anchor_sections,
    save_anchor_sections,
    _fallback_decisions_from_assistant,
)
from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch
from llgraph.context.runtime_context import set_active_thread_id
from llgraph.session.session_manifest import (
    SESSION_MANIFEST_TAG,
    build_session_manifest_system_message,
    conversation_anchor_json_path,
)
from llgraph.context.context_session import ContextSession


def test_manifest_is_not_anchor_message() -> None:
    manifest = build_session_manifest_system_message(
        Path("/tmp/ws"),
        "t1",
        ContextSession(),
        "hi",
        anchor_path="conversation_anchor.json",
    )
    assert is_conversation_anchor_message(manifest) is False


def test_real_anchor_message_detected() -> None:
    msg = HumanMessage(content=f"{CONVERSATION_ANCHOR_TAG}\n## 会话目标\n做台账\n</conversation-anchor>")
    assert is_conversation_anchor_message(msg) is True


def test_ensure_anchor_sections_fills_session_goal(tmp_path: Path) -> None:
    sections = ensure_anchor_sections_minimum(
        {},
        workspace=tmp_path,
        thread_id="t1",
        span_messages=[HumanMessage(content="设计部门大数据成本分摊接口")],
    )
    assert "大数据" in sections["session_goal"]


def test_inject_anchor_into_messages(tmp_path: Path) -> None:
    thread_id = "cli-test"
    save_anchor_sections(
        tmp_path,
        thread_id,
        ensure_anchor_sections_minimum(
            {},
            workspace=tmp_path,
            thread_id=thread_id,
            span_messages=[HumanMessage(content="实现数据台账 API")],
        ),
    )
    manifest = build_session_manifest_system_message(
        tmp_path,
        thread_id,
        ContextSession(),
        "",
        anchor_path="conversation_anchor.json",
    )
    messages = [manifest, HumanMessage(content="继续")]
    injected = ensure_messages_include_conversation_anchor(tmp_path, thread_id, messages)
    anchors = [m for m in injected if is_conversation_anchor_message(m)]
    assert len(anchors) == 1
    assert "数据台账" in str(anchors[0].content)


def test_dispatch_includes_anchor_before_user_not_in_system(tmp_path: Path) -> None:
    thread_id = "cli-dispatch"
    save_anchor_sections(
        tmp_path,
        thread_id,
        ensure_anchor_sections_minimum(
            {},
            workspace=tmp_path,
            thread_id=thread_id,
            span_messages=[HumanMessage(content="新增 DeptBigdataCost 接口")],
        ),
    )
    manifest = build_session_manifest_system_message(
        tmp_path,
        thread_id,
        ContextSession(),
        "",
        anchor_path="conversation_anchor.json",
    )
    set_active_thread_id(thread_id)
    try:
        prepared = prepare_messages_for_llm_dispatch(
            [HumanMessage(content="继续"), manifest],
            agent_system_content="你是测试 Agent。",
            workspace=tmp_path,
        )
    finally:
        set_active_thread_id(None)
    system = prepared[0]
    assert isinstance(system, SystemMessage)
    system_text = system.content if isinstance(system.content, str) else str(system.content)
    assert CONVERSATION_ANCHOR_TAG not in system_text
    assert SESSION_MANIFEST_TAG not in system_text
    anchor_msgs = [m for m in prepared if is_conversation_anchor_message(m)]
    assert len(anchor_msgs) == 1
    anchor_idx = prepared.index(anchor_msgs[0])
    user_idx = next(
        i
        for i, m in enumerate(prepared)
        if isinstance(m, HumanMessage)
        and not is_conversation_anchor_message(m)
        and "继续" in str(m.content)
    )
    manifest_idx = next(i for i, m in enumerate(prepared) if SESSION_MANIFEST_TAG in str(m.content))
    assert manifest_idx < anchor_idx < user_idx
    assert "DeptBigdataCost" in str(anchor_msgs[0].content) or "接口" in str(anchor_msgs[0].content)


def test_legacy_anchor_sections_not_loaded_or_rendered(tmp_path: Path) -> None:
    thread_id = "cli-legacy"
    anchor_path = conversation_anchor_json_path(tmp_path, thread_id)
    anchor_path.parent.mkdir(parents=True)
    anchor_path.write_text(
        json.dumps(
            {
                "version": 1,
                "compression_count": 2,
                "sections": {
                    "session_goal": "做台账",
                    "pending_tasks": "不应出现",
                    "related_code": "并行检索 Top5",
                    "files_modified": "",
                    "decisions": "",
                    "errors_resolved": "",
                    "detail_pointers": "",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = load_anchor_sections(tmp_path, thread_id)
    assert loaded["session_goal"] == "做台账"
    assert "pending_tasks" not in loaded
    assert "related_code" not in loaded

    injected = ensure_messages_include_conversation_anchor(
        tmp_path,
        thread_id,
        [HumanMessage(content="继续")],
    )
    anchor_text = str(
        next(m for m in injected if is_conversation_anchor_message(m)).content
    )
    assert "不应出现" not in anchor_text
    assert "并行检索 Top5" not in anchor_text
    assert "未完成与下一步" not in anchor_text
    assert "相关代码" not in anchor_text


def test_ensure_anchor_does_not_fill_pending_from_last_user(tmp_path: Path) -> None:
    sections = ensure_anchor_sections_minimum(
        {},
        workspace=tmp_path,
        thread_id="t1",
        span_messages=[
            HumanMessage(content="实现数据台账 API"),
            HumanMessage(content="这里只是用户在反思策略，不是待办"),
        ],
    )
    assert "反思策略" not in json.dumps(sections, ensure_ascii=False)
    assert "pending_tasks" not in sections


def test_fallback_decisions_from_assistant_bullets() -> None:
    reply = AIMessage(
        content=(
            "## LangMem 三种记忆\n"
            "- Semantic：存事实与偏好\n"
            "- Episodic：存具体经历与 few-shot\n"
            "- Procedural：存行为规则与 prompt 优化\n"
        )
    )
    body = _fallback_decisions_from_assistant([reply])
    assert "Semantic" in body
    assert "Episodic" in body
    assert "Procedural" in body


def test_ensure_anchor_fills_decisions_when_llm_empty(tmp_path: Path) -> None:
    long_reply = AIMessage(
        content=(
            "## 对比\n"
            "- LangMem 基于 LangGraph BaseStore\n"
            "- Checkpointer 只管同 thread 短期状态\n"
            + ("补充说明 " * 80)
        )
    )
    sections = ensure_anchor_sections_minimum(
        {"session_goal": "介绍 LangMem"},
        workspace=tmp_path,
        thread_id="t1",
        span_messages=[HumanMessage(content="介绍 LangMem"), long_reply],
    )
    assert sections["decisions"].strip()
    assert "LangMem" in sections["decisions"] or "Checkpointer" in sections["decisions"]
