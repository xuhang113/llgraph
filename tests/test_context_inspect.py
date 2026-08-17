"""context detail API 测试。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from llgraph.context.context_inspect import collect_context_detail
from llgraph.context.context_session import ContextSession
from llgraph.context.conversation_anchor import build_conversation_anchor_message
from llgraph.session.session_file_store import save_session_messages
from llgraph.session.user_storage import session_thread_dir


def test_collect_context_detail_includes_messages(tmp_path: Path) -> None:
    ws = tmp_path
    tid = "cli-detail1"
    session_thread_dir(ws, tid).mkdir(parents=True, exist_ok=True)
    anchor = build_conversation_anchor_message(
        ws,
        tid,
        {"session_goal": "测试", "files_modified": "", "decisions": ""},
    )
    save_session_messages(
        ws,
        tid,
        [
            anchor,
            HumanMessage(content="你好"),
            AIMessage(content="你好，有什么可以帮你？"),
        ],
    )

    detail = collect_context_detail(
        ws,
        context_session=ContextSession(),
        thread_id=tid,
        allow_write=False,
        max_preview_chars=2000,
    )
    data = detail.to_dict()

    assert data["usage"]["message_count"] >= 3
    assert len(data["stored_messages"]) >= 3
    assert any(m["kind"] == "anchor" for m in data["stored_messages"])
    assert any(m["kind"] == "user" for m in data["stored_messages"])
    assert data["fixed_sections"]
    assert "system_prompt" in {s["key"] for s in data["fixed_sections"]}
    assert data["breakdown_sections"]
    assert any(s["key"] == "system_prompt" for s in data["breakdown_sections"])
    assert any(s["key"] == "conversation" for s in data["breakdown_sections"])
    summarized = next(
        (s for s in data["breakdown_sections"] if s["key"] == "summarized_conversation"),
        None,
    )
    assert summarized is not None
    assert summarized["tokens"] > 0
    assert summarized.get("messages")
    assert any(m["kind"] == "anchor" for m in summarized["messages"])
    conv = next(s for s in data["breakdown_sections"] if s["key"] == "conversation")
    assert conv.get("messages")
    assert any(m["kind"] == "user" for m in conv["messages"])
    assert data["compress_threshold"] > 0
    assert "compress_strategy" in data["settings"]
