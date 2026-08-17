"""用户问句钉住：<user_query> 包装与步间 reminder。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.context_builder import wrap_user_message_with_context
from llgraph.context.react_step_reminder import (
    append_react_step_reminder_for_dispatch,
    react_step_reminder_content,
)


def test_wrap_puts_user_query_tag() -> None:
    out = wrap_user_message_with_context("是前端 bug 吗？", "## 工具并行\n并行")
    assert "<workspace-context>" in out
    assert "<user_query>" in out
    assert "是前端 bug 吗？" in out
    assert out.index("</workspace-context>") < out.index("<user_query>")


def test_wrap_without_context_still_tags_query() -> None:
    out = wrap_user_message_with_context("hello", "")
    assert out.strip() == "<user_query>\nhello\n</user_query>"


def test_step_reminder_pins_user_query() -> None:
    msgs = [
        HumanMessage(
            content=(
                "<workspace-context>\nx\n</workspace-context>\n\n"
                "<user_query>\n埋点删除后确认无效，是前端 bug 吗\n</user_query>"
            )
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "grep_files", "args": {"pattern": "a"}, "id": "c0"}],
        ),
        ToolMessage(content="hit", tool_call_id="c0", name="grep_files"),
    ]
    text = react_step_reminder_content(msgs)
    assert "<system-reminder>" in text
    assert "钉住 <user_query>" in text
    assert "前端 bug" in text
    assert "工具往返预算" in text
    assert "提纲" not in text

    out = append_react_step_reminder_for_dispatch(msgs)
    assert out[-1].content.startswith("<system-reminder>")
