"""用户纠正 nudge：首轮无助手结论时不注入；文案为双向对抗复核。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from llgraph.context.user_correction_nudge import (
    USER_CORRECTION_NUDGE,
    append_user_correction_nudge_for_dispatch,
    looks_like_user_correction,
    should_inject_user_correction_nudge,
)


def test_no_nudge_on_first_user_turn() -> None:
    msgs = [
        HumanMessage(content="埋点属性删除后滚动条变短，确认却没删掉，是前端 bug 吗？"),
    ]
    assert should_inject_user_correction_nudge(msgs) is False


def test_nudge_when_prior_assistant_and_correction() -> None:
    msgs = [
        HumanMessage(content="这个按钮怎么点？"),
        AIMessage(content="点右侧删除即可。"),
        HumanMessage(content="不对，确认之后属性还在。"),
    ]
    assert should_inject_user_correction_nudge(msgs) is True
    out = append_user_correction_nudge_for_dispatch(msgs)
    body = str(out[-1].content)
    assert "对抗复核" in body
    assert "都可能错" in body
    assert "勿开场「你说的对」" in body
    assert "原始证据" in body
    assert "指名对象" in body or "点名" in body
    assert "旁支" in body
    assert body == USER_CORRECTION_NUDGE


def test_correction_patterns_cover_keyword_mismatch() -> None:
    """通用异议意图：不对齐问句 / 纠正，不绑日志关键字等业务词。"""
    assert looks_like_user_correction(
        "感觉你回答的和我问的不一样呢"
    )
    assert looks_like_user_correction("我在问的是另一件事")
    assert looks_like_user_correction("答非所问")
    assert looks_like_user_correction("对吗")
    assert looks_like_user_correction("that's wrong")
    assert not looks_like_user_correction("ok")
    assert not looks_like_user_correction("INFO/WARN 级别的日志我也没看到")
