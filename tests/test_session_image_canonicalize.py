"""图片落盘：think_nudge 之后仍须把内联图写成 image_ref。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from llgraph.core.user_message_content import StoredImageRef, human_content_has_image_refs
from llgraph.session.session_image_store import canonicalize_messages_image_refs


def test_canonicalize_image_refs_despite_think_nudge() -> None:
    refs = [
        StoredImageRef(
            image_id="img-test-1",
            filename="job.png",
            media_type="image/png",
        )
    ]
    user_with_image = HumanMessage(
        content=[
            {"type": "text", "text": "<user_query>\n\n</user_query>"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aaaa",
                },
            },
        ]
    )
    thinking_ai = AIMessage(content="")
    nudge = HumanMessage(content="[系统] 你上一轮仅在 thinking，请继续输出可见答复或工具调用。")
    final_ai = AIMessage(content="这是 XXL-JOB 任务 allRefreshValidCouponInfo")

    out = canonicalize_messages_image_refs(
        [user_with_image, thinking_ai, nudge, final_ai],
        turn_image_refs=refs,
    )
    humans = [m for m in out if isinstance(m, HumanMessage)]
    assert len(humans) == 2
    assert human_content_has_image_refs(humans[0].content)
    # nudge 不应被当成挂图目标；用户消息须保留 image_ref
    assert not human_content_has_image_refs(humans[1].content)


def test_canonicalize_strips_orphan_inline_when_no_refs() -> None:
    user_with_image = HumanMessage(
        content=[
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "x"},
            }
        ]
    )
    out = canonicalize_messages_image_refs([user_with_image], turn_image_refs=None)
    assert isinstance(out[0], HumanMessage)
    content = out[0].content
    if isinstance(content, list):
        assert not any(isinstance(b, dict) and b.get("type") == "image" for b in content)
