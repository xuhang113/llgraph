"""用户消息多模态内容测试。"""

from __future__ import annotations

import base64

import pytest
from langchain_core.messages import HumanMessage

from llgraph.core.user_message_content import (
    StoredImageRef,
    build_human_content_blocks,
    build_stored_user_content,
    extract_images_from_human_content,
    extract_text_from_human_content,
    normalize_uploaded_images,
    prepare_human_content_for_llm_dispatch,
    strip_inline_images_from_messages,
)


def test_normalize_uploaded_images_png() -> None:
    raw = b"\x89PNG\r\n\x1a\n"
    images = normalize_uploaded_images([("image/png", raw)])
    assert len(images) == 1
    assert images[0].media_type == "image/png"
    assert images[0].data == raw


def test_build_stored_user_content_with_image_ref() -> None:
    ref = StoredImageRef(
        image_id="img-abc",
        filename="img-abc.png",
        media_type="image/png",
        caption="界面报错弹窗",
    )
    content = build_stored_user_content("这是什么？", image_refs=[ref])
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_ref"
    assert content[1]["id"] == "img-abc"
    assert extract_text_from_human_content(content) == "这是什么？"
    imgs = extract_images_from_human_content(
        content,
        attachment_url_for=lambda i: f"/api/x/sessions/t/attachments/{i}",
    )
    assert len(imgs) == 1
    assert imgs[0]["url"].endswith("/attachments/img-abc")


def test_dispatch_strips_image_ref_keeps_caption_text() -> None:
    ref = StoredImageRef(
        image_id="img-1",
        filename="img-1.png",
        media_type="image/png",
        caption="红色按钮在右上角",
    )
    stored = build_stored_user_content("分析", image_refs=[ref])
    dispatched = prepare_human_content_for_llm_dispatch(stored)
    assert isinstance(dispatched, str)
    assert "分析" in dispatched
    assert "[图片识别]" in dispatched
    assert "红色按钮" in dispatched


def test_dispatch_keeps_inline_image_for_current_turn() -> None:
    from llgraph.core.user_message_content import ChatImageInput

    content = build_human_content_blocks(
        "识别",
        images=[ChatImageInput(media_type="image/jpeg", data=b"fakejpeg")],
    )
    dispatched = prepare_human_content_for_llm_dispatch(content)
    assert isinstance(dispatched, list)
    assert any(b.get("type") == "image" for b in dispatched)
    block = next(b for b in dispatched if b.get("type") == "image")
    assert block["source"]["data"] == base64.b64encode(b"fakejpeg").decode("ascii")


def test_strip_inline_images_from_stored_messages() -> None:
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "看图"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
            },
        ]
    )
    cleaned, changed = strip_inline_images_from_messages([msg])
    assert changed
    assert cleaned[0].content == "看图"


def test_normalize_rejects_oversized() -> None:
    big = b"x" * (6 * 1024 * 1024)
    with pytest.raises(ValueError, match="超过"):
        normalize_uploaded_images([("image/png", big)])
