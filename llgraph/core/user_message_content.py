"""用户消息多模态内容（文本 + 附件引用；首轮出站才加载图片字节）。"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

MAX_CHAT_IMAGES = 50
MAX_IMAGE_BYTES = 5 * 1024 * 1024
_ALLOWED_MEDIA_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
})

IMAGE_RECOGNITION_PREFIX = "[图片识别]"


@dataclass(frozen=True)
class ChatImageInput:
    """multipart 上传或 attachments 读出的单张图片（原始字节）。"""

    media_type: str
    data: bytes


@dataclass(frozen=True)
class StoredImageRef:
    """messages.jsonl 中的图片句柄。"""

    image_id: str
    filename: str
    media_type: str
    caption: str = ""


def _normalize_media_type(raw: str) -> str:
    mt = str(raw or "").strip().lower().split(";", 1)[0].strip()
    if mt == "image/jpg":
        return "image/jpeg"
    return mt


def normalize_uploaded_images(
    items: list[tuple[str, bytes]],
) -> list[ChatImageInput]:
    """
    校验 multipart 上传的图片（原始字节）。

    @param items (media_type, raw_bytes) 列表
    @return 规范化后的图片列表
    @raises ValueError 校验失败
    """
    if not items:
        return []
    if len(items) > MAX_CHAT_IMAGES:
        raise ValueError(f"单次最多上传 {MAX_CHAT_IMAGES} 张图片")
    out: list[ChatImageInput] = []
    for idx, (raw_mt, raw_bytes) in enumerate(items):
        media_type = _normalize_media_type(raw_mt)
        if media_type not in _ALLOWED_MEDIA_TYPES:
            raise ValueError(f"images[{idx}] 不支持的类型: {media_type or '(空)'}")
        if not raw_bytes:
            raise ValueError(f"images[{idx}] 文件为空")
        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"images[{idx}] 超过 {MAX_IMAGE_BYTES // (1024 * 1024)}MB 限制")
        out.append(ChatImageInput(media_type=media_type, data=raw_bytes))
    return out


def build_image_ref_block(ref: StoredImageRef) -> dict[str, Any]:
    """构建 image_ref 块（落盘 / Web 预览）。"""
    block: dict[str, Any] = {
        "type": "image_ref",
        "id": ref.image_id,
        "filename": ref.filename,
        "media_type": _normalize_media_type(ref.media_type),
    }
    caption = str(ref.caption or "").strip()
    if caption:
        block["caption"] = caption
    return block


def image_ref_block_from_stored(block: dict[str, Any]) -> StoredImageRef | None:
    """从 content 块解析 StoredImageRef。"""
    if block.get("type") != "image_ref":
        return None
    image_id = str(block.get("id") or "").strip()
    if not image_id:
        return None
    return StoredImageRef(
        image_id=image_id,
        filename=str(block.get("filename") or "").strip() or image_id,
        media_type=_normalize_media_type(str(block.get("media_type") or "image/png")),
        caption=str(block.get("caption") or "").strip(),
    )


def human_content_has_inline_images(content: Any) -> bool:
    """是否含 Anthropic 内联 image 块（当前轮次未落盘前）。"""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "image"
        for block in content
    )


def human_content_has_image_refs(content: Any) -> bool:
    """是否含 image_ref 块（已落盘轮次）。"""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "image_ref"
        for block in content
    )


def strip_inline_image_blocks(content: Any) -> str | list[dict[str, Any]]:
    """移除内联 image 块，保留 text / image_ref。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    kept: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            continue
        if isinstance(block, dict):
            kept.append(block)
        elif isinstance(block, str) and block.strip():
            kept.append({"type": "text", "text": block})
    if not kept:
        return ""
    if len(kept) == 1 and kept[0].get("type") == "text":
        return str(kept[0].get("text") or "")
    return kept


def _caption_lines_from_refs(blocks: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        ref = image_ref_block_from_stored(block)
        if ref is None or not ref.caption:
            continue
        lines.append(f"{IMAGE_RECOGNITION_PREFIX} {ref.image_id}: {ref.caption}")
    return lines


def human_content_text_for_llm(content: Any) -> str:
    """
    提取发往 LLM 的用户可见文本（不含 image_ref 元数据；caption 合并为文字）。

    @param content HumanMessage content
    @return 纯文本
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    text_parts: list[str] = []
    ref_blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        elif block.get("type") == "image_ref":
            ref_blocks.append(block)
    merged = "".join(text_parts).strip()
    caption_lines = _caption_lines_from_refs(ref_blocks)
    if caption_lines:
        extra = "\n".join(caption_lines)
        merged = f"{merged}\n\n{extra}".strip() if merged else extra
    return merged


def prepare_human_content_for_llm_dispatch(content: Any) -> str | list[dict[str, Any]]:
    """
    出站 LLM 用 HumanMessage content。

    - 已落盘（含 image_ref）：仅文字（含 caption）
    - 当前轮（含内联 image）：保留 multimodal 块供首轮识别
    """
    if human_content_has_image_refs(content):
        text = human_content_text_for_llm(content)
        return text
    if human_content_has_inline_images(content):
        return content
    if isinstance(content, list):
        return human_content_text_for_llm(content)
    return content


def prepare_messages_for_multimodal_dispatch(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """将 HumanMessage 转为 LLM 出站格式（剥离 image_ref，保留当前轮内联图）。"""
    out: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            out.append(msg)
            continue
        new_content = prepare_human_content_for_llm_dispatch(getattr(msg, "content", ""))
        if new_content == getattr(msg, "content", ""):
            out.append(msg)
        else:
            out.append(msg.model_copy(update={"content": new_content}))
    return out


def extract_text_from_human_content(content: Any) -> str:
    """
    从 HumanMessage content 提取 Web 可见用户正文（忽略 image / image_ref）。

    @param content str 或 block 列表
    @return 文本
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def strip_inline_images_from_messages(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], bool]:
    """
    移除 HumanMessage 中已废弃的内联 base64 image 块（仅保留 text / image_ref）。

    @param messages 消息列表
    @return (新列表, 是否有改动)
    """
    changed = False
    out: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and human_content_has_inline_images(
            getattr(msg, "content", "")
        ):
            new_content = strip_inline_image_blocks(getattr(msg, "content", ""))
            out.append(msg.model_copy(update={"content": new_content}))
            changed = True
        else:
            out.append(msg)
    return out, changed


def extract_images_from_human_content(
    content: Any,
    *,
    attachment_url_for: Any | None = None,
) -> list[dict[str, str]]:
    """
    从 HumanMessage content 提取 image_ref（供 Web 历史预览）。

    @param content 消息 content
    @param attachment_url_for 可选 callable(image_id) -> url
    @return [{media_type, url?, id}, ...]
    """
    if not isinstance(content, list):
        return []
    images: list[dict[str, str]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image_ref":
            continue
        ref = image_ref_block_from_stored(block)
        if ref is None:
            continue
        row: dict[str, str] = {
            "id": ref.image_id,
            "media_type": ref.media_type,
        }
        if attachment_url_for is not None:
            row["url"] = str(attachment_url_for(ref.image_id))
        images.append(row)
    return images


def build_human_content_blocks(
    text: str,
    *,
    images: list[ChatImageInput] | None = None,
    context_block: str = "",
) -> str | list[dict[str, Any]]:
    """
    构建首轮发往模型的 HumanMessage content（Anthropic：text + image 块）。

    图片字节仅在此阶段注入，不落 messages.jsonl。
    """
    blocks: list[dict[str, Any]] = []
    user_text = str(text or "").strip()
    ctx = str(context_block or "").strip()
    if ctx:
        wrapped = f"<workspace-context>\n{ctx}\n</workspace-context>"
        if user_text:
            blocks.append({"type": "text", "text": f"{wrapped}\n\n{user_text}"})
        else:
            blocks.append({"type": "text", "text": wrapped})
    elif user_text:
        blocks.append({"type": "text", "text": user_text})
    for img in images or []:
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _normalize_media_type(img.media_type),
                "data": base64.b64encode(img.data).decode("ascii"),
            },
        })
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text") or "")
    return blocks


def build_stored_user_content(
    text: str,
    *,
    image_refs: list[StoredImageRef] | None = None,
) -> str | list[dict[str, Any]]:
    """
    落盘用用户消息（text + image_ref，不含 base64 与 workspace-context）。

    @param text 用户输入
    @param image_refs 附件引用
    @return HumanMessage content
    """
    blocks: list[dict[str, Any]] = []
    user_text = str(text or "").strip()
    if user_text:
        blocks.append({"type": "text", "text": user_text})
    for ref in image_refs or []:
        blocks.append(build_image_ref_block(ref))
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text") or "")
    return blocks
