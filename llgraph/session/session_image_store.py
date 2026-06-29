"""会话图片附件：落盘 attachments/，messages.jsonl 仅存 image_ref。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage

from llgraph.core.user_message_content import (
    ChatImageInput,
    StoredImageRef,
    build_image_ref_block,
    extract_text_from_human_content,
    human_content_has_inline_images,
    image_ref_block_from_stored,
    strip_inline_image_blocks,
)
from llgraph.session.user_storage import session_attachments_dir

_IMAGE_ID_RE = re.compile(r"^img-[a-z0-9-]+$", re.I)
_EXT_BY_MEDIA = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def _ext_for_media_type(media_type: str) -> str:
    return _EXT_BY_MEDIA.get(media_type.lower(), ".bin")


def _safe_image_id(raw: str) -> str:
    text = str(raw or "").strip()
    if not _IMAGE_ID_RE.match(text):
        raise ValueError("无效的图片 ID")
    return text


def save_chat_images(
    workspace: Path,
    thread_id: str,
    images: list[ChatImageInput],
) -> list[StoredImageRef]:
    """
    将上传图片写入 sessions/<thread_id>/attachments/。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param images 校验后的图片
    @return 落盘引用列表
    """
    if not images:
        return []
    att_dir = session_attachments_dir(workspace, thread_id)
    att_dir.mkdir(parents=True, exist_ok=True)
    refs: list[StoredImageRef] = []
    for img in images:
        image_id = f"img-{uuid.uuid4().hex[:12]}"
        filename = f"{image_id}{_ext_for_media_type(img.media_type)}"
        path = att_dir / filename
        path.write_bytes(img.data)
        refs.append(
            StoredImageRef(
                image_id=image_id,
                filename=filename,
                media_type=img.media_type,
                caption="",
            )
        )
    return refs


def resolve_attachment_file(
    workspace: Path,
    thread_id: str,
    image_id: str,
) -> Path | None:
    """
    解析附件文件路径（防路径穿越）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param image_id 图片 ID
    @return 文件路径；不存在返回 None
    """
    safe_id = _safe_image_id(image_id)
    att_dir = session_attachments_dir(workspace, thread_id)
    if not att_dir.is_dir():
        return None
    for path in att_dir.iterdir():
        if path.is_file() and path.stem == safe_id:
            return path
    return None


def load_chat_image_input(
    workspace: Path,
    thread_id: str,
    ref: StoredImageRef,
) -> ChatImageInput:
    """
    从 attachments 读取图片，供首轮 multimodal 出站（仅内存/网关，不落 jsonl base64）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param ref 图片引用
    @return ChatImageInput
    """
    path = resolve_attachment_file(workspace, thread_id, ref.image_id)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"附件不存在: {ref.image_id}")
    data = path.read_bytes()
    return ChatImageInput(media_type=ref.media_type, data=data)


def load_chat_images_as_input(
    workspace: Path,
    thread_id: str,
    refs: list[StoredImageRef],
) -> list[ChatImageInput]:
    """批量加载附件为 ChatImageInput。"""
    return [load_chat_image_input(workspace, thread_id, ref) for ref in refs]


def replace_human_inline_images_with_refs(
    content: object,
    refs: list[StoredImageRef],
) -> str | list[dict]:
    """
    将 HumanMessage 内联 image 块替换为 image_ref（落盘/持久化用）。

    @param content HumanMessage content
    @param refs 本轮附件引用
    @return 替换后的 content
    """
    if not refs:
        return strip_inline_image_blocks(content)
    stripped = strip_inline_image_blocks(content)
    ref_blocks = [build_image_ref_block(r) for r in refs]
    if isinstance(stripped, str):
        text = stripped.strip()
        if not text and not ref_blocks:
            return ""
        if not ref_blocks:
            return text
        blocks: list[dict] = []
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.extend(ref_blocks)
        if len(blocks) == 1 and blocks[0].get("type") == "text":
            return str(blocks[0].get("text") or "")
        return blocks
    if isinstance(stripped, list):
        blocks = list(stripped)
        blocks.extend(ref_blocks)
        return blocks
    return stripped


def canonicalize_messages_image_refs(
    messages: list[BaseMessage],
    *,
    turn_image_refs: list[StoredImageRef] | None,
) -> list[BaseMessage]:
    """
    轮次结束后：最后一条含内联图的 user 消息改为 image_ref；其余去掉残留内联图。

    @param messages Agent 消息列表
    @param turn_image_refs 本轮新保存的附件
    @return 规范化后的消息
    """
    if not messages:
        return messages
    last_human_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = idx
    out: list[BaseMessage] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, HumanMessage):
            out.append(msg)
            continue
        content = getattr(msg, "content", "")
        if (
            idx == last_human_idx
            and turn_image_refs
            and human_content_has_inline_images(content)
        ):
            new_content = replace_human_inline_images_with_refs(content, turn_image_refs)
            out.append(msg.model_copy(update={"content": new_content}))
            continue
        if human_content_has_inline_images(content):
            new_content = strip_inline_image_blocks(content)
            out.append(msg.model_copy(update={"content": new_content}))
            continue
        out.append(msg)
    return out


def attachment_api_path(slug: str, thread_id: str, image_id: str) -> str:
    """Web 预览 URL（含 /api 前缀）。"""
    return f"/api/workspaces/{slug}/sessions/{thread_id}/attachments/{image_id}"
