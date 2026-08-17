"""用户纠正/异议时的 ephemeral dispatch nudge（不落盘、不删历史）。"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.core.user_message_content import extract_text_from_human_content

USER_CORRECTION_NUDGE_MARKER = "[系统·用户异议]"
USER_CORRECTION_NUDGE = (
    "<system-reminder>\n"
    f"{USER_CORRECTION_NUDGE_MARKER} 用户对先前结论提出质疑。"
    "这是**对抗复核**，不是默认改口：先前结论与用户说法**都可能错**，"
    "勿开场「你说的对」，也勿死守旧结论。\n"
    "裁决顺序：\n"
    "1) 先对齐用户给出的**原始证据或指名对象**（贴出的原文、堆栈首行、"
    "明确点名的字符串/字段/路径、用户说在核对的那一项等）——"
    "相关现象 ≠ 同一对象；找到「能解释类似现象的别的东西」"
    "≠ 已回答用户问的那一项。\n"
    "用户点名某一具体字面量/主张时，第一句须直接回应该项"
    "（成立 / 不成立 / 证据不足），禁止用旁支机制顶替。\n"
    "2) 用工具核对对立点；相关检索词尽量合并进一条 grep。\n"
    "3) 终答写清：哪一侧成立/不成立、依据是哪段原文或哪个 path，"
    "以及断言适用范围（哪条路径 / 组件 / 前提）。"
    "标题/首句与后文机制须一致。证据不足则写不确定，禁止附和。\n"
    "</system-reminder>"
)

# 通用异议/纠正意图（不问具体业务词：日志、关键字、cursor 等）
_CORRECTION_PATTERNS = (
    re.compile(r"(搞错|说错|不对|有误|不一致|矛盾|并非如此|答非所问|跑偏|跑题)"),
    re.compile(r"不是.{1,40}(吗|么)[？?]?"),
    re.compile(r"(和我问的|我问的是|我在问的?)"),
    re.compile(
        r"\b(?:that'?s|you\s+are)\s+wrong\b|\bincorrect\b|"
        r"\bnot\s+what\s+I\s+(?:asked|meant)\b",
        re.IGNORECASE,
    ),
)


def _is_ephemeral_nudge(msg: BaseMessage) -> bool:
    if not isinstance(msg, HumanMessage):
        return False
    text = str(getattr(msg, "content", "") or "").strip()
    return (
        text.startswith("<system-reminder>")
        or text.startswith("[系统·调查重置]")
        or text.startswith("[系统·用户异议]")
        or text.startswith("[系统·ReAct")
    )


def _last_real_human_index(messages: list[BaseMessage]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage) and not _is_ephemeral_nudge(msg):
            return idx
    return -1


def _tool_rounds_since_last_user(messages: list[BaseMessage]) -> int:
    start = _last_real_human_index(messages)
    if start < 0:
        return 0
    rounds = 0
    for msg in messages[start + 1 :]:
        if isinstance(msg, AIMessage) and ai_message_tool_calls(msg):
            rounds += 1
    return rounds


def looks_like_user_correction(text: str) -> bool:
    """
    是否像用户对先前结论的纠正/异议。

    @param text 用户正文
    @return 是否匹配
    """
    sample = str(text or "").strip()
    if not sample:
        return False
    if re.fullmatch(
        r"(?:对吗|是吗|对不对|真的吗|你确定)\??|"
        r"(?:really|sure)\??|"
        r"is\s+that\s+(?:right|correct)\??",
        sample,
        re.IGNORECASE,
    ):
        return True
    if len(sample) < 4:
        return False
    return any(p.search(sample) for p in _CORRECTION_PATTERNS)


def has_prior_assistant_reply(
    messages: list[BaseMessage],
    *,
    before_idx: int | None = None,
) -> bool:
    """
    当前用户轮之前是否已有助手可见正文结论（不含纯 tool 调用）。

    @param messages 消息列表
    @param before_idx 截止下标（不含）；默认取最近真实 Human 下标
    @return 是否已有先前终答/正文
    """
    end = before_idx if before_idx is not None else _last_real_human_index(messages)
    if end < 0:
        end = len(messages)
    for msg in messages[:end]:
        if not isinstance(msg, AIMessage):
            continue
        if ai_message_tool_calls(msg):
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            # 占位「无用户可见正文」不算终答
            if "无用户可见正文" in content and len(content) < 80:
                continue
            return True
        if isinstance(content, list) and any(
            isinstance(b, dict) and (b.get("text") or "").strip() for b in content
        ):
            return True
    return False


def should_inject_user_correction_nudge(messages: list[BaseMessage]) -> bool:
    """
    是否在发往模型前注入纠正 nudge（每 user 轮仅一次，轮首）。

    @param messages 出站消息列表
    @return 是否注入
    """
    user_idx = _last_real_human_index(messages)
    if user_idx < 0:
        return False
    # 首轮用户消息（会话尚无助手结论）绝不注入「异议/重查」
    if not has_prior_assistant_reply(messages, before_idx=user_idx):
        return False
    if _tool_rounds_since_last_user(messages) > 0:
        return False
    text = extract_text_from_human_content(messages[user_idx].content)
    if not looks_like_user_correction(text):
        return False
    for msg in messages[user_idx + 1 :]:
        if isinstance(msg, HumanMessage):
            body = str(getattr(msg, "content", "") or "").strip()
            if USER_CORRECTION_NUDGE_MARKER in body:
                return False
    return True


def append_user_correction_nudge_for_dispatch(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """
    纠正态 ephemeral 提醒（不写 checkpoint）。

    @param messages 即将发往模型的消息
    @return 可能追加 HumanMessage 的副本
    """
    if not should_inject_user_correction_nudge(messages):
        return messages
    return [*messages, HumanMessage(content=USER_CORRECTION_NUDGE)]
