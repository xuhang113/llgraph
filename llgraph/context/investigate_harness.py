"""ReAct 稳定性辅助：字面量优先 grep、工具往返上限收口。

意图与排查/归因策略由模型自行判断；本模块不按问句分流、不拦截「未写提纲」的工具。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.core.agent_config import load_agent_config
from llgraph.core.search_path_tracker import extract_grep_batch_terms
from llgraph.core.user_message_content import extract_text_from_human_content
from llgraph.context.chat_history_repair import ai_message_tool_calls

SOFT_CLOSE_NUDGE_MARKER = "[系统·防偏航收口]"
SOFT_CLOSE_TOOL_BLOCK_MSG = (
    f"{SOFT_CLOSE_NUDGE_MARKER} 已达工具往返上限：工具已拦截。"
    "禁止再开检索；根据已有证据写终答；证据不足写不确定。"
)

# 历史会话兼容（旧提纲门 / 量变门标记仍视为 ephemeral）
OUTLINE_GATE_MARKER = "[系统·排查提纲]"
_SOFT_CLOSE_NUDGE_MARKER_LEGACY = "[系统·探索预算]"
DEFAULT_SOFT_CLOSE_AFTER_ROUNDS = 20
SOFT_CLOSE_AFTER_ROUNDS_CAP = 80

_USER_QUERY_RE = re.compile(
    r"<user_query>\s*([\s\S]*?)\s*</user_query>",
    re.IGNORECASE,
)
_WORKSPACE_CTX_RE = re.compile(
    r"<workspace-context>[\s\S]*?</workspace-context>\s*",
    re.IGNORECASE,
)
_CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_FILE_EXT_RE = re.compile(
    r"\b[\w./-]+\.(?:vue|ts|tsx|js|jsx|java|py|go|xml)\b",
    re.IGNORECASE,
)
_PROJECT_HINT_RE = re.compile(
    r"(?:应该是|属于|在)?\s*([a-zA-Z][\w.-]{2,40})\s*项目"
    r"|([a-zA-Z][\w.-]*workstation[\w.-]*)"
    r"|([a-zA-Z][\w.-]{3,40})\s*(?:前端|后端|仓库)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InvestigateHarnessSettings:
    """ReAct 稳定性开关（不含意图分流）。"""

    guard_parallel_when_literals: bool = True
    # 已废弃：默认关；保留键仅为兼容旧 agent.json
    outline_gate: bool = False
    # 工具往返达到后注入强制结案；0=关闭；适用于所有任务
    soft_close_after_rounds: int = DEFAULT_SOFT_CLOSE_AFTER_ROUNDS


def resolve_investigate_harness_settings(workspace: Path | None) -> InvestigateHarnessSettings:
    """
    读取 agent.json → agent.investigate（及扁平兼容键）。

    @param workspace 工作区根
    @return 设置
    """
    raw: dict = {}
    if workspace is not None:
        cfg = load_agent_config(workspace)
        agent = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
        inv = agent.get("investigate") if isinstance(agent.get("investigate"), dict) else {}
        raw = {**agent, **inv}

    def _bool(key: str, default: bool) -> bool:
        if key not in raw:
            return default
        val = raw[key]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "on"}
        return default

    def _int(key: str, default: int, *, lo: int, hi: int) -> int:
        if key not in raw:
            return default
        try:
            return max(lo, min(hi, int(raw[key])))
        except (TypeError, ValueError):
            return default

    return InvestigateHarnessSettings(
        guard_parallel_when_literals=_bool("guard_parallel_when_literals", True),
        outline_gate=False,  # 强制关闭：框架不因提纲干预工具
        soft_close_after_rounds=_int(
            "soft_close_after_rounds",
            DEFAULT_SOFT_CLOSE_AFTER_ROUNDS,
            lo=0,
            hi=SOFT_CLOSE_AFTER_ROUNDS_CAP,
        ),
    )


def is_ephemeral_harness_human(msg: BaseMessage) -> bool:
    """是否为 dispatch/图内临时 Human（探索预算、system-reminder 等）。"""
    if not isinstance(msg, HumanMessage):
        return False
    text = str(getattr(msg, "content", "") or "").strip()
    return (
        text.startswith("<system-reminder>")
        or text.startswith("[系统·结案合同]")
        or text.startswith(SOFT_CLOSE_NUDGE_MARKER)
        or text.startswith(OUTLINE_GATE_MARKER)
        or text.startswith("[系统·量变写回门]")
        or text.startswith(_SOFT_CLOSE_NUDGE_MARKER_LEGACY)
        or text.startswith("[系统·ReAct")
        or text.startswith("[系统·调查重置]")
        or text.startswith("[系统·用户异议]")
        or text.startswith("[系统] 你上一轮仅在 thinking")
    )


def strip_ephemeral_harness_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """落盘前去掉临时 harness Human。"""
    return [m for m in messages if not is_ephemeral_harness_human(m)]


def last_real_user_text(messages: list[BaseMessage]) -> str:
    """
    最近一条真实用户正文（去 workspace-context / 取 user_query）。

    @param messages 对话消息
    @return 用户原文
    """
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        if is_ephemeral_harness_human(msg):
            continue
        raw = extract_text_from_human_content(msg.content)
        m = _USER_QUERY_RE.search(raw)
        text = (m.group(1) if m else _WORKSPACE_CTX_RE.sub("", raw)).strip()
        if text:
            return text
    return ""


def user_has_directed_search_literals(text: str) -> bool:
    """
    用户消息是否已含可定向 grep 的字面量（此时应禁止先 parallel）。

    @param text 用户正文
    @return 是否应优先 grep
    """
    sample = str(text or "").strip()
    if not sample:
        return False
    terms = extract_grep_batch_terms(sample)
    if len(terms) >= 2:
        return True
    if len(terms) >= 1 and any(len(t) >= 6 for t in terms):
        return True
    if _CAMEL_RE.search(sample) or _FILE_EXT_RE.search(sample):
        return True
    if re.search(r"\btb_[a-z][\w]*\b", sample, re.IGNORECASE):
        return True
    return False


def suggest_grep_pattern_from_user(text: str, *, max_terms: int = 12) -> str:
    """从用户消息拼一条建议 grep pattern。"""
    terms = list(extract_grep_batch_terms(text))
    for m in _CAMEL_RE.findall(text or ""):
        if m.lower() not in {t.lower() for t in terms}:
            terms.append(m)
    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", text or ""):
        zh = m.group(0)
        if zh.lower() not in {t.lower() for t in terms} and len(terms) < max_terms:
            terms.append(zh)
    if not terms:
        return "关键词A|关键词B"
    return "|".join(terms[:max_terms])


def parallel_blocked_for_literals_message(user_text: str) -> str:
    """字面量场景拦截 parallel 的返回文案。"""
    pat = suggest_grep_pattern_from_user(user_text)
    return (
        "【llgraph 拦截】当前用户问题已含可检索字面量，禁止先调用 search_code_parallel。\n"
        f"请改用：`grep_files(pattern=\"{pat}\", path=\".\")`，命中后再 `read_files` 一次读齐。\n"
        "仅当完全不知模块/仓库且无任何符号线索时，才允许 parallel（且每问 ≤1 次）。"
    )


def count_tool_rounds_since_user(messages: list[BaseMessage]) -> int:
    """自最近真实 user 以来，含 tool_calls 的 assistant 轮数。"""
    start = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage) and not is_ephemeral_harness_human(msg):
            start = idx
            break
    if start < 0:
        return 0
    rounds = 0
    for msg in messages[start + 1 :]:
        if isinstance(msg, AIMessage) and ai_message_tool_calls(msg):
            rounds += 1
    return rounds


def suggest_repo_path_hint(user_text: str) -> str:
    """
    从用户话里抽仓库/项目线索（启发式，可为空；不注入决策文案）。

    @param user_text 用户正文
    @return 建议 path 前缀或空串
    """
    sample = str(user_text or "")
    for m in _PROJECT_HINT_RE.finditer(sample):
        token = next((g for g in m.groups() if g), "")
        token = str(token or "").strip(".,;:：")
        if len(token) >= 3:
            return token
    return ""


def _tool_call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or "").strip()
    return str(getattr(call, "id", "") or "").strip()


def _tool_call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "").strip()
    return str(getattr(call, "name", "") or "").strip()


def soft_close_nudge_pending(messages: list[BaseMessage]) -> bool:
    """本用户问题是否已注入强制结案提示。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not is_ephemeral_harness_human(msg):
            break
        if isinstance(msg, HumanMessage):
            body = str(getattr(msg, "content", "") or "")
            if SOFT_CLOSE_NUDGE_MARKER in body or _SOFT_CLOSE_NUDGE_MARKER_LEGACY in body:
                return True
    return False


def should_inject_soft_close(
    messages: list[BaseMessage],
    *,
    workspace: Path | None = None,
) -> bool:
    """
    工具往返达阈值：稳定性收口（禁新检索），不区分问句类型。

    @param messages 当前消息
    @param workspace 工作区根
    @return 是否注入
    """
    settings = resolve_investigate_harness_settings(workspace)
    if settings.soft_close_after_rounds <= 0:
        return False
    if soft_close_nudge_pending(messages):
        return False
    return count_tool_rounds_since_user(messages) >= settings.soft_close_after_rounds


def build_soft_close_nudge(
    messages: list[BaseMessage],
    *,
    workspace: Path | None = None,
) -> str:
    """防偏航收口 Human 正文（与 SOFT_CLOSE_TOOL_BLOCK_MSG 口径一致）。"""
    settings = resolve_investigate_harness_settings(workspace)
    used = count_tool_rounds_since_user(messages)
    limit = settings.soft_close_after_rounds
    return (
        f"{SOFT_CLOSE_NUDGE_MARKER} 工具往返已达 {used}（上限 {limit}）。\n"
        "**本轮禁止再开一切工具。** 根据已有证据写终答；证据不足写不确定。"
    )


def append_soft_close_for_dispatch(
    messages: list[BaseMessage],
    *,
    workspace: Path | None = None,
) -> list[BaseMessage]:
    """工具回程后 ephemeral 强制结案（不落盘）。"""
    if not should_inject_soft_close(messages, workspace=workspace):
        return messages
    return [*messages, HumanMessage(content=build_soft_close_nudge(messages, workspace=workspace))]


def guard_soft_close_tools(
    state: dict[str, Any],
    *,
    workspace: Path | None = None,
) -> tuple[dict[str, Any], list[ToolMessage]]:
    """
    SoftClose 已注入后仍发工具：拦截全部 tool_calls。

    @return (改写 state, 占位 ToolMessage)
    """
    settings = resolve_investigate_harness_settings(workspace)
    if settings.soft_close_after_rounds <= 0:
        return state, []
    messages = list(state.get("messages") or [])
    if not soft_close_nudge_pending(messages):
        if count_tool_rounds_since_user(messages) < settings.soft_close_after_rounds:
            return state, []

    blocked: list[ToolMessage] = []
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, AIMessage):
            continue
        calls = ai_message_tool_calls(msg)
        if not calls:
            continue
        for call in calls:
            cid = _tool_call_id(call) or "soft-close"
            name = _tool_call_name(call) or "tool"
            blocked.append(
                ToolMessage(
                    content=SOFT_CLOSE_TOOL_BLOCK_MSG,
                    tool_call_id=cid,
                    name=name,
                )
            )
        patched = (
            messages[:idx]
            + [msg.model_copy(update={"tool_calls": []})]
            + messages[idx + 1 :]
        )
        state = {**state, "messages": patched}
        break
    return state, blocked
