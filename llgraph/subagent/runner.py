"""统一 run_subagent：构建图 → fork → invoke → 摘要结果。"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable
from typing import Any

from llgraph.core.agent import build_system_prompt
from llgraph.core.llm import create_gateway_llm
from llgraph.core.tools import get_agent_tools
from llgraph.subagent.engine import (
    ReactSubgraphSpec,
    build_react_subgraph,
    collect_subgraph_messages,
    extract_subagent_result_text,
    invoke_react_subgraph_turn,
)
from llgraph.subagent.profile import (
    SubagentProfile,
    get_subagent_profile,
    resolve_role_block,
)
from llgraph.subagent.result import SubagentResult
from llgraph.subagent.runtime import (
    SubagentRuntime,
    fork_subagent_runtime,
    subagent_edit_tracker,
)


def _notify_parent_subagent(
    parent: SubagentRuntime,
    *,
    event_type: str,
    kind: str,
    sub_id: str,
    sub_thread: str,
    title: str,
    status: str,
) -> None:
    """向父会话 SSE 推送子 Agent 登记变更（供侧栏 refreshTree）。"""
    emit = parent.sse_emit
    if emit is None:
        return
    emit(
        {
            "type": event_type,
            "kind": kind,
            "sub_id": sub_id,
            "sub_thread": sub_thread,
            "title": title,
            "status": status,
            "subgraph_kind": kind,
        }
    )


def new_subagent_id(prefix: str = "") -> str:
    """生成短 sub_id（如 a1b2c3）。"""
    token = secrets.token_hex(3)
    p = (prefix or "").strip()
    return f"{p}{token}" if p else token


def build_subagent_system_prompt(
    runtime: SubagentRuntime,
    role_block: str,
    *,
    allow_write: bool,
) -> str:
    """Agent 工具规范 + 角色块。"""
    base = build_system_prompt(
        runtime.workspace,
        allow_write=allow_write,
        web_search_enabled=runtime.web_search_enabled,
    )
    if runtime.sandbox_policy is not None and runtime.sandbox_policy.enabled:
        from llgraph.config.sandbox_settings import format_sandbox_config_hint

        base = (
            f"{base}\n\n"
            f"OS 沙箱已启用（{runtime.sandbox_policy.backend}，mode={runtime.sandbox_policy.mode}）。\n"
            f"{format_sandbox_config_hint(runtime.workspace)}"
        )
    return f"{base}{role_block}"


def run_subagent(
    parent: SubagentRuntime,
    *,
    kind: str = "explore",
    user_prompt: str,
    sub_id: str | None = None,
    role_block: str | None = None,
    allow_write: bool | None = None,
    max_turns: int | None = None,
    title: str = "",
    profile: SubagentProfile | None = None,
    enable_spawn_tools: bool = False,
    on_started: Callable[[str, str, str], None] | None = None,
) -> SubagentResult:
    """
    启动一轮隔离子 Agent，返回摘要级结果。

    @param parent 父运行时（未 fork）
    @param kind explore | general
    @param user_prompt 指派任务（完整上下文由调用方写入 prompt）
    @param sub_id 子 id；空则自动生成
    @param role_block 覆盖 Profile 角色块
    @param allow_write 覆盖 Profile 写权限
    @param max_turns 覆盖步数
    @param title 展示标题
    @param profile 显式 Profile
    @param enable_spawn_tools 子 Agent 是否再挂 spawn（默认否，防嵌套）
    @param on_started 启动登记后回调 (sub_thread, title, kind)，供父 Trace 即时入口
    @return SubagentResult
    """
    from llgraph.subagent.registry import register_subagent_child

    prof = profile or get_subagent_profile(kind, workspace=parent.workspace)
    sid = (sub_id or "").strip() or new_subagent_id()
    sub_thread = prof.format_sub_thread(parent.parent_thread_id, sid)
    write = prof.allow_write if allow_write is None else bool(allow_write)
    turns = max_turns if max_turns is not None else prof.max_turns
    label = (title or prof.title or prof.kind).strip()

    # 启动即登记，侧栏在跑中即可点进子会话
    register_subagent_child(
        parent.workspace,
        parent.parent_thread_id,
        {
            "kind": prof.kind,
            "sub_id": sid,
            "sub_thread": sub_thread,
            "title": label,
            "status": "running",
        },
    )
    _notify_parent_subagent(
        parent,
        event_type="subagent_started",
        kind=prof.kind,
        sub_id=sid,
        sub_thread=sub_thread,
        title=label,
        status="running",
    )
    if on_started is not None:
        try:
            on_started(sub_thread, label, prof.kind)
        except Exception:
            pass

    child = fork_subagent_runtime(
        parent,
        sub_thread=sub_thread,
        subgraph_kind=prof.kind,
        task_id=sid,
        allow_write=write,
        max_turns=turns,
    )

    role = resolve_role_block(prof, parent.workspace, role_block)
    system_prompt = build_subagent_system_prompt(child, role, allow_write=write)
    edit_tracker = subagent_edit_tracker(child, sub_thread, allow_write=write)
    mcp = list(child.mcp_tools) if prof.include_mcp else []
    tools = get_agent_tools(
        workspace_root=child.workspace,
        allow_write=write,
        mcp_tools=mcp,
        web_search_enabled=child.web_search_enabled,
        sandbox_policy=child.sandbox_policy,
        edit_tracker=edit_tracker,
        on_file_changed=child.on_file_changed,
        write_failure_tracker=child.write_failure_tracker,
    )
    if enable_spawn_tools:
        from llgraph.subagent.agent_tools import create_subagent_tools

        tools = [
            *tools,
            *create_subagent_tools(parent_runtime=child, nestable=False),
        ]

    llm = create_gateway_llm(child.workspace)
    spec = ReactSubgraphSpec(
        node_id=prof.kind,
        subgraph_kind=prof.kind,
        thread_suffix=prof.thread_suffix,
    )
    subgraph = build_react_subgraph(
        llm,
        tools,
        system_prompt,
        workspace=child.workspace,
        thread_key=sub_thread,
        subgraph_kind=None,
    )

    try:
        raw = invoke_react_subgraph_turn(
            child,
            subgraph,
            user_prompt,
            sub_thread=sub_thread,
            role_label=label,
            spec=spec,
            allow_write=write,
            recursion_limit=turns,
        )
        status = "ok"
    except Exception as exc:  # noqa: BLE001 — 摘要回父，避免拖垮主会话
        raw = f"subagent failed: {exc}"
        status = "failed"

    messages = collect_subgraph_messages(subgraph, sub_thread)
    if not (raw or "").strip():
        raw = extract_subagent_result_text(messages, subgraph_kind=None)

    summary = (raw or "").strip()
    files_changed = edit_tracker.unique_paths() if edit_tracker is not None else []

    register_subagent_child(
        parent.workspace,
        parent.parent_thread_id,
        {
            "kind": prof.kind,
            "sub_id": sid,
            "sub_thread": sub_thread,
            "title": label,
            "status": status,
        },
    )
    _notify_parent_subagent(
        parent,
        event_type="subagent_updated",
        kind=prof.kind,
        sub_id=sid,
        sub_thread=sub_thread,
        title=label,
        status=status,
    )

    # 落盘 messages（与 Plan subgraphs 目录对齐：subagents/{kind}-{id}/）
    try:
        from llgraph.subagent.persist import persist_subagent_messages

        persist_subagent_messages(
            parent.workspace,
            parent.parent_thread_id,
            f"{prof.kind}-{sid}",
            messages or [],
            sub_thread=sub_thread,
        )
    except Exception:
        pass

    return SubagentResult(
        sub_thread=sub_thread,
        kind=prof.kind,
        sub_id=sid,
        summary=summary,
        status=status,
        files_changed=files_changed,
        raw_text=(raw or "").strip(),
        meta={"title": label},
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从正文提取首个 JSON object（供适配层使用）。"""
    raw = (text or "").strip()
    if not raw:
        return None
    m = _JSON_FENCE_RE.search(raw)
    candidate = (m.group(1) if m else raw).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
