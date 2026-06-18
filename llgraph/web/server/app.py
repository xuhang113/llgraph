"""Bundled Web UI 本地 HTTP 适配层（非公开 API；集成请用 ``llgraph.console``）。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llgraph.commands.meta_commands import handle_meta_command, is_registered_meta_command
from llgraph.core.session_bootstrap import AgentRuntimeBundle, build_agent_session_for_thread
from llgraph.terminal.output import capture_terminal_output, format_captured_output
from llgraph.console.discovery import (
    build_session_tree,
    discover_workspaces,
    dismiss_workspace_from_recent,
    touch_workspace_opened,
    list_edits,
    load_plan_detail,
    load_worker_detail,
    read_jsonl_lines,
    register_workspace_path,
    simplify_message,
    workspace_path_from_slug,
    workspace_plans_payload,
    workspace_sessions_payload,
)
from llgraph.session.session_meta import load_session_meta
from llgraph.session.user_storage import session_messages_path, session_thread_dir
from llgraph.console.runtime.agent_service import AgentChatRequest, create_agent_session, start_agent_chat_async
from llgraph.console.runtime.capabilities import load_capabilities
from llgraph.console.runtime.event_hub import HUB
from llgraph.console.runtime.plan_service import (
    check_plan_task_runnable,
    confirm_plan,
    continue_plan,
    create_plan_session,
    discuss_plan,
    get_plan_status,
    run_plan_task,
    start_plan_with_goal,
)
from llgraph.console.runtime.session_lock import LOCKS
from llgraph.console.runtime.sse_utils import format_sse, merge_sse_streams
from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER

app = FastAPI(title="llgraph-internal-ui", version="0.3.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterWorkspaceBody(BaseModel):
    """注册工作区。"""

    path: str


class PickDirectoryBody(BaseModel):
    """本机目录选择。"""

    initial_path: str = ""


class CreateSessionBody(BaseModel):
    """创建会话。"""

    kind: str = Field(description="agent | plan")
    title: str = ""
    goal: str = ""


class ChatBody(BaseModel):
    """发送消息。"""

    message: str
    allow_write: bool = False


class TraceModeBody(BaseModel):
    """Trace 模式。"""

    mode: str


class PlanConfirmBody(BaseModel):
    """Plan 确认。"""

    action: str = "approve"
    allow_worker_write: bool = False
    revise_note: str = ""


class MetaCommandBody(BaseModel):
    """元命令。"""

    command: str
    allow_write: bool = False
    thread_id: str = ""


class LlmSettingsBody(BaseModel):
    """LLM 模型与 thinking 设置。"""

    model: str | None = None
    thinking_enabled: bool | None = None
    reset_model: bool = False
    reset_thinking: bool = False


class UndoBody(BaseModel):
    """还原文件改动。"""

    target: str = "all"


class PlanUndoBody(BaseModel):
    """Plan / Work 回滚。"""

    target: str = "all"
    task_id: str | None = None


class SurveyFormatBody(BaseModel):
    """问卷答案格式化。"""

    answers: dict[str, str] = Field(default_factory=dict)
    allow_write: bool = False


class SurveyResolveBody(BaseModel):
    """从助手回复解析问卷。"""

    text: str = ""


class ReviewBody(BaseModel):
    """代码评审。"""

    topic: str = ""


class SessionTitleBody(BaseModel):
    """会话标题。"""

    title: str


class BatchDeleteSessionsBody(BaseModel):
    """批量删除会话。"""

    thread_ids: list[str] = Field(default_factory=list)


class WebSearchBody(BaseModel):
    """联网搜索开关。"""

    enabled: bool
    thread_id: str = ""
    allow_write: bool = False


class SandboxBody(BaseModel):
    """OS 沙箱开关。"""

    enabled: bool
    thread_id: str = ""
    allow_write: bool = False


class CompressBody(BaseModel):
    """压缩上下文。"""

    thread_id: str
    allow_write: bool = False


class IndexActionBody(BaseModel):
    """索引操作。"""

    action: str = "status"


class SkillToggleBody(BaseModel):
    """Skill 置顶开关。"""

    active: bool


class RuleToggleBody(BaseModel):
    """Rule 启用开关。"""

    enabled: bool


class CodeSearchBody(BaseModel):
    """代码搜索。"""

    query: str
    top_k: int = 15
    mode: str = "parallel"
    path_prefix: str = "."


def _sandbox_payload(rt) -> dict[str, Any]:
    """序列化沙箱状态。"""
    policy = rt.sandbox_policy
    cli = getattr(rt, "sandbox_cli_enabled", None)
    if policy is None:
        return {
            "active": False,
            "enabled": False,
            "backend": None,
            "mode": "",
            "network": "",
            "cli_override": cli,
        }
    return {
        "active": bool(getattr(policy, "active", False)),
        "enabled": bool(getattr(policy, "enabled", False)),
        "backend": getattr(policy, "backend", None),
        "mode": str(getattr(policy, "mode", "") or ""),
        "network": str(getattr(policy, "network", "") or ""),
        "cli_override": cli,
    }


def _ws(slug: str) -> Path:
    try:
        return workspace_path_from_slug(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _meta_agent_session(
    workspace: Path,
    thread_id: str,
    *,
    allow_write: bool,
) -> Any | None:
    """
    为 Agent 会话构建元命令所需的 AgentSessionContext。

    @param workspace 工作区根
    @param thread_id cli-* thread
    @param allow_write 是否可写
    @return AgentSessionContext；非 Agent 会话则 None
    """
    if not thread_id.strip():
        return None
    meta = load_session_meta(workspace, thread_id.strip())
    if meta.get("session_kind") != "agent":
        return None
    rt = RUNTIME_MANAGER.get(workspace, allow_write=allow_write)
    bundle = AgentRuntimeBundle(
        workspace=workspace,
        trace_session=rt.trace_session,
        context_session=rt.context_session,
        allow_write=allow_write,
        mcp_tools=rt.mcp_tools,
        mcp_registry=rt.mcp_registry,
        watch_service=None,
        web_search_enabled=rt.web_search_enabled,
        sandbox_policy=rt.sandbox_policy,
        sandbox_cli_enabled=rt.sandbox_cli_enabled,
        no_spill=False,
        memory_kind="memory",
        mcp_summary=rt.mcp_summary,
        watch_active=False,
    )
    return build_agent_session_for_thread(bundle, thread_id.strip())


def _session_edit_tracker(workspace: Path, thread_id: str) -> Any:
    """
    加载会话编辑账本（undo / changes，与 allow_write 无关）。

    @param workspace 工作区根
    @param thread_id cli-* thread
    @return SessionEditTracker
    """
    from llgraph.session.session_edits import SessionEditTracker

    return SessionEditTracker(workspace, session_id=thread_id.strip())


@app.get("/api/health")
def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok", "version": "0.2.0"}


@app.get("/api/me")
def me() -> dict[str, str]:
    """当前用户数据根。"""
    from llgraph.session.user_storage import user_context_root

    return {"context_root": str(user_context_root())}


# ── 工作区 ──


@app.get("/api/workspaces")
def list_workspaces() -> dict[str, list]:
    """列举工作区。"""
    from dataclasses import asdict

    return {"workspaces": [asdict(w) for w in discover_workspaces()]}


@app.post("/api/workspaces/register")
def register_workspace(body: RegisterWorkspaceBody) -> dict[str, Any]:
    """注册/打开工作区路径。"""
    try:
        info = register_workspace_path(body.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    from dataclasses import asdict

    return asdict(info)


@app.post("/api/workspaces/pick-directory")
def pick_workspace_directory(body: PickDirectoryBody) -> dict[str, Any]:
    """唤起本机目录选择对话框。"""
    from llgraph.web.directory_picker import pick_directory

    path = pick_directory(initial_path=body.initial_path)
    if not path:
        return {"path": None, "cancelled": True}
    return {"path": path, "cancelled": False}


@app.delete("/api/workspaces/{slug}/recent")
def dismiss_workspace_recent(slug: str) -> dict[str, Any]:
    """从最近工作区列表移除（不删除 Agent/Plan 会话数据）。"""
    try:
        dismiss_workspace_from_recent(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "slug": slug, "message": "已从最近工作区移除"}


@app.post("/api/workspaces/{slug}/touch")
def touch_workspace(slug: str) -> dict[str, Any]:
    """记录工作区最近打开时间（最近列表排序）。"""
    try:
        touch_workspace_opened(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "slug": slug}


@app.get("/api/workspaces/{slug}")
def get_workspace(slug: str) -> dict[str, Any]:
    """工作区详情 + 会话树。"""
    workspace = _ws(slug)
    tree = build_session_tree(workspace)
    lock_info = []
    return {
        "slug": slug,
        "path": str(workspace),
        "tree": tree,
        "locks": lock_info,
    }


@app.get("/api/workspaces/{slug}/tree")
def get_session_tree(slug: str) -> dict[str, Any]:
    """会话树。"""
    return build_session_tree(_ws(slug))


@app.get("/api/workspaces/{slug}/capabilities")
def get_capabilities(slug: str, allow_write: bool = False) -> dict[str, Any]:
    """工具 / MCP / Skill 清单。"""
    payload = load_capabilities(_ws(slug), allow_write=allow_write)
    rt = RUNTIME_MANAGER.get(_ws(slug), allow_write=allow_write)
    payload["sandbox"] = _sandbox_payload(rt)
    return payload


def _resolve_catalog_skill(workspace: Path, name: str):
    from llgraph.loaders.skills_loader import discover_skills

    key = name.strip().lower()
    if not key:
        return None
    for skill in discover_skills(workspace):
        if skill.name.lower() == key:
            return skill
    return None


def _resolve_catalog_rule(workspace: Path, rule_id: str):
    from pathlib import Path as PathLib

    from llgraph.loaders.rules_loader import discover_rules

    key = rule_id.strip().lower()
    if not key:
        return None
    rules = discover_rules(workspace)
    for rule in rules:
        if rule.rule_id.lower() == key:
            return rule
    basename = PathLib(key).name.lower()
    if basename and basename != key:
        for rule in rules:
            if PathLib(rule.rule_id).name.lower() == basename:
                return rule
    return None


@app.get("/api/workspaces/{slug}/catalog/skill/{name}")
def get_skill_catalog_detail(slug: str, name: str) -> dict[str, Any]:
    """Skill 正文（Web 目录详情）。"""
    from llgraph.config.catalog_paths import format_catalog_path, scope_label

    workspace = _ws(slug)
    skill = _resolve_catalog_skill(workspace, name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {name}")
    skill_file = skill.skill_dir / "SKILL.md"
    return {
        "name": skill.name,
        "description": skill.description,
        "scope": skill.scope,
        "scope_label": scope_label(skill.scope),
        "path": format_catalog_path(workspace, skill_file, skill.scope),
        "body": skill.body,
    }


@app.get("/api/workspaces/{slug}/catalog/rule/{rule_id:path}")
def get_rule_catalog_detail(slug: str, rule_id: str) -> dict[str, Any]:
    """Rule 正文（Web 目录详情）。"""
    from llgraph.config.catalog_paths import format_catalog_path, scope_label

    workspace = _ws(slug)
    rule = _resolve_catalog_rule(workspace, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule 不存在: {rule_id}")
    return {
        "id": rule.rule_id,
        "description": rule.description,
        "scope": rule.scope,
        "scope_label": scope_label(rule.scope),
        "path": format_catalog_path(workspace, rule.source_path, rule.scope),
        "body": rule.body,
        "always_apply": rule.always_apply,
        "globs": rule.globs,
    }


@app.post("/api/workspaces/{slug}/trace-mode")
def set_trace_mode(slug: str, body: TraceModeBody) -> dict[str, str]:
    """设置 trace 模式。"""
    mode = RUNTIME_MANAGER.set_trace_mode(_ws(slug), body.mode)
    return {"mode": mode.value}


@app.get("/api/workspaces/{slug}/slash-catalog")
def get_slash_catalog(slug: str) -> dict[str, Any]:
    """斜杠命令补全目录（Skills / Commands / 内置，与终端一致）。"""
    from llgraph.terminal.slash_catalog import build_slash_catalog, slash_category_badge

    workspace = _ws(slug)
    items = build_slash_catalog(workspace)
    return {
        "items": [
            {
                "name": item.name,
                "description": item.description,
                "category": item.category,
                "badge": slash_category_badge(item.category),
                "insert_text": item.insert_text,
                "origin": item.origin,
            }
            for item in items
        ],
    }


@app.get("/api/workspaces/{slug}/llm-settings")
def get_llm_settings(slug: str) -> dict[str, Any]:
    """当前模型与 thinking 配置。"""
    from llgraph.console.runtime.llm_settings_api import build_llm_settings_payload

    return build_llm_settings_payload(_ws(slug))


@app.post("/api/workspaces/{slug}/llm-settings")
def post_llm_settings(slug: str, body: LlmSettingsBody) -> dict[str, Any]:
    """切换模型或 thinking 开关。"""
    from llgraph.console.runtime.llm_settings_api import apply_llm_settings

    return apply_llm_settings(
        _ws(slug),
        model=body.model,
        thinking_enabled=body.thinking_enabled,
        reset_model=body.reset_model,
        reset_thinking=body.reset_thinking,
    )


@app.post("/api/workspaces/{slug}/meta")
def run_meta_command(slug: str, body: MetaCommandBody) -> dict[str, Any]:
    """执行元命令（/trace、/skill、/model 等），返回终端同等文本输出。"""
    workspace = _ws(slug)
    command = body.command.strip()
    rt = RUNTIME_MANAGER.get(workspace, allow_write=body.allow_write)
    trace_mode = rt.trace_session.mode.value

    if not command.startswith("/"):
        return {
            "handled": False,
            "registered": False,
            "output": "",
            "trace_mode": trace_mode,
        }

    if not is_registered_meta_command(command, workspace):
        return {
            "handled": False,
            "registered": False,
            "output": "",
            "trace_mode": trace_mode,
        }

    agent_session = _meta_agent_session(
        workspace,
        body.thread_id,
        allow_write=body.allow_write,
    )
    edit_tracker = agent_session.edit_tracker if agent_session is not None else None

    with capture_terminal_output() as buf:
        handled = handle_meta_command(
            command,
            workspace=workspace,
            trace_session=rt.trace_session,
            context_session=rt.context_session,
            allow_write=body.allow_write,
            edit_tracker=edit_tracker,
            agent_session=agent_session,
            mcp_summary=rt.mcp_summary,
        )

    output = format_captured_output(buf)
    if not handled and not output:
        preview = command.split("\n", 1)[0]
        output = f"未知命令 {preview}，输入 /help 查看。"

    return {
        "handled": handled or bool(output),
        "registered": True,
        "output": output,
        "trace_mode": rt.trace_session.mode.value,
    }


@app.post("/api/workspaces/{slug}/web-search")
def set_web_search(slug: str, body: WebSearchBody) -> dict[str, Any]:
    """切换联网搜索（工作区 Runtime；可选同步当前 Agent 会话）。"""
    workspace = _ws(slug)
    try:
        RUNTIME_MANAGER.set_web_search_enabled(workspace, enabled=body.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    message = "已启用 Web 搜索。" if body.enabled else "已禁用 Web 搜索。"
    if body.thread_id.strip():
        agent_session = _meta_agent_session(
            workspace,
            body.thread_id,
            allow_write=body.allow_write,
        )
        if agent_session is not None:
            from llgraph.session.session_web_search import set_session_web_search_mode

            _, message = set_session_web_search_mode(agent_session, enabled=body.enabled)

    return {"enabled": body.enabled, "message": message}


@app.post("/api/workspaces/{slug}/sandbox")
def set_sandbox(slug: str, body: SandboxBody) -> dict[str, Any]:
    """切换 OS 沙箱（工作区 Runtime；可选同步当前 Agent 会话）。"""
    workspace = _ws(slug)
    try:
        RUNTIME_MANAGER.set_sandbox_enabled(workspace, enabled=body.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rt = RUNTIME_MANAGER.get(workspace, allow_write=body.allow_write)
    sandbox = _sandbox_payload(rt)
    message = (
        f"已启用 OS 沙箱（{sandbox.get('backend') or 'unknown'}，mode={sandbox.get('mode', '')}）。"
        if sandbox.get("enabled")
        else "已禁用 OS 沙箱。"
    )
    if body.thread_id.strip():
        agent_session = _meta_agent_session(
            workspace,
            body.thread_id,
            allow_write=body.allow_write,
        )
        if agent_session is not None:
            from llgraph.session.session_sandbox import set_session_sandbox_mode

            _, message = set_session_sandbox_mode(agent_session, enabled=body.enabled)

    return {"sandbox": sandbox, "message": message}


@app.get("/api/workspaces/{slug}/context")
def get_context_usage(
    slug: str,
    allow_write: bool = False,
    thread_id: str = "",
) -> dict[str, Any]:
    """上下文 token 用量（结构化，对标 /context）。"""
    workspace = _ws(slug)
    rt = RUNTIME_MANAGER.get(workspace, allow_write=allow_write)
    agent_session = _meta_agent_session(
        workspace,
        thread_id,
        allow_write=allow_write,
    )
    from llgraph.context.context_settings import resolve_context_settings
    from llgraph.context.context_stats import collect_context_usage
    from llgraph.core.model_context_window import format_context_budget_note

    web_enabled = (
        agent_session.web_search_enabled
        if agent_session is not None
        else rt.web_search_enabled
    )
    breakdown = collect_context_usage(
        workspace,
        context_session=rt.context_session,
        allow_write=allow_write,
        web_search_enabled=web_enabled,
        agent_session=agent_session,
    )
    settings = resolve_context_settings(workspace)
    limit = settings.max_tokens_estimate
    total = breakdown.total
    ratio = total / limit if limit > 0 else 0.0
    return {
        "total": total,
        "limit": limit,
        "ratio": ratio,
        "pct": min(100, int(ratio * 100)),
        "message_count": breakdown.message_count,
        "tool_count": breakdown.tool_count,
        "mcp_tool_count": breakdown.mcp_tool_count,
        "breakdown": {
            "system_prompt": breakdown.system_prompt,
            "tool_definitions": breakdown.tool_definitions,
            "rules": breakdown.rules,
            "skills": breakdown.skills,
            "mcp": breakdown.mcp,
            "markdowns_index": breakdown.markdowns_index,
            "summarized_conversation": breakdown.summarized_conversation,
            "conversation": breakdown.conversation,
        },
        "budget_note": format_context_budget_note(
            workspace,
            max_tokens=settings.max_tokens_estimate,
            source=settings.budget_source,
            model_id=settings.context_model_id,
            ratio=settings.auto_compress_ratio,
        ),
        "has_session": agent_session is not None,
    }


@app.post("/api/workspaces/{slug}/compress")
def compress_session_context(slug: str, body: CompressBody) -> dict[str, Any]:
    """压缩 Agent 会话历史（对标 /compress）。"""
    workspace = _ws(slug)
    agent_session = _meta_agent_session(
        workspace,
        body.thread_id,
        allow_write=body.allow_write,
    )
    if agent_session is None or not agent_session.with_memory:
        raise HTTPException(
            status_code=400,
            detail="需要 Agent 会话且有多轮对话历史",
        )

    from llgraph.context.context_compressor import (
        apply_compress_to_agent_state,
        format_compress_report,
    )
    from llgraph.context.context_settings import is_auto_compress_strategy, resolve_context_settings
    from llgraph.display.execution_log import log_compress_event
    from llgraph.session.session_manifest import sync_session_manifest_to_agent_state

    settings = resolve_context_settings(workspace)
    preserve = False if is_auto_compress_strategy(settings.compress_strategy) else None
    report = apply_compress_to_agent_state(
        agent_session.agent,
        thread_id=agent_session.thread_id,
        workspace=workspace,
        force=True,
        preserve_current_turn=preserve,
    )
    if report is None:
        return {"ok": True, "compressed": False, "message": "无需压缩或消息为空。"}

    log_compress_event(
        workspace,
        thread_id=agent_session.thread_id,
        report=report,
        trigger="manual",
    )
    if agent_session.context_session is not None:
        sync_session_manifest_to_agent_state(
            agent_session.agent,
            thread_id=agent_session.thread_id,
            workspace=workspace,
            session=agent_session.context_session,
            user_message="",
            with_memory=True,
            archive_path=report.archive_path,
            allow_write=agent_session.allow_write,
        )
    return {
        "ok": True,
        "compressed": True,
        "message": format_compress_report(report),
        "archive_path": report.archive_path,
    }


@app.get("/api/workspaces/{slug}/index-status")
def get_index_status_api(slug: str) -> dict[str, Any]:
    """代码索引状态（对标 /index status）。"""
    workspace = _ws(slug)
    from llgraph.code_index.embedder import format_embedding_status
    from llgraph.code_index.index_settings import resolve_index_settings
    from llgraph.code_index.manifest import load_manifest
    from llgraph.code_index.paths import meta_path
    from llgraph.code_index.store import get_index_status

    status = get_index_status(workspace)
    idx_cfg = resolve_index_settings(workspace)
    manifest = load_manifest(workspace)
    sync_complete = None
    meta_file = meta_path(workspace)
    if meta_file.is_file():
        import json

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            sync_complete = meta.get("sync_complete")
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "exists": status.exists,
        "chunk_count": status.chunk_count,
        "vector_dim": status.vector_dim,
        "last_indexed_at": status.last_indexed_at,
        "lance_path": status.lance_path,
        "manifest_files": len(manifest),
        "sync_complete": sync_complete,
        "watch_enabled": idx_cfg.watch_enabled,
        "watch_with_agent": idx_cfg.watch_with_agent,
        "embedding": format_embedding_status(workspace),
        "max_files": idx_cfg.max_files,
    }


@app.post("/api/workspaces/{slug}/index")
def run_index_action(slug: str, body: IndexActionBody) -> dict[str, Any]:
    """执行索引操作（incremental / full / rebuild / dry-run）。"""
    workspace = _ws(slug)
    from llgraph.code_index.index_dispatch import dispatch_index

    action = body.action.strip().lower()
    argv_map = {
        "status": ["status"],
        "full": ["full"],
        "incremental": ["incremental"],
        "rebuild": ["rebuild"],
        "dry-run": ["dry-run"],
    }
    argv = argv_map.get(action)
    if argv is None:
        raise HTTPException(status_code=400, detail=f"未知操作: {body.action}")

    result = dispatch_index(
        workspace,
        argv,
        prog="/index",
        bare_means_status=False,
    )
    return {
        "ok": result.exit_code == 0,
        "exit_code": result.exit_code,
        "action": action,
        "log_path": str(result.log_path) if result.log_path else None,
    }


@app.post("/api/workspaces/{slug}/catalog/skill/{name}/toggle")
def toggle_skill(slug: str, name: str, body: SkillToggleBody) -> dict[str, Any]:
    """置顶或关闭 Skill（对标 /skill）。"""
    workspace = _ws(slug)
    skill = _resolve_catalog_skill(workspace, name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {name}")

    rt = RUNTIME_MANAGER.get(workspace)
    if body.active:
        rt.context_session.activate_skill(skill.name)
        message = f"已置顶技能: {skill.name}"
    else:
        if not rt.context_session.deactivate_skill(skill.name):
            message = f"技能未启用: {skill.name}"
        else:
            message = f"已关闭技能: {skill.name}"
    return {
        "ok": True,
        "name": skill.name,
        "active": skill.name.lower()
        in {s.lower() for s in rt.context_session.active_skills},
        "active_skills": list(rt.context_session.active_skills),
        "message": message,
    }


@app.post("/api/workspaces/{slug}/catalog/rule/{rule_id:path}/toggle")
def toggle_rule(slug: str, rule_id: str, body: RuleToggleBody) -> dict[str, Any]:
    """强制启用或禁用 Rule（对标 /rule on|off）。"""
    workspace = _ws(slug)
    rule = _resolve_catalog_rule(workspace, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule 不存在: {rule_id}")

    rt = RUNTIME_MANAGER.get(workspace)
    rid = rule.rule_id
    if body.enabled:
        rt.context_session.disabled_rules.discard(rid)
        rt.context_session.forced_rules.add(rid)
        message = f"已强制启用: {rid}"
    else:
        rt.context_session.forced_rules.discard(rid)
        rt.context_session.disabled_rules.add(rid)
        message = f"已禁用: {rid}"
    return {
        "ok": True,
        "id": rid,
        "forced": rid in rt.context_session.forced_rules,
        "disabled": rid in rt.context_session.disabled_rules,
        "message": message,
    }


@app.post("/api/workspaces/{slug}/sessions/delete-empty")
def delete_empty_sessions(slug: str) -> dict[str, Any]:
    """删除空壳会话（对标 /session delete empty）。"""
    workspace = _ws(slug)
    from llgraph.session.session_delete import delete_sessions
    from llgraph.session.session_registry import list_empty_session_ids

    empty_ids = list_empty_session_ids(workspace)
    if not empty_ids:
        return {"ok": True, "deleted": 0, "message": "无空壳会话"}
    report = delete_sessions(workspace, empty_ids)
    return {
        "ok": report.failure_count == 0,
        "deleted": report.success_count,
        "failed": report.failure_count,
        "message": f"已删除 {report.success_count} 个空壳会话",
    }


@app.post("/api/workspaces/{slug}/code-search")
def code_search(slug: str, body: CodeSearchBody) -> dict[str, Any]:
    """并行/语义代码搜索（对标 llgraph search）。"""
    workspace = _ws(slug)
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    from llgraph.code_index.parallel_search import search_parallel
    from llgraph.code_index.search import search_semantic

    mode = body.mode.strip().lower()
    top_k = max(1, min(body.top_k, 50))
    if mode == "semantic":
        text = search_semantic(
            workspace,
            query,
            top_k=top_k,
            path_prefix=body.path_prefix,
            source="web",
            tool="search_code_semantic",
        )
    else:
        text = search_parallel(
            workspace,
            query,
            top_k=top_k,
            path_prefix=body.path_prefix,
            source="web",
            tool="search_code_parallel",
        )
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {
        "query": query,
        "mode": mode if mode == "semantic" else "parallel",
        "top_k": top_k,
        "text": text,
        "lines": lines,
        "count": len(lines),
    }


@app.get("/api/workspaces/{slug}/execution-log")
def get_execution_log(slug: str, limit: int = 30) -> dict[str, Any]:
    """执行日志尾部（对标 /log tail）。"""
    workspace = _ws(slug)
    from llgraph.display.execution_log import (
        execution_log_path,
        format_execution_record,
        read_execution_tail,
    )

    cap = max(1, min(limit, 200))
    records = read_execution_tail(workspace, limit=cap)
    path = execution_log_path(workspace)
    return {
        "path": str(path),
        "records": records,
        "lines": [format_execution_record(r) for r in records],
        "count": len(records),
    }


@app.post("/api/workspaces/{slug}/execution-log/purge")
def purge_execution_log(slug: str) -> dict[str, Any]:
    """清理过期日志（对标 /log purge）。"""
    workspace = _ws(slug)
    from llgraph.display.log_retention import format_purge_report, run_log_retention

    report = run_log_retention(workspace, quiet=False)
    return {
        "ok": True,
        "message": format_purge_report(report),
        "report": report,
    }


# ── 只读（保留） ──


@app.get("/api/workspaces/{slug}/sessions")
def list_sessions(slug: str) -> dict:
    """Agent 会话列表。"""
    return workspace_sessions_payload(_ws(slug))


@app.get("/api/workspaces/{slug}/plans")
def list_plans(slug: str) -> dict:
    """Plan 列表。"""
    return workspace_plans_payload(_ws(slug))


@app.get("/api/workspaces/{slug}/plans/{thread_id}")
def get_plan(slug: str, thread_id: str) -> dict:
    """Plan 详情。"""
    workspace = _ws(slug)
    if not session_thread_dir(workspace, thread_id).is_dir():
        raise HTTPException(status_code=404, detail=f"Plan 不存在: {thread_id}")
    detail = load_plan_detail(workspace, thread_id)
    detail["job"] = get_plan_status(thread_id)
    lock = LOCKS.get(thread_id)
    detail["lock"] = {"owner": lock.owner, "since": lock.since} if lock else None
    return detail


@app.get("/api/workspaces/{slug}/plans/{thread_id}/tasks/{task_id}")
def get_worker(slug: str, thread_id: str, task_id: str) -> dict:
    """Worker 详情。"""
    return load_worker_detail(_ws(slug), thread_id, task_id)


@app.get("/api/workspaces/{slug}/sessions/{thread_id}")
def get_session(slug: str, thread_id: str) -> dict:
    """会话元数据。"""
    workspace = _ws(slug)
    meta = load_session_meta(workspace, thread_id)
    _, total = read_jsonl_lines(session_messages_path(workspace, thread_id), offset=0, limit=0)
    lock = LOCKS.get(thread_id)
    from llgraph.session.session_meta import resolve_session_display_title

    return {
        "thread_id": thread_id,
        "meta": meta,
        "title": resolve_session_display_title(workspace, thread_id),
        "message_total": total,
        "lock": {"owner": lock.owner, "since": lock.since} if lock else None,
    }


@app.patch("/api/workspaces/{slug}/sessions/{thread_id}/title")
def patch_session_title(slug: str, thread_id: str, body: SessionTitleBody) -> dict[str, Any]:
    """重命名 Agent / Plan 会话标题。"""
    from llgraph.console.runtime.session_title_api import update_session_display_title

    ok, msg, normalized = update_session_display_title(
        _ws(slug),
        thread_id,
        body.title,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "title": normalized, "message": msg}


@app.get("/api/workspaces/{slug}/sessions/{thread_id}/messages")
def get_session_messages(
    slug: str,
    thread_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """分页 messages。"""
    msg_path = session_messages_path(_ws(slug), thread_id)
    rows, total = read_jsonl_lines(msg_path, offset=offset, limit=limit)
    return {
        "messages": [simplify_message(r) for r in rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/workspaces/{slug}/sessions/{thread_id}/last-trace")
def get_session_last_trace(slug: str, thread_id: str) -> dict:
    """Web 右侧 Trace 面板：最后一轮逐步日志。"""
    from llgraph.session.web_trace_store import load_last_web_trace

    data = load_last_web_trace(_ws(slug), thread_id)
    if not data:
        return {"log_lines": [], "steps": []}
    log_lines = data.get("log_lines")
    steps = data.get("steps")
    return {
        "log_lines": log_lines if isinstance(log_lines, list) else [],
        "steps": steps if isinstance(steps, list) else [],
    }


@app.get("/api/workspaces/{slug}/sessions/{thread_id}/edits")
def get_session_edits(slug: str, thread_id: str) -> dict:
    """edits.jsonl。"""
    return {"edits": list_edits(_ws(slug), thread_id)}


@app.get("/api/workspaces/{slug}/sessions/{thread_id}/file-changes")
def get_session_file_changes(slug: str, thread_id: str) -> dict[str, Any]:
    """本会话可还原文件摘要（Undo All UI）。"""
    tracker = _session_edit_tracker(_ws(slug), thread_id)
    return tracker.web_changes_payload()


@app.get("/api/workspaces/{slug}/sessions/{thread_id}/diff")
def get_session_diff(slug: str, thread_id: str, path: str = Query(..., min_length=1)) -> dict[str, str]:
    """单文件 diff（会话快照 vs 当前磁盘）。"""
    from llgraph.console.edit_service import session_diff_text

    return {"path": path.strip(), "diff": session_diff_text(_ws(slug), thread_id, path)}


@app.post("/api/workspaces/{slug}/sessions/{thread_id}/undo")
def undo_session_files(slug: str, thread_id: str, body: UndoBody) -> dict[str, Any]:
    """还原单个文件或全部改动（等同 /undo）。"""
    from llgraph.console.edit_service import undo_session_files as undo_files

    target = body.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="target 不能为空")
    try:
        return undo_files(_ws(slug), thread_id, target=target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/workspaces/{slug}/sessions/{thread_id}/review")
def review_session_files(slug: str, thread_id: str, body: ReviewBody) -> dict[str, Any]:
    """对会话改动执行 /review。"""
    from llgraph.console.edit_service import run_session_review

    return run_session_review(_ws(slug), thread_id, topic=body.topic.strip())


@app.get("/api/workspaces/{slug}/plans/{thread_id}/file-changes")
def get_plan_file_changes(slug: str, thread_id: str) -> dict[str, Any]:
    """聚合 Plan 各 Work 的文件改动。"""
    from llgraph.console.edit_service import plan_file_changes

    return plan_file_changes(_ws(slug), thread_id)


@app.post("/api/workspaces/{slug}/plans/{thread_id}/undo")
def undo_plan_files(slug: str, thread_id: str, body: PlanUndoBody) -> dict[str, Any]:
    """Plan 整体或单 Work 回滚（等同放弃对应产出物）。"""
    from llgraph.console.edit_service import plan_undo_files

    target = body.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="target 不能为空")
    try:
        return plan_undo_files(
            _ws(slug),
            thread_id,
            target=target,
            task_id=body.task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/workspaces/{slug}/plans/{thread_id}/review")
def review_plan_files(slug: str, thread_id: str, body: ReviewBody) -> dict[str, Any]:
    """对 Plan 各 Work 改动执行 /review。"""
    from llgraph.console.edit_service import plan_run_review

    return plan_run_review(_ws(slug), thread_id, topic=body.topic.strip())


@app.post("/api/workspaces/{slug}/survey/format")
def format_survey_answers(body: SurveyFormatBody) -> dict[str, str]:
    """将问卷答案格式化为 Agent 用户消息。"""
    from llgraph.survey.survey_prompt import format_survey_answers_for_agent

    message = format_survey_answers_for_agent(
        body.answers,
        allow_write=body.allow_write,
    )
    return {"message": message}


@app.post("/api/workspaces/{slug}/survey/resolve")
def resolve_survey_from_text(body: SurveyResolveBody) -> dict[str, Any]:
    """从助手回复解析 survey（JSON 块或 Markdown 编号列表）。"""
    from llgraph.console.runtime.agent_service import _survey_spec_to_dict
    from llgraph.survey.survey_prompt import resolve_survey_from_assistant

    text = body.text.strip()
    if not text:
        return {"survey": None}
    spec = resolve_survey_from_assistant(text)
    if spec is None:
        return {"survey": None}
    return {"survey": _survey_spec_to_dict(spec)}


# ── 交互：创建会话 ──


@app.post("/api/workspaces/{slug}/sessions/create")
def create_session(slug: str, body: CreateSessionBody) -> dict[str, str]:
    """新建 Agent 或 Plan 会话。"""
    workspace = _ws(slug)
    kind = body.kind.strip().lower()
    if kind == "agent":
        thread_id = create_agent_session(workspace, title=body.title)
        return {"thread_id": thread_id, "kind": "agent"}
    if kind == "plan":
        thread_id = create_plan_session(workspace, goal=body.goal)
        return {"thread_id": thread_id, "kind": "plan", "goal": body.goal}
    raise HTTPException(status_code=400, detail="kind 须为 agent 或 plan")


@app.delete("/api/workspaces/{slug}/sessions/{thread_id}")
def delete_session_endpoint(slug: str, thread_id: str) -> dict[str, Any]:
    """删除 Agent 或 Plan 会话（Plan 含 Worker 级联，委托 llgraph）。"""
    from llgraph.console.session_service import delete_session_for_web

    workspace = _ws(slug)
    if LOCKS.get(thread_id) is not None:
        raise HTTPException(status_code=409, detail="会话正在使用中，请稍后再试")
    result = delete_session_for_web(str(workspace), thread_id)
    if not result.get("ok"):
        status = 409 if result.get("error") and "正在执行" in str(result.get("error")) else 400
        raise HTTPException(status_code=status, detail=result.get("error") or "删除失败")
    return result


@app.post("/api/workspaces/{slug}/sessions/batch-delete")
def batch_delete_sessions_endpoint(slug: str, body: BatchDeleteSessionsBody) -> dict[str, Any]:
    """批量删除 Agent / Plan 会话。"""
    from llgraph.console.session_service import delete_sessions_for_web

    workspace = _ws(slug)
    ids = [tid.strip() for tid in body.thread_ids if tid and tid.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="thread_ids 不能为空")
    locked = [tid for tid in ids if LOCKS.get(tid) is not None]
    if locked:
        raise HTTPException(
            status_code=409,
            detail=f"以下会话正在使用中: {', '.join(locked[:5])}",
        )
    return delete_sessions_for_web(str(workspace), ids)


# ── 交互：Agent SSE Chat ──


@app.post("/api/workspaces/{slug}/sessions/{thread_id}/chat")
async def agent_chat_stream(slug: str, thread_id: str, body: ChatBody) -> StreamingResponse:
    """Agent 对话 SSE 流。"""
    workspace = _ws(slug)
    loop = asyncio.get_event_loop()
    channel = f"agent:{thread_id}"
    queue = HUB.subscribe(channel)

    req = AgentChatRequest(
        workspace=workspace,
        thread_id=thread_id,
        message=body.message,
        allow_write=body.allow_write,
        channel=channel,
    )
    start_agent_chat_async(req, loop, queue)

    async def gen():
        try:
            async for chunk in merge_sse_streams(queue):
                yield chunk
        finally:
            HUB.unsubscribe(channel, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── 交互：Plan ──


@app.post("/api/workspaces/{slug}/plans/{thread_id}/start")
async def plan_start_stream(
    slug: str,
    thread_id: str,
    body: ChatBody,
) -> StreamingResponse:
    """启动/续跑 Plan（goal 或空继续）。"""
    workspace = _ws(slug)
    loop = asyncio.get_event_loop()
    channel = f"plan:{thread_id}"
    queue = HUB.subscribe(channel)

    if body.message.strip():
        start_plan_with_goal(
            workspace,
            thread_id,
            body.message.strip(),
            allow_write=body.allow_write,
            channel=channel,
            loop=loop,
            block_first=False,
        )
    else:
        from llgraph.console.runtime.plan_service import continue_plan

        continue_plan(
            workspace,
            thread_id,
            allow_write=body.allow_write,
            channel=channel,
            loop=loop,
        )

    async def gen():
        try:
            async for chunk in merge_sse_streams(queue):
                yield chunk
        finally:
            HUB.unsubscribe(channel, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/workspaces/{slug}/plans/{thread_id}/confirm")
async def plan_confirm_stream(
    slug: str,
    thread_id: str,
    body: PlanConfirmBody,
) -> StreamingResponse:
    """Plan 确认 Survey 决策。"""
    workspace = _ws(slug)
    loop = asyncio.get_event_loop()
    channel = f"plan:{thread_id}"
    queue = HUB.subscribe(channel)
    decision = {
        "action": body.action,
        "allow_worker_write": body.allow_worker_write,
        "revise_note": body.revise_note,
    }
    confirm_plan(
        workspace,
        thread_id,
        decision,
        allow_write=body.allow_worker_write,
        channel=channel,
        loop=loop,
    )

    async def gen():
        try:
            async for chunk in merge_sse_streams(queue):
                yield chunk
        finally:
            HUB.unsubscribe(channel, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/workspaces/{slug}/plans/{thread_id}/continue")
async def plan_continue_stream(
    slug: str,
    thread_id: str,
    body: ChatBody,
) -> StreamingResponse:
    """Plan task_step_confirm 后继续。"""
    workspace = _ws(slug)
    loop = asyncio.get_event_loop()
    channel = f"plan:{thread_id}"
    queue = HUB.subscribe(channel)
    continue_plan(
        workspace,
        thread_id,
        allow_write=body.allow_write,
        channel=channel,
        loop=loop,
    )

    async def gen():
        try:
            async for chunk in merge_sse_streams(queue):
                yield chunk
        finally:
            HUB.unsubscribe(channel, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/workspaces/{slug}/plans/{thread_id}/discuss")
async def plan_discuss_stream(
    slug: str,
    thread_id: str,
    body: ChatBody,
) -> StreamingResponse:
    """Plan 终止后基于最终报告问答。"""
    workspace = _ws(slug)
    loop = asyncio.get_event_loop()
    channel = f"plan:{thread_id}"
    queue = HUB.subscribe(channel)
    discuss_plan(
        workspace,
        thread_id,
        body.message.strip(),
        channel=channel,
        loop=loop,
    )

    async def gen():
        try:
            async for chunk in merge_sse_streams(queue):
                yield chunk
        finally:
            HUB.unsubscribe(channel, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/workspaces/{slug}/plans/{thread_id}/tasks/{task_id}/runnable")
def plan_task_runnable(slug: str, thread_id: str, task_id: str) -> dict[str, Any]:
    """检查 Work task 是否可执行（依赖是否满足）。"""
    return check_plan_task_runnable(_ws(slug), thread_id, task_id)


@app.post("/api/workspaces/{slug}/plans/{thread_id}/tasks/{task_id}/run")
async def plan_run_task_stream(
    slug: str,
    thread_id: str,
    task_id: str,
    body: ChatBody,
) -> StreamingResponse:
    """手动执行单个 Work task。"""
    workspace = _ws(slug)
    loop = asyncio.get_event_loop()
    channel = f"plan:{thread_id}"
    queue = HUB.subscribe(channel)
    run_plan_task(
        workspace,
        thread_id,
        task_id,
        allow_write=body.allow_write,
        channel=channel,
        loop=loop,
    )

    async def gen():
        try:
            async for chunk in merge_sse_streams(queue):
                yield chunk
        finally:
            HUB.unsubscribe(channel, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/workspaces/{slug}/plans/{thread_id}/events")
async def plan_events_subscribe(slug: str, thread_id: str) -> StreamingResponse:
    """订阅 Plan trace / 状态事件（长连接）。"""
    channel = f"plan:{thread_id}"
    queue = HUB.subscribe(channel)

    async def gen():
        try:
            yield format_sse({"type": "subscribed", "channel": channel})
            detail = load_plan_detail(_ws(slug), thread_id)
            yield format_sse({"type": "plan_state", "phase": detail.get("phase")})
            async for chunk in merge_sse_streams(queue, timeout_sec=86400):
                yield chunk
        finally:
            HUB.unsubscribe(channel, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/workspaces/{slug}/plans/{thread_id}/job")
def plan_job_status(slug: str, thread_id: str) -> dict:
    """Plan 后台 job 状态。"""
    return get_plan_status(thread_id)


# ── 静态资源 ──


def _repo_root() -> Path:
    """仓库根目录（含 web-ui/）。"""
    return Path(__file__).resolve().parents[3]


def _static_dir() -> Path | None:
    env = os.environ.get("LLGRAPH_WEB_STATIC", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_dir():
            return path
    pkg = _repo_root() / "web-ui" / "dist"
    if pkg.is_dir():
        return pkg
    return None


_static = _static_dir()
if _static is not None:
    app.mount("/assets", StaticFiles(directory=_static / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        """SPA 回退。"""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        index = _static / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(index)
