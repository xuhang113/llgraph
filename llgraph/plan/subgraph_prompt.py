"""Plan 子 Agent 系统提示（复用 Agent skills/rules 与工具规范）。"""

from __future__ import annotations

from llgraph.core.agent import build_system_prompt
from llgraph.plan.runtime import PlanRuntimeContext


def build_planner_role_block(ctx: PlanRuntimeContext) -> str:
    """
    Planner 角色说明块。

    @param ctx Plan 运行时上下文
    @return 角色追加文本
    """
    from llgraph.loaders.prompt_loader import compose_plan_planner_role

    return compose_plan_planner_role(workspace=ctx.workspace)


def build_worker_role_block(
    ctx: PlanRuntimeContext,
    task: dict,
    *,
    allow_write: bool,
) -> str:
    """
    Worker 角色说明块。

    @param ctx Plan 运行时上下文
    @param task task 定义
    @param allow_write 是否允许写文件
    @return 角色追加文本
    """
    scope = task.get("scope") if isinstance(task.get("scope"), dict) else {}
    globs = scope.get("path_globs") if isinstance(scope.get("path_globs"), list) else ["."]
    mode = "可写" if allow_write and not task.get("readonly") else "只读"
    from llgraph.loaders.prompt_loader import compose_plan_worker_role, prompt_text

    write_hint = ""
    if allow_write and not task.get("readonly"):
        write_hint = prompt_text("plan", "worker", "write_hint_enabled")
    return compose_plan_worker_role(
        task_title=str(task.get("title") or ""),
        task_description=str(task.get("description") or ""),
        path_globs=", ".join(str(g) for g in globs),
        mode=mode,
        write_hint=write_hint,
    )


def build_subagent_system_prompt(
    ctx: PlanRuntimeContext,
    role_block: str,
    *,
    allow_write: bool,
) -> str:
    """
    子 Agent 完整系统提示：Agent 工具规范 + session-manifest/rules/skills + 角色块。

    @param ctx Plan 运行时上下文
    @param role_block Planner/Worker 角色说明
    @param allow_write 是否可写
    @return 系统提示
    """
    from llgraph.config.survey_settings import survey_interactive_enabled

    base = build_system_prompt(
        ctx.workspace,
        allow_write=allow_write,
        web_search_enabled=ctx.web_search_enabled,
        survey_interactive_enabled=survey_interactive_enabled(ctx.workspace, ctx.context_session),
    )
    if ctx.sandbox_policy is not None and ctx.sandbox_policy.enabled:
        from llgraph.config.sandbox_settings import format_sandbox_config_hint

        base = (
            f"{base}\n\n"
            f"OS 沙箱已启用（{ctx.sandbox_policy.backend}，mode={ctx.sandbox_policy.mode}）。\n"
            f"{format_sandbox_config_hint(ctx.workspace)}"
        )
    return f"{base}{role_block}"
