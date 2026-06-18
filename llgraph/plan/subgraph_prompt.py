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
    return (
        "\n\n--- Plan Planner 角色 ---\n"
        "你是 Plan 模式的规划 Agent（Planner）。职责：只读调研工作区，输出结构化计划。\n"
        f"工作区: {ctx.workspace}\n"
        "禁止修改代码或写文件。可用检索/读文件工具了解现状。\n"
        "最终必须在回复末尾输出一个 JSON 代码块，格式：\n"
        "```json\n"
        "{\n"
        '  "title": "计划标题",\n'
        '  "tasks": [\n'
        '    {"id": "w1", "title": "...", "description": "...", '
        '"scope": {"path_globs": ["."]}, "depends_on": [], "readonly": true}\n'
        "  ]\n"
        "}\n"
        "```\n"
        "task 应可独立执行、职责单一；depends_on 引用其它 task id。"
    )


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
    return (
        "\n\n--- Plan Worker 角色 ---\n"
        f"你是 Plan Worker，执行单个 task：{task.get('title')}\n"
        f"描述: {task.get('description')}\n"
        f"路径范围: {', '.join(str(g) for g in globs)}\n"
        f"模式: {mode}\n"
        "完成后在回复末尾输出 JSON 摘要：\n"
        "```json\n"
        '{"summary": "一行摘要", "artifacts": [], "status": "done", "files_changed": []}\n'
        "```\n"
        "files_changed 列出本 task 修改/创建的工作区相对路径。"
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
