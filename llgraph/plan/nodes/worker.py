"""Worker 节点：执行单个 task（支持并行 batch）。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from llgraph.plan.plan_store import load_plan, save_task_result, update_task_status
from llgraph.plan.plan_sync import mutate_plan_on_disk
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.subgraphs.worker import WORKER_SUBGRAPH_SPEC
from llgraph.plan.state import PlanPhase, TaskStatus
from llgraph.plan.workflow_view import build_workflow_snapshot
from llgraph.display.trace_display import emit_trace_milestone


def _find_task(plan: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    for task in tasks:
        if isinstance(task, dict) and str(task.get("id")) == task_id:
            return task
    return None


def _parse_worker_result(text: str) -> dict[str, Any]:
    from llgraph.plan.nodes.planner import _extract_plan_json_raw

    raw = _extract_plan_json_raw(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not data.get("summary"):
        data["summary"] = text.strip()[:500] if text.strip() else "（无摘要）"
    data.setdefault("status", TaskStatus.DONE)
    return data


def _build_worker_user_prompt(
    task: dict[str, Any],
    task_id: str,
    task_results: dict[str, Any],
) -> str:
    lines = [
        f"执行 task {task_id}：{task.get('title')}",
        str(task.get("description") or ""),
    ]
    deps = task.get("depends_on") if isinstance(task.get("depends_on"), list) else []
    if deps:
        ctx: list[str] = ["", "## 依赖 Work 已完成结果（供本 task 参考，勿重复劳动）"]
        for dep in deps:
            dep_id = str(dep).strip()
            if not dep_id:
                continue
            row = task_results.get(dep_id)
            if not isinstance(row, dict):
                continue
            summary = str(row.get("summary") or "").strip()
            files = row.get("files_changed") if isinstance(row.get("files_changed"), list) else []
            ctx.append(f"### {dep_id}")
            if summary:
                ctx.append(summary)
            if files:
                ctx.append("改动文件: " + ", ".join(str(f) for f in files if str(f).strip()))
        if len(ctx) > 2:
            lines.extend(ctx)
    return "\n".join(lines)


def run_worker_for_task(
    state: dict[str, Any],
    ctx: PlanRuntimeContext,
    task_id: str,
) -> dict[str, Any]:
    """
    执行单个 Worker task。

    @param state PlanState
    @param ctx 运行时上下文
    @param task_id Task ID
    @return state 更新片段
    """
    plan = dict(state.get("plan") or {})
    task = _find_task(plan, task_id)
    if task is None:
        return {"error": f"task 不存在: {task_id}"}

    allow_write = ctx.worker_allow_write(state) and not bool(task.get("readonly"))
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
    plans_dir = ctx.settings.plans_dir

    def _mark_status(status: str, *, error: str | None = None) -> dict[str, Any]:
        return mutate_plan_on_disk(
            ctx.workspace,
            plan_id,
            lambda p: update_task_status(p, task_id, status, error=error),
            plans_dir=plans_dir,
            fallback_plan=plan,
        )

    plan = _mark_status(TaskStatus.RUNNING)

    from llgraph.plan.execution_coordinator import is_cancel_requested, is_task_cancel_requested
    from llgraph.plan.subgraphs.worker import run_worker_subagent

    if is_task_cancel_requested(ctx.thread_id, task_id) or is_cancel_requested(ctx.thread_id):
        plan = _mark_status(TaskStatus.SKIPPED, error="用户取消")
        return {
            "plan": plan,
            "task_results": dict(state.get("task_results") or {}),
            "error": f"Worker {task_id} 已取消",
        }

    task_results = dict(state.get("task_results") or {})
    prompt = _build_worker_user_prompt(task, task_id, task_results)
    sub_thread = f"{ctx.thread_id}{WORKER_SUBGRAPH_SPEC.thread_suffix.format(task_id=task_id)}"
    worker_ctx = ctx.fork_worker_runtime(task_id=task_id, sub_thread=sub_thread)
    try:
        text, messages, files_changed = run_worker_subagent(
            worker_ctx,
            task,
            task_id=task_id,
            allow_write=allow_write,
            user_prompt=prompt,
            plan_state=state,
        )
    except Exception as exc:
        if is_task_cancel_requested(ctx.thread_id, task_id) or is_cancel_requested(ctx.thread_id):
            plan = _mark_status(TaskStatus.SKIPPED, error="用户取消")
            err = f"Worker {task_id} 已取消"
        else:
            plan = _mark_status(TaskStatus.FAILED, error=str(exc))
            err = f"Worker {task_id} 失败: {exc}"
        return {
            "plan": plan,
            "task_results": task_results,
            "error": err,
        }

    if is_task_cancel_requested(ctx.thread_id, task_id) or is_cancel_requested(ctx.thread_id):
        plan = _mark_status(TaskStatus.SKIPPED, error="用户取消")
        return {
            "plan": plan,
            "task_results": task_results,
            "error": f"Worker {task_id} 已取消",
        }

    parsed = _parse_worker_result(text)
    final_status = TaskStatus.DONE if parsed.get("status") != TaskStatus.FAILED else TaskStatus.FAILED
    err_msg = str(parsed.get("error") or "") or None
    merged_files = files_changed or [
        str(f) for f in (parsed.get("files_changed") or []) if str(f).strip()
    ]
    if allow_write and not merged_files and final_status == TaskStatus.DONE:
        final_status = TaskStatus.FAILED
        err_msg = err_msg or "可写任务未产生文件改动（须调用 write_file 落盘）"
    plan = _mark_status(final_status, error=err_msg)

    result_doc = {
        "task_id": task_id,
        "summary": parsed.get("summary"),
        "artifacts": parsed.get("artifacts") or [],
        "status": final_status,
        "allow_write": allow_write,
        "files_changed": merged_files,
    }
    save_task_result(ctx.workspace, plan_id, task_id, result_doc, plans_dir=plans_dir)

    task_results = dict(state.get("task_results") or {})
    task_results[task_id] = result_doc

    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.EXECUTING,
        plan=plan,
        current_node="worker",
        current_task_id=task_id,
    )
    return {
        "plan": plan,
        "task_results": task_results,
        "current_task_id": task_id,
        "workflow_snapshot": snapshot,
    }


def _merge_worker_updates(
    state: dict[str, Any],
    ctx: PlanRuntimeContext,
    updates: list[dict[str, Any]],
    *,
    batch: list[str],
) -> dict[str, Any]:
    """
    合并并行 Worker 返回值并刷新 plan 快照。

    @param state 执行前 PlanState
    @param ctx 运行时上下文
    @param updates 各 Worker 返回片段
    @param batch 本批 task id
    @return 合并后的 state 更新
    """
    task_results = dict(state.get("task_results") or {})
    errors: list[str] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        partial = update.get("task_results")
        if isinstance(partial, dict):
            task_results.update(partial)
        err = update.get("error")
        if err:
            errors.append(str(err))

    plan = dict(state.get("plan") or {})
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
    if plan_id:
        disk_plan = load_plan(ctx.workspace, plan_id, plans_dir=ctx.settings.plans_dir)
        if disk_plan:
            plan = disk_plan

    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.EXECUTING,
        plan=plan,
        current_node="worker",
        current_task_id=batch[0] if len(batch) == 1 else None,
    )
    merged: dict[str, Any] = {
        "plan": plan,
        "task_results": task_results,
        "current_task_id": batch[-1] if batch else None,
        "workflow_snapshot": snapshot,
        "parallel_batch": [],
    }
    if errors:
        merged["error"] = "; ".join(errors)
    return merged


def worker_node(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    Worker 节点：顺序或并行执行 batch。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    from llgraph.plan.execution_coordinator import is_cancel_requested

    if is_cancel_requested(ctx.thread_id) or state.get("cancel_requested"):
        return {"parallel_batch": []}

    batch = state.get("parallel_batch")
    if not isinstance(batch, list) or not batch:
        tid = state.get("current_task_id")
        if tid:
            batch = [str(tid)]
        else:
            return {}

    task_ids = [str(tid) for tid in batch if str(tid).strip()]
    if not task_ids:
        return {"parallel_batch": []}

    from llgraph.plan.execution_coordinator import is_task_cancel_requested

    task_ids = [tid for tid in task_ids if not is_task_cancel_requested(ctx.thread_id, tid)]
    if not task_ids:
        return {"parallel_batch": []}

    if len(task_ids) == 1:
        merged = run_worker_for_task(dict(state), ctx, task_ids[0])
        merged["parallel_batch"] = []
    else:
        emit_trace_milestone(
            ctx.trace_session,
            f"并行执行 {len(task_ids)} 个 Worker: {', '.join(task_ids)}"
            "（/trace 日志可能交错，各 Worker 子会话 thread 不同）",
        )
        updates: list[dict[str, Any]] = []
        max_workers = min(len(task_ids), ctx.settings.max_parallel_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_worker_for_task, dict(state), ctx, tid): tid
                for tid in task_ids
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    updates.append(future.result())
                except Exception as exc:
                    updates.append({"error": f"Worker {tid} 失败: {exc}"})
        merged = _merge_worker_updates(state, ctx, updates, batch=task_ids)

    if state.get("step_confirm_each_task"):
        interrupt({"type": "task_step_confirm", "task_id": task_ids[-1] if task_ids else None})

    return merged
