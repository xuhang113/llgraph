"""llgraph plan 子命令。"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """
    Plan CLI 入口。

    @param argv 参数列表（不含 llgraph plan）
    """
    from llgraph.config.config import load_llgraph_env
    from llgraph.context.context_session import ContextSession
    from llgraph.core.checkpointer_factory import checkpointer_kind
    from llgraph.core.session_bootstrap import AgentRuntimeBundle, build_agent_session_for_thread
    from llgraph.core.tools import load_mcp_tool_bundle
    from llgraph.display.trace_display import TraceMode, TraceSession, parse_trace_mode
    from llgraph.plan.config import resolve_plan_settings
    from llgraph.plan.plan_registry import format_plans_list
    from llgraph.session.session_registry import session_is_resumable
    from llgraph.session.session_web_search import resolve_initial_web_search_enabled
    from llgraph.sandbox.policy import build_sandbox_policy
    from llgraph.config.sandbox_settings import resolve_sandbox_settings
    from llgraph.terminal.mode_loop import run_plan_agent_loop

    load_llgraph_env()

    parser = argparse.ArgumentParser(description="llgraph plan — 多 Agent 编排")
    parser.add_argument("goal", nargs="?", default=None, help="计划目标说明")
    parser.add_argument("--once", "-1", action="store_true", help="单轮后退出")
    parser.add_argument("--list-plans", action="store_true", help="列举历史 Plan 会话")
    parser.add_argument("--thread-id", default=None, help="恢复 plan-{hex} 会话")
    parser.add_argument("-C", "--workspace", default=None, metavar="DIR", help="工作区根")
    parser.add_argument("-w", "--write", action="store_true", help="CLI 允许写（Worker 仍须 Survey 勾选）")
    parser.add_argument("--step", action="store_true", help="每个 task 结束后 interrupt")
    parser.add_argument("--trace", choices=[m.value for m in TraceMode], default=TraceMode.STEPS.value)
    parser.add_argument("--no-survey", action="store_true", help="禁用 Survey（Confirm 用默认批准）")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace or os.getcwd()).expanduser().resolve()
    if not workspace.is_dir():
        print(f"错误: 工作区不是有效目录: {workspace}", file=sys.stderr)
        sys.exit(1)

    settings = resolve_plan_settings(workspace)
    if not settings.enabled:
        print("错误: Plan 模式已在 agent.json 中禁用（plan.enabled=false）", file=sys.stderr)
        sys.exit(1)

    if args.list_plans:
        print(format_plans_list(workspace), flush=True)
        return

    trace_session = TraceSession(mode=parse_trace_mode(args.trace) or TraceMode.STEPS)

    if args.no_survey:
        from llgraph.config.survey_settings import set_survey_cli_disabled

        set_survey_cli_disabled(True)

    explicit_thread = (args.thread_id or "").strip()
    thread_id = explicit_thread or f"plan-{uuid.uuid4().hex[:8]}"
    resume_hint = ""
    if explicit_thread:
        _ok, resume_hint = session_is_resumable(workspace, thread_id)

    mcp_tools, mcp_registry, mcp_summary = load_mcp_tool_bundle(workspace, allow_write=args.write)
    web_search_enabled = resolve_initial_web_search_enabled(workspace)
    sandbox_settings = resolve_sandbox_settings(workspace)
    sandbox_policy = build_sandbox_policy(
        workspace, sandbox_settings, cli_enabled=None, allow_write=args.write
    )
    memory_kind = checkpointer_kind(workspace, with_memory=True)

    from llgraph.code_index.index_watch import attach_watch_shutdown, start_index_watch_with_agent

    watch_service = start_index_watch_with_agent(workspace, no_watch=False)
    attach_watch_shutdown(watch_service)

    bundle = AgentRuntimeBundle(
        workspace=workspace,
        trace_session=trace_session,
        context_session=ContextSession(),
        allow_write=args.write,
        mcp_tools=mcp_tools,
        mcp_registry=mcp_registry,
        watch_service=watch_service,
        web_search_enabled=web_search_enabled,
        sandbox_policy=sandbox_policy,
        sandbox_cli_enabled=None,
        no_spill=False,
        memory_kind=memory_kind,
        mcp_summary=mcp_summary,
        watch_active=watch_service is not None and watch_service.active,
    )

    plan_common = {
        "trace_session": trace_session,
        "context_session": bundle.context_session,
        "settings": settings,
        "allow_write": args.write,
        "step_confirm": args.step,
        "single_turn": args.once,
        "mcp_tools": mcp_tools,
        "sandbox_policy": sandbox_policy,
        "web_search_enabled": web_search_enabled,
        "resume_hint": resume_hint,
        "watch_active": bundle.watch_active,
        "mcp_summary": mcp_summary,
        "memory_kind": memory_kind,
    }

    try:
        run_plan_agent_loop(
            workspace=workspace,
            initial_plan_thread=thread_id,
            opening_goal=(args.goal or "").strip(),
            plan_common=plan_common,
            build_agent_session=lambda tid: build_agent_session_for_thread(bundle, tid),
        )
    finally:
        from llgraph.runtime.shutdown import shutdown_agent_resources

        shutdown_agent_resources(mcp_registry=mcp_registry)


if __name__ == "__main__":
    main(sys.argv[1:])
