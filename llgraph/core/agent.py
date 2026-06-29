"""
LangGraph ReAct Agent（参照官方 StateGraph / prebuilt 用法）。

文档: https://docs.langchain.com/oss/python/langgraph/overview
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
import time

from llgraph.core.react_limits import resolve_agent_max_turns
from llgraph.core.react_graph import build_react_graph

from llgraph.core.checkpointer_factory import create_checkpointer

from llgraph.context.context_builder import build_workspace_context_block
from llgraph.core.user_message_content import (
    ChatImageInput,
    build_human_content_blocks,
)
from llgraph.context.context_compressor import (
    apply_compress_to_agent_state,
    format_compress_report,
)
from llgraph.context.context_session import ContextSession
from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_settings import resolve_effective_model
from llgraph.core.tools import get_agent_tools
from llgraph.session.session_edits import SessionEditTracker
from llgraph.context.context_spill import ContextSpill
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.sandbox.policy import SandboxPolicy
from llgraph.loaders.thought_loader import build_thought_prompt_block
from llgraph.session.session_manifest import sync_session_manifest_to_agent_state
from llgraph.context.message_normalize import make_prompt_normalizer
from llgraph.display.trace_display import TraceSession, stream_agent_turn


def build_system_prompt(
    workspace_root: Path,
    *,
    allow_write: bool,
    web_search_enabled: bool = False,
    survey_interactive_enabled: bool = True,
) -> str:
    """
    根据工作区与权限生成系统提示。

    @param workspace_root 工作区绝对路径
    @param allow_write 是否允许写入
    @param web_search_enabled 是否已注册 web_search
    @param survey_interactive_enabled 是否启用交互式 survey 向导
    @return 系统提示词
    """
    from llgraph.code_index.index_ready import code_index_is_ready
    from llgraph.core.model_thinking import resolve_model_thinking_payload
    from llgraph.loaders.prompt_loader import compose_agent_system_prompt, compose_search_order_hint

    mode = "可读写" if allow_write else "只读（禁止调用 write_file、append_file、search_replace）"
    index_ready = code_index_is_ready(workspace_root)
    tools_read, search_order_hint = compose_search_order_hint(index_ready=index_ready)
    if web_search_enabled:
        tools_read += "、web_search"
    tools_write = "、search_replace、append_file、write_file" if allow_write else ""
    edit_hint = ""
    if allow_write:
        from llgraph.loaders.prompt_loader import prompt_text

        edit_hint = prompt_text("agent", "tools", "edit_hint")

    model_id = resolve_effective_model(workspace_root)
    from llgraph.core.model_thinking import is_thinking_enabled, resolve_model_thinking_payload

    thinking_payload = (
        resolve_model_thinking_payload(workspace_root, model_id)
        if is_thinking_enabled(workspace_root, model_id)
        else None
    )

    base = compose_agent_system_prompt(
        workspace_root=workspace_root,
        model_id=model_id,
        mode=mode,
        tools_read=tools_read,
        tools_write=tools_write,
        edit_hint=edit_hint,
        search_order_hint=search_order_hint,
        thinking_payload=thinking_payload,
        web_search_enabled=web_search_enabled,
        allow_write=allow_write,
        survey_interactive_enabled=survey_interactive_enabled,
    )

    thought_block = build_thought_prompt_block(workspace_root)
    if thought_block:
        base = base + "\n\n" + thought_block
    return base


def build_agent(
    *,
    with_memory: bool = False,
    workspace_root: str | Path | None = None,
    allow_write: bool = False,
    edit_tracker: SessionEditTracker | None = None,
    on_file_changed: Callable[[str], None] | None = None,
    mcp_tools: list | None = None,
    context_spill: ContextSpill | None = None,
    write_failure_tracker: WriteFailureTracker | None = None,
    web_search_enabled: bool = False,
    context_session: ContextSession | None = None,
    sandbox_policy: SandboxPolicy | None = None,
):
    """
    构建 ReAct Agent：LLM（Gateway）+ 工具循环。

    @param with_memory 是否启用会话记忆（MemorySaver）
    @param workspace_root 文件工具的工作区根目录
    @param allow_write 是否允许写入文件（默认只读）
    @param edit_tracker 会话编辑账本
    @param on_file_changed 写文件成功回调
    @param mcp_tools 外部 MCP 工具
    @param context_spill 工具结果落盘
    @param write_failure_tracker 写工具失败提醒
    @param web_search_enabled 是否注册 web_search
    @param sandbox_policy OS 沙箱策略
    @return 已编译的 LangGraph Runnable
    """
    root = Path(workspace_root or ".").expanduser().resolve()
    llm = create_gateway_llm(root)
    tools = get_agent_tools(
        workspace_root=root,
        allow_write=allow_write,
        edit_tracker=edit_tracker,
        on_file_changed=on_file_changed,
        mcp_tools=mcp_tools,
        context_spill=context_spill,
        write_failure_tracker=write_failure_tracker,
        web_search_enabled=web_search_enabled,
        sandbox_policy=sandbox_policy,
    )
    from llgraph.core.llm_settings import resolve_effective_model
    from llgraph.core.prompt_cache import (
        build_cache_control,
        tag_tools_for_prompt_cache,
    )
    from llgraph.core.prompt_cache_settings import (
        prompt_cache_enabled_for_model,
        resolve_prompt_cache_settings,
    )

    cache_settings = resolve_prompt_cache_settings(root)
    model_id = resolve_effective_model(root)
    if prompt_cache_enabled_for_model(root, model_id) and cache_settings.enabled:
        cache_control = build_cache_control(cache_settings)
        if cache_settings.tag_tools:
            tools = tag_tools_for_prompt_cache(tools, cache_control)
    checkpointer = create_checkpointer(root, with_memory=with_memory)
    from llgraph.config.survey_settings import survey_interactive_enabled

    system_prompt = build_system_prompt(
        root,
        allow_write=allow_write,
        web_search_enabled=web_search_enabled,
        survey_interactive_enabled=survey_interactive_enabled(root, context_session),
    )
    if sandbox_policy is not None and sandbox_policy.enabled:
        from llgraph.config.sandbox_settings import format_sandbox_config_hint

        system_prompt = (
            f"{system_prompt}\n\n"
            f"OS 沙箱已启用（{sandbox_policy.backend}，mode={sandbox_policy.mode}）。"
            "文件工具与 Shell 受 sandbox.json 路径/网络策略约束；"
            "被拒绝时会提示配置路径。\n"
            f"{format_sandbox_config_hint(root)}"
        )

    return build_react_graph(
        llm,
        tools,
        prompt=make_prompt_normalizer(system_prompt, root),
        checkpointer=checkpointer,
        workspace=root,
    )


def rebuild_agent_preserving_memory(
    agent_session: Any,
    *,
    allow_write: bool,
    web_search_enabled: bool | None = None,
    mcp_tools: list | None = None,
    on_file_changed: Callable[[str], None] | None = None,
    sandbox_policy: SandboxPolicy | None = None,
) -> Any:
    """
    切换模型等场景下重建 Agent，并尽量保留会话消息。

    @param agent_session AgentSessionContext
    @param allow_write 是否允许写文件
    @param web_search_enabled 是否注册 web_search；None 时沿用会话状态
    @param mcp_tools MCP 工具列表
    @param on_file_changed 写文件回调
    @param sandbox_policy 沙箱策略；None 时沿用 agent_session.sandbox_policy
    @return 新 Agent
    """
    config = {"configurable": {"thread_id": agent_session.thread_id}}
    messages: list = []
    if agent_session.with_memory:
        try:
            state = agent_session.agent.get_state(config)
            messages = list((state.values or {}).get("messages") or [])
        except Exception:
            messages = []

    web = (
        web_search_enabled
        if web_search_enabled is not None
        else getattr(agent_session, "web_search_enabled", False)
    )
    policy = (
        sandbox_policy
        if sandbox_policy is not None
        else getattr(agent_session, "sandbox_policy", None)
    )
    if sandbox_policy is not None:
        agent_session.sandbox_policy = sandbox_policy

    new_agent = build_agent(
        with_memory=agent_session.with_memory,
        workspace_root=agent_session.workspace,
        allow_write=allow_write,
        edit_tracker=agent_session.edit_tracker,
        on_file_changed=on_file_changed,
        mcp_tools=mcp_tools,
        context_spill=agent_session.context_spill,
        write_failure_tracker=agent_session.write_failure_tracker,
        web_search_enabled=web,
        context_session=agent_session.context_session,
        sandbox_policy=policy,
    )
    if messages and agent_session.with_memory:
        from llgraph.context.message_normalize import reorder_pinned_system_messages

        messages = reorder_pinned_system_messages(messages)
        try:
            new_agent.update_state(config, {"messages": messages})
        except Exception:
            pass
    agent_session.agent = new_agent
    return new_agent


def _extract_last_assistant_text(messages: list) -> str:
    """从消息列表取最后一条助手文本。"""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            text = "".join(parts).strip()
            if text:
                return text
    return ""


def invoke_agent(
    agent,
    user_message: str,
    *,
    workspace_root: Path | str | None = None,
    thread_id: str = "default",
    with_memory: bool = False,
    trace_session: TraceSession | None = None,
    context_session: ContextSession | None = None,
    effective_message_override: str | list[dict[str, Any]] | None = None,
    write_failure_tracker: WriteFailureTracker | None = None,
    context_spill: Any | None = None,
    allow_write: bool = False,
    cancel_check: Any | None = None,
    run_source: str = "cli",
    images: list[ChatImageInput] | None = None,
) -> str:
    """
    执行一轮对话并返回助手最后一条文本。

    @param agent build_agent 返回值
    @param user_message 用户输入
    @param workspace_root 工作区根（注入 Rules/Skills/文档索引）
    @param thread_id 会话线程 ID（启用 memory 时区分会话）
    @param with_memory 是否与 build_agent(with_memory=True) 一致
    @param trace_session 过程展示（/trace 切换；默认 steps）
    @param context_session Rule/Skill 会话状态
    @param effective_message_override 覆盖发给模型的消息（自定义命令用）
    @param images 用户附带的图片（Web 多模态）
    @param write_failure_tracker 写工具失败跟踪
    @param allow_write Web/CLI 当前是否可写（同步 manifest 与 workspace-context）
    @param cancel_check 可选；返回 True 时在 ReAct 步间中断（Web Stop）
    @param run_source 运行来源（cli | web），写入 last_run.json
    @return 助手回复文本
    """
    trace = trace_session or TraceSession()
    ctx = context_session or ContextSession()
    root = Path(workspace_root or ".").expanduser().resolve()
    from llgraph.context.runtime_context import set_active_thread_id
    from llgraph.display.trace_display import print_invoke_prelude

    print_invoke_prelude(trace)

    set_active_thread_id(thread_id if with_memory else None)
    if with_memory and user_message.strip():
        from llgraph.session.session_meta import ensure_session_title_auto

        ensure_session_title_auto(root, thread_id, user_message)
    archive_path: str | None = None
    compress_report = None
    spill_count_at_start = 0
    if context_spill is not None:
        spill_count_at_start = context_spill.spill_count()
    if with_memory:
        from llgraph.context.chat_history_repair import ensure_agent_chat_history_sanitized
        from llgraph.context.incremental_context import (
            apply_incremental_tool_prune_to_agent_state,
            format_tool_prune_report,
        )

        prune_report = apply_incremental_tool_prune_to_agent_state(
            agent,
            thread_id=thread_id,
            workspace=root,
        )
        if prune_report is not None:
            from llgraph.terminal.ops_notice import ops_notice

            ops_notice(format_tool_prune_report(prune_report))

        from llgraph.context.context_settings import is_auto_compress_strategy, resolve_context_settings

        ctx_settings = resolve_context_settings(root)
        invoke_preserve = (
            False if is_auto_compress_strategy(ctx_settings.compress_strategy) else None
        )
        compress_report = apply_compress_to_agent_state(
            agent,
            thread_id=thread_id,
            workspace=root,
            force=False,
            preserve_current_turn=invoke_preserve,
        )
        if compress_report is not None:
            from llgraph.terminal.ops_notice import ops_notice

            ops_notice(format_compress_report(compress_report))
            archive_path = compress_report.archive_path
            from llgraph.display.execution_log import log_compress_event

            log_compress_event(
                root,
                thread_id=thread_id,
                report=compress_report,
                trigger="auto",
            )

    sync_session_manifest_to_agent_state(
        agent,
        thread_id=thread_id,
        workspace=root,
        session=ctx,
        user_message=user_message,
        with_memory=with_memory,
        archive_path=archive_path,
        allow_write=allow_write,
    )
    if with_memory:
        from llgraph.context.chat_history_repair import ensure_agent_chat_history_sanitized

        # 仅 canonical 落盘清理（不按当前模型展开），出站修链在 prompt normalizer
        ensure_agent_chat_history_sanitized(agent, root, thread_id)

    recent_messages: list[BaseMessage] | None = None
    edited_paths: list[str] | None = None
    if with_memory:
        from langchain_core.messages import BaseMessage

        try:
            state = agent.get_state({"configurable": {"thread_id": thread_id}})
            recent_messages = list((state.values or {}).get("messages") or [])
        except Exception:
            recent_messages = None
        if allow_write:
            try:
                from llgraph.console.edit_service import session_edit_tracker

                tracker = session_edit_tracker(root, thread_id)
                paths = tracker.unique_paths()
                edited_paths = paths if paths else None
            except Exception:
                edited_paths = None

    context_block = build_workspace_context_block(
        root,
        ctx,
        user_message,
        allow_write=allow_write,
        recent_messages=recent_messages,
        edited_paths=edited_paths,
    )
    turn_image_refs: list = []
    if effective_message_override is not None:
        effective = effective_message_override
    else:
        images_for_llm = images
        if with_memory and images:
            from llgraph.session.session_image_store import (
                load_chat_images_as_input,
                save_chat_images,
            )

            turn_image_refs = save_chat_images(root, thread_id, images)
            images_for_llm = load_chat_images_as_input(root, thread_id, turn_image_refs)
        effective = build_human_content_blocks(
            user_message,
            images=images_for_llm,
            context_block=context_block,
        )
    if with_memory and (user_message.strip() or images):
        from llgraph.session.session_file_store import append_pending_user_turn

        append_pending_user_turn(
            root,
            thread_id,
            user_message,
            image_refs=turn_image_refs or None,
        )

    from llgraph.display.execution_log import log_turn_end, log_turn_failure, log_turn_start
    from llgraph.session.session_run_log import (
        UserCancelledError,
        trace_run_context,
        write_session_last_run,
    )

    log_turn_start(
        root,
        thread_id=thread_id,
        user_message=user_message,
        trace_mode=trace.mode.value,
    )

    turn_wall_start = time.perf_counter()
    try:
        turn_result = stream_agent_turn(
            agent,
            user_message,
            thread_id=thread_id,
            with_memory=with_memory,
            trace_session=trace,
            effective_message=effective,
            write_failure_tracker=write_failure_tracker,
            workspace=root,
            context_session=ctx,
            recursion_limit=resolve_agent_max_turns(root),
            cancel_check=cancel_check,
        )
    except Exception as exc:
        partial_tools: list[str] = []
        printer = trace.active_printer
        if printer is not None and getattr(printer, "_tool_names", None):
            partial_tools = list(printer._tool_names)
        run_ctx = trace_run_context(trace)
        duration = time.perf_counter() - turn_wall_start
        outcome = "timeout" if type(exc).__name__ in ("TimeoutError", "ReadTimeout", "APITimeoutError") else "error"
        log_turn_failure(
            root,
            thread_id=thread_id,
            with_memory=with_memory,
            agent=agent,
            duration_sec=duration,
            error=exc,
            tool_names=partial_tools,
            compress_report=compress_report,
            spill=context_spill,
            spill_count_at_start=spill_count_at_start,
            trace_mode=trace.mode.value,
            outcome=outcome,
            trace_context=run_ctx,
            user_message=user_message,
        )
        if with_memory:
            from llgraph.core.llm_settings import resolve_effective_model

            write_session_last_run(
                root,
                thread_id,
                outcome=outcome,
                duration_sec=duration,
                model=resolve_effective_model(root),
                user_message=user_message,
                error=exc,
                trace_context=run_ctx,
                source=run_source,
            )
        raise

    duration = turn_result.duration_sec
    run_ctx = trace_run_context(trace)
    cancelled = cancel_check is not None and cancel_check()
    if cancelled:
        log_turn_failure(
            root,
            thread_id=thread_id,
            with_memory=with_memory,
            agent=agent,
            duration_sec=duration,
            error=UserCancelledError("用户停止当前生成"),
            tool_names=turn_result.tool_names,
            compress_report=compress_report,
            spill=context_spill,
            spill_count_at_start=spill_count_at_start,
            trace_mode=trace.mode.value,
            outcome="cancelled",
            trace_context=run_ctx,
            user_message=user_message,
        )
        if with_memory:
            from llgraph.core.llm_settings import resolve_effective_model

            write_session_last_run(
                root,
                thread_id,
                outcome="cancelled",
                duration_sec=duration,
                model=resolve_effective_model(root),
                user_message=user_message,
                error=UserCancelledError("用户停止当前生成"),
                trace_context=run_ctx,
                source=run_source,
            )
    else:
        log_turn_end(
            root,
            thread_id=thread_id,
            with_memory=with_memory,
            agent=agent,
            tool_names=turn_result.tool_names,
            duration_sec=duration,
            compress_report=compress_report,
            spill=context_spill,
            spill_count_at_start=spill_count_at_start,
            trace_mode=trace.mode.value,
            trace_context=run_ctx,
            user_message=user_message,
        )
        if with_memory:
            from llgraph.core.llm_settings import resolve_effective_model

            write_session_last_run(
                root,
                thread_id,
                outcome="ok",
                duration_sec=duration,
                model=resolve_effective_model(root),
                user_message=user_message,
                trace_context=run_ctx,
                source=run_source,
            )
    if with_memory:
        from llgraph.context.incremental_context import (
            apply_incremental_tool_prune_to_agent_state,
            format_tool_prune_report,
        )
        from llgraph.session.session_file_store import persist_agent_session

        persist_agent_session(
            agent,
            root,
            thread_id,
            turn_image_refs=turn_image_refs or None,
        )
        end_prune = apply_incremental_tool_prune_to_agent_state(
            agent,
            thread_id=thread_id,
            workspace=root,
        )
        if end_prune is not None:
            from llgraph.terminal.ops_notice import ops_notice

            ops_notice(format_tool_prune_report(end_prune))
    return turn_result.text
