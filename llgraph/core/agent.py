"""
LangGraph ReAct Agent（参照官方 StateGraph / prebuilt 用法）。

文档: https://docs.langchain.com/oss/python/langgraph/overview
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.prebuilt import create_react_agent

from llgraph.core.checkpointer_factory import create_checkpointer

from llgraph.context.context_builder import (
    build_workspace_context_block,
    wrap_user_message_with_context,
)
from llgraph.context.context_compressor import (
    apply_compress_to_agent_state,
    format_compress_report,
)
from llgraph.context.context_session import ContextSession
from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_settings import resolve_effective_model
from llgraph.core.tools import get_agent_tools
from llgraph.survey.edit_confirm import EditConfirmGate
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
    mode = "可读写" if allow_write else "只读（禁止调用 write_file、append_file、search_replace）"
    from llgraph.code_index.index_ready import code_index_is_ready

    index_ready = code_index_is_ready(workspace_root)
    batch_search_hint = (
        "同时查找**多个已知文件名/脚本名**时："
        "用**一次** grep_files（如 pattern=\"collect_alert|gitlab_monitor|dm-daily-task\"，"
        "path=\"markdowns\" 或 path=\".\"）；**禁止**对每个文件名逐个调用 glob_files。\n"
        "glob 全未命中或连续多个 glob 为 0："
        "源文件可能不在工作区、仅在 markdowns/docs 台账/crontab 文档中被引用；"
        "改用 grep_files（path=\"markdowns\" 或 path=\"docs\"）或 read_file 已知文档，"
        "勿再逐个 glob。\n"
    )
    path_scope_hint = (
        "**glob_files / grep_files 的 path（必须遵守）**：\n"
        "- path = 相对工作区根的**搜索根（子树边界）**，无 cwd；只在 path 子树内匹配，**不含兄弟目录**。\n"
        "- **禁止** path 含 \"../\"（会越界或解析错误）。\n"
        "- 默认 path=\".\" 或 \"<仓库名>\"（如 auth-api）；"
        "**禁止**沿用 list_directory 后的深层 path 做跨包 glob"
        "（如 path=.../activity 却 glob **/basic/**）。\n"
        "- list_directory 仅用于浏览；glob/grep 搜类/模块时 path 至少到**仓库根**。\n"
        "- glob/grep 0 命中后**禁止同 path 再试**；须换更宽 path 或 search_code_hybrid。\n"
    )
    batch_read_hint = (
        "**批量读文件（减少 I/O 与 ReAct 轮次，对齐 Cursor）**：\n"
        "- hybrid/glob/grep 已给出多个**完整相对路径**时：**尽量一次** "
        "read_files(paths=[...])（最多 8 个）取齐，**禁止**每个路径单独 read_file 各占一轮 LLM。\n"
        "- 能批量就批量：同一轮需要对比/梳理 N 个类时，一次 read_files 优于 N 次 read_file。\n"
        "- 推荐流程：search_code_hybrid 规划 → read_files 批量读 → 正文汇总；"
        "单文件过大或只需局部时用 read_file(start_line/end_line)。\n"
        "- read/glob path **禁止** \"../\" 或猜路径；须从检索结果**原样复制**完整路径。\n"
        "- read_file/read_files 结果**不落盘**，全文直接进上下文；超大单文件请分段 read_file。\n"
        "- read_file/read_files 失败后**禁止**再 glob 同名补救（应回到 hybrid 换路径）。\n"
    )
    if index_ready:
        tools_read = (
            "list_directory、glob_files、grep_files、search_code_hybrid、"
            "search_code_semantic、search_workspace、"
            "read_file、read_files、search_session_history、"
            "run_shell_command、get_current_utc_time"
        )
        search_order_hint = (
            "本工作区**代码向量索引已就绪**（llgraph index）。\n"
            "目录浏览与文件发现**禁止** run_shell_command 的 ls/ls -la/find/tree/du；"
            "用内置工具：列目录 **list_directory**；"
            "找文件名 **glob_files**；搜内容 **grep_files**；"
            "读文件 **read_file** / 批量 **read_files**。\n"
            + path_scope_hint
            + batch_read_hint
            + "找文件/脚本/类/业务概念时检索顺序：\n"
            "1) **search_code_hybrid** — **首选**（路径匹配 + grep + 语义，一次调用）；\n"
            "2) **read_files** — hybrid/glob 已列出多个路径时**一次批量读**；\n"
            "3) **grep_files** — 已知精确类名/表名/符号（path 用 \".\" 或仓库名）；\n"
            "4) **glob_files** — 仅当知精确**文件名**且 path=\".\" 或仓库名；\n"
            "5) search_code_semantic / search_workspace — 兜底。\n"
            + batch_search_hint
            + "glob/grep 0 命中：先看工具返回的 path 作用域提示；"
            "内容可能在 markdowns/docs，或 path 过深，勿重复 find/ls/glob。\n"
        )
    else:
        tools_read = (
            "list_directory、glob_files、grep_files、search_code_hybrid、"
            "search_code_semantic、search_workspace、search_files、"
            "read_file、read_files、search_session_history、"
            "run_shell_command、get_current_utc_time"
        )
        search_order_hint = (
            "本工作区**尚未建立代码向量索引**（请 llgraph index -C .）。\n"
            "列目录用 **list_directory**；**禁止** run_shell_command 的 ls/find/tree 做目录浏览或文件发现。\n"
            + path_scope_hint
            + batch_read_hint
            + "检索顺序：glob_files → grep_files → read_files（多路径）→ search_files / search_workspace；"
            "建索引后优先 search_code_hybrid。\n"
            + batch_search_hint
        )
    if web_search_enabled:
        tools_read += "、web_search"
    tools_write = "、search_replace、append_file、write_file" if allow_write else ""
    edit_hint = (
        "修改已有文件优先 search_replace（局部替换）；新建文件先用 write_file 写骨架，"
        "长 Markdown/文档分节用 append_file 或 search_replace 追加，禁止一次 tool_call 塞入整篇长文；"
        "每次 write_file/append_file 必须同时提供 path 与 content，禁止只传 path。"
        "替换前须 read_file 确认 old_string 与磁盘一致（含缩进）。"
        if allow_write
        else ""
    )

    model_id = resolve_effective_model(workspace_root)
    from llgraph.core.model_thinking import resolve_model_thinking_payload

    thinking_payload = resolve_model_thinking_payload(workspace_root, model_id)
    thinking_hint = ""
    if thinking_payload is not None:
        thinking_hint = (
            f"网关 thinking 已启用（agent.json）：{thinking_payload}。"
            "用户询问扩展思考/thinking 模式时，如实说明已开启；勿回答「不支持」或「无法传 thinking 参数」。\n"
            "**thinking 输出约束（必须遵守）**：\n"
            "- thinking/reasoning 仅作内部推理，**终端用户看不到**；"
            "面向用户的结论、梳理、代码说明、步骤总结**必须写在正文 text**，不可只写在 thinking 里。\n"
            "- 调用工具前可以只有 thinking + tool_call；"
            "**一旦本轮不再调用工具（结束 turn），必须输出完整可见正文**，"
            "禁止 thinking-only 就 end_turn。\n"
            "- 用户要求「仅对话、不落盘」时：结论仍须出现在正文 text，"
            "只是不要 write_file；**不是**把答案只放在 thinking。\n"
            "- 工具读完后要给用户汇总时：先停止调工具，再单独一轮用正文输出完整梳理，"
            "不要 endless read_file 后在 thinking 里草稿即结束。\n"
        )
    base = (
        "你是工作区编程助手，通过 OpenAI 兼容 API 网关（LLGRAPH_API_BASE_URL）调用大模型。\n"
        f"当前模型: {model_id}\n"
        + thinking_hint
        + "若用户问「你是什么模型 / 你是谁」，只回答当前模型 id，不要自称 Claude、GPT 或其它未在配置中的名称。\n"
        f"当前工作区根目录: {workspace_root}\n"
        f"文件访问模式: {mode}。\n"
        f"可用工具: {tools_read}{tools_write}。\n"
        + (f"{edit_hint}\n" if edit_hint else "")
        + "**终端展示与落盘（勿混淆）**：\n"
        "- 用户看到的回复 = 助手**正文 text**（终端流式输出）；thinking、tool 结果预览**不算**用户可见答复。\n"
        "- 用户说「放到 text / 正文 / 对话里展示 / 不要落盘 / 仅对话」时：**禁止** write_file/append_file；"
        "完整梳理须直接写在助手正文里，**不是**写 .md 文件，也不是只写 thinking。\n"
        "- write_file 仅当用户**明确要求**保存到某路径（如「写到 docs/xxx.md」）时使用；"
        "写文件后仍须在正文给出可读的结论摘要，不能只有「已写入 xxx.md」一句。\n"
        + "Skill/Rule 全量目录与简介在置顶 <session-manifest>；正文不在上下文中。"
        "需要某技能/规则时用 read_file 对应路径（见 manifest 目录）。"
        "/skill <name> 可会话置顶该技能路径；不自动匹配、不注入 SKILL 正文。\n"
        + "置顶 <session-manifest>、<conversation-anchor>（路径见 manifest.json；通常在 ~/.llgraph/context/<工作区>/sessions/） "
        "在上下文压缩后仍保留路径指针；远早对话原文不在每轮上下文中。\n"
        + "会话失忆/指代不清/压缩或换模型后：先读置顶 anchor，仍不足则 **search_session_history(query=关键词)** "
        "检索归档与 messages.jsonl，勿臆造未检索过的历史细节；需要全文再 read_file 归档路径。\n"
        + "业务归属、项目对照：先 read_file 技能/规则/ markdowns 索引中的文档，再扫代码；"
        "禁止未读文档硬猜服务名。\n"
        + search_order_hint
        + "run_shell_command **仅**用于需要真实 shell 环境的操作："
        "pwd、git status/log/diff、mvn/npm 构建、运行**已定位**的脚本/测试、docker 等；"
        "cwd 相对工作区。\n"
        "**禁止**用 shell 的 ls/ls -la/find/grep/rg/cat/head/tail 做目录浏览、文件发现或读代码"
        "（请用 list_directory / glob_files / grep_files / read_file / read_files）。"
        "不要声称无法执行 shell。\n"
        "路径一律使用相对工作区的路径（如 services/order-api/README.md）。\n"
        "工具返回若含「工具结果已落盘」与路径，表示全文在磁盘；用 read_file/grep_files 按需读取，"
        "勿假设预览即全文。\n"
        + (
            "需要互联网最新信息（版本、新闻、公开文档）时使用 web_search；"
            "本地代码与 .llgraph 文档优先用检索工具。\n"
            if web_search_enabled
            else ""
        )
        + "回答简洁、准确；需要实时信息时使用工具。"
    )
    if allow_write and survey_interactive_enabled:
        base = (
            base
            + "\n\n需要用户确认时：禁止要求用户「输入序号/打字」；"
            "须在回复末尾输出 <<<llgraph-survey>>> JSON <<<end-survey>>>，"
            "必须含 title 与 questions 数组（每题含 id、prompt、options≥2、step_label；"
            "多选题为 multi_select:true，终端用 Space 勾选、Enter 确认；"
            "option_hints 为每题可选说明，禁止放在 JSON 根级）。"
            "终端会隐藏该 JSON 并弹出确认向导。"
            "问卷仅由 Agent 在回复末尾输出 <<<llgraph-survey>>> 触发；"
            "llgraph 不在 Agent 运行前弹出固定问卷。"
            "遵循 project-organize 等 Skill 时："
            "单项目（仅 1 仓）无 doc 时直接落盘、有 doc 时 survey 仅 mode+overwrite；"
            "多项目（≥2 仓或业务域）survey 必须含 scope 确认改动哪些仓库，"
            "禁止因依赖关系自动扩仓落盘；无 doc 时至少先 scope 再读代码/落盘。"
        )
    elif allow_write:
        base = (
            base
            + "\n\n本环境已禁用交互式 survey（--no-survey 或非交互 Agent）。"
            "需要用户确认时：直接在回复中列出选项与建议，勿输出 <<<llgraph-survey>>>；"
            "不要等待用户在下拉菜单中确认，按合理默认继续执行并说明假设。"
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
    edit_confirm_gate: EditConfirmGate | None = None,
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
    @param edit_confirm_gate 写文件前终端确认
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
        edit_confirm_gate=edit_confirm_gate,
        sandbox_policy=sandbox_policy,
    )
    from llgraph.core.llm_settings import resolve_effective_model
    from llgraph.core.prompt_cache import (
        apply_prompt_cache_to_llm,
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
        llm = apply_prompt_cache_to_llm(llm, root)
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

    return create_react_agent(
        llm,
        tools,
        prompt=make_prompt_normalizer(system_prompt, root),
        checkpointer=checkpointer,
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
        edit_confirm_gate=getattr(agent_session, "edit_confirm_gate", None),
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
    effective_message_override: str | None = None,
    write_failure_tracker: WriteFailureTracker | None = None,
    context_spill: Any | None = None,
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
    @param write_failure_tracker 写工具失败跟踪
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

        from llgraph.context.context_settings import resolve_context_settings

        ctx_settings = resolve_context_settings(root)
        invoke_preserve = (
            False if ctx_settings.compress_strategy == "cursor" else None
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
    )
    if with_memory:
        from llgraph.context.chat_history_repair import ensure_agent_chat_history_sanitized

        # 仅 canonical 落盘清理（不按当前模型展开），出站修链在 prompt normalizer
        ensure_agent_chat_history_sanitized(agent, root, thread_id)

    context_block = build_workspace_context_block(root, ctx, user_message)
    if effective_message_override is not None:
        effective = effective_message_override
    else:
        effective = wrap_user_message_with_context(user_message, context_block)
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
    )
    from llgraph.display.execution_log import log_turn_end

    log_turn_end(
        root,
        thread_id=thread_id,
        with_memory=with_memory,
        agent=agent,
        tool_names=turn_result.tool_names,
        duration_sec=turn_result.duration_sec,
        compress_report=compress_report,
        spill=context_spill,
        spill_count_at_start=spill_count_at_start,
        trace_mode=trace.mode.value,
    )
    if with_memory:
        from llgraph.context.incremental_context import (
            apply_incremental_tool_prune_to_agent_state,
            format_tool_prune_report,
        )
        from llgraph.session.session_file_store import persist_agent_session

        persist_agent_session(agent, root, thread_id)
        end_prune = apply_incremental_tool_prune_to_agent_state(
            agent,
            thread_id=thread_id,
            workspace=root,
        )
        if end_prune is not None:
            from llgraph.terminal.ops_notice import ops_notice

            ops_notice(format_tool_prune_report(end_prune))
    return turn_result.text
