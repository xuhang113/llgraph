"""上下文压缩配置（.llgraph/agent.json context 段）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config


@dataclass(frozen=True)
class ContextSettings:
    """上下文压缩参数。"""

    max_tokens_estimate: int
    auto_compress_ratio: float
    keep_recent_turns: int
    keep_recent_token_ratio: float
    compress_model: str | None
    session_archive_on_compress: bool
    compress_retrieval_enabled: bool
    compress_retrieval_top_k: int
    compress_tool_mask_max_chars: int
    tool_result_max_chars: int
    tool_result_preview_lines: int
    spill_dir: str
    spill_enabled: bool
    spill_exempt_tools: tuple[str, ...]
    budget_source: str
    context_model_id: str
    context_fallback_max_tokens: int
    incremental_tool_prune: bool
    keep_recent_tool_messages: int
    compress_trigger_max_tokens: int | None
    session_history_search_enabled: bool
    session_history_search_top_k: int
    dispatch_keep_user_turns: int
    dispatch_min_user_turns: int
    dispatch_max_user_turns: int
    dispatch_window_token_ratio: float
    compress_strategy: str
    compress_during_react: bool
    compress_summary_chunk_chars: int
    dispatch_tool_chain_compress: bool
    dispatch_keep_full_tool_messages: int
    read_tool_result_max_chars: int
    read_file_max_bytes: int
    read_file_max_lines: int
    tool_result_preview_head_lines: int
    dispatch_dedupe_read_paths: bool
    dispatch_min_tool_rounds: int
    grep_context_lines: int
    spill_hit_context_lines: int


def is_auto_compress_strategy(strategy: str) -> bool:
    """
    是否为自动压缩/出站策略（auto；兼容旧配置 cursor）。

    @param strategy compress_strategy 配置值
    @return 是否 auto 族
    """
    return strategy.strip().lower() in ("auto", "cursor")


def normalize_compress_strategy(raw: object) -> str:
    """
    解析 compress_strategy，cursor 映射为 auto。

    @param raw 配置原值
    @return auto | legacy
    """
    if isinstance(raw, str):
        value = raw.strip().lower()
    else:
        value = "auto"
    if value in ("auto", "cursor"):
        return "auto"
    if value == "legacy":
        return "legacy"
    return "auto"


# agent.json → context._docs 与 /context 展示用（_docs 键不参与运行）
CONTEXT_CONFIG_DOCS: dict[str, str] = {
    "compress_strategy": (
        "压缩与出站策略。可选值：\n"
        "  auto（默认）— 接近满窗时用 LLM 将远早对话摘要为 <conversation-anchor>；"
        "出站 user 轮按 token 自动扩展（见 dispatch_*）。\n"
        "  legacy — 不滚动 LLM 摘要，按 keep_recent_turns / token 比例保留最近对话尾段。\n"
        "  cursor — 已废弃别名，等同 auto。"
    ),
    "dispatch_keep_user_turns": (
        "发往模型前保留的 user 轮数。0=自动（在 dispatch_min～max 与 token 预算内尽量多留）；"
        ">0=固定只保留最近 N 个 user 轮（含其间 assistant/tool 链）。"
    ),
    "dispatch_min_user_turns": "自动出站时至少保留的 user 轮数（默认 2）。",
    "dispatch_max_user_turns": "自动出站时最多保留的 user 轮数（默认 8）。",
    "dispatch_window_token_ratio": (
        "自动出站时，历史对话可用上下文上限 = max_tokens_estimate × 本比例（默认 0.35）。"
        "最近轮次 token 少时可多留 3～4 轮；轮次很长时可能只留 2 轮。"
    ),
    "compress_during_react": (
        "ReAct 单轮内工具链过长时是否中途压缩（auto 默认 true，legacy 默认 false）。"
    ),
    "auto_compress_ratio": "自动触发压缩的上下文占用比例阈值（auto 默认 0.85，legacy 默认 0.65）。",
    "keep_recent_turns": "legacy 策略压缩后至少保留的 user 轮数下限。",
    "incremental_tool_prune": "是否将较早 ToolMessage 超长输出替换为指针（默认 true）。",
    "keep_recent_tool_messages": "incremental_tool_prune 落盘会话时保留全文 ToolMessage 条数（auto 默认 4）。",
    "dispatch_tool_chain_compress": (
        "发往模型前是否压缩 tool 链：较早 ToolMessage 替换为指针，仅最近 N 条保留全文（默认 true）。"
    ),
    "dispatch_keep_full_tool_messages": (
        "dispatch_tool_chain_compress 时出站保留全文的 ToolMessage 条数（默认 2）。"
    ),
    "spill_exempt_tools": "不参与落盘的工具名；默认空（read_file/read_files 超长也会落盘+指针）。",
    "tool_result_max_chars": "grep/shell 等工具 spill 阈值（auto 默认 12000）。",
    "read_tool_result_max_chars": (
        "read_file/read_files 单独 spill 阈值（auto 默认 36000）；"
        "高于 tool_result_max_chars，减少「仅尾部预览→反复 read」。"
    ),
    "read_file_max_bytes": (
        "read_file/read_files 单文件磁盘读取字节上限（auto 默认 600000，约 6000 行×100 字符）；"
        "超过需 start_line/end_line 分段。"
    ),
    "read_file_max_lines": (
        "read_file/read_files 单次返回最大行数（auto 默认 6000，通用推荐区间 4000~8000 的中位）；"
        "超出截断并提示继续 read。"
    ),
    "tool_result_preview_head_lines": "read 落盘时除尾部外保留的开头预览行数（auto 默认 25，含 package/import）。",
    "dispatch_dedupe_read_paths": "出站时同路径旧 read 替换为短指针，仅保留最新一次全文（auto 默认 true）。",
    "dispatch_min_tool_rounds": (
        "单 user 长 ReAct 时，出站至少保留的 tool 段数；超 token 预算后裁剪更早段（auto 默认 12）。"
    ),
    "grep_context_lines": "grep_files / ripgrep 每条命中上下附加上下文行数（auto 默认 5）。",
    "spill_hit_context_lines": "read 落盘时，对历史 grep/parallel 命中行在源文件 ±N 行嵌入预览（auto 默认 100）。",
}


def format_context_config_help(workspace: Path | None = None) -> str:
    """
    context 段配置说明（/context、/config 用）。

    @param workspace 工作区根；传入时附加当前生效值
    @return 多行说明
    """
    lines = [
        "agent.json → context 配置说明",
        "（可在 context._docs 查看字段注释；_docs 不参与运行）",
        "",
    ]
    order = (
        "compress_strategy",
        "dispatch_keep_user_turns",
        "dispatch_min_user_turns",
        "dispatch_max_user_turns",
        "dispatch_window_token_ratio",
        "compress_during_react",
        "auto_compress_ratio",
        "keep_recent_turns",
        "incremental_tool_prune",
        "keep_recent_tool_messages",
    )
    for key in order:
        doc = CONTEXT_CONFIG_DOCS.get(key, "")
        if doc:
            lines.append(f"{key}:")
            for part in doc.split("\n"):
                lines.append(f"  {part}")
            lines.append("")

    if workspace is not None:
        settings = resolve_context_settings(workspace)
        dispatch_mode = (
            f"固定 {settings.dispatch_keep_user_turns} 轮"
            if settings.dispatch_keep_user_turns > 0
            else (
                f"自动 {settings.dispatch_min_user_turns}～{settings.dispatch_max_user_turns} 轮 "
                f"(预算 {int(settings.dispatch_window_token_ratio * 100)}%)"
            )
        )
        lines.extend(
            [
                "当前生效值:",
                f"  compress_strategy: {settings.compress_strategy}",
                f"  出站窗口: {dispatch_mode}",
                f"  compress_during_react: {settings.compress_during_react}",
                f"  auto_compress_ratio: {settings.auto_compress_ratio}",
            ]
        )
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class SpillSettings:
    """工具结果落盘参数（P6）。"""

    enabled: bool
    tool_result_max_chars: int
    read_tool_result_max_chars: int
    tool_result_preview_lines: int
    tool_result_preview_head_lines: int
    spill_dir: str
    spill_exempt_tools: tuple[str, ...]
    spill_hit_context_lines: int


_DEFAULT_SPILL_EXEMPT_TOOLS: tuple[str, ...] = ()


def _parse_spill_exempt_tools(ctx: dict) -> tuple[str, ...]:
    """
    解析不参与落盘的工具名列表。

    @param ctx agent.json context 段
    @return 工具名元组；未配置时默认空（read 也走 spill）
    """
    if "spill_exempt_tools" not in ctx:
        return _DEFAULT_SPILL_EXEMPT_TOOLS
    raw = ctx.get("spill_exempt_tools")
    if not isinstance(raw, list):
        return _DEFAULT_SPILL_EXEMPT_TOOLS
    return tuple(str(item).strip() for item in raw if str(item).strip())


def resolve_context_settings(workspace: Path) -> ContextSettings:
    """
    解析 context 配置。

    @param workspace 工作区根
    @return ContextSettings
    """
    cfg = load_agent_config(workspace)
    ctx = cfg.get("context") if isinstance(cfg.get("context"), dict) else {}

    budget_source = ctx.get("budget_source", "model")
    if isinstance(budget_source, str):
        budget_source = budget_source.strip().lower()
    else:
        budget_source = "model"
    # 兼容旧字段 use_model_context_window=false → 固定 config
    use_model_flag = ctx.get("use_model_context_window")
    if use_model_flag is not None:
        if isinstance(use_model_flag, str):
            use_model_flag = use_model_flag.strip().lower() not in ("0", "false", "no")
        if not use_model_flag:
            budget_source = "config"

    fallback_raw = ctx.get("context_fallback_max_tokens", 200_000)
    try:
        context_fallback = max(10_000, int(fallback_raw))
    except (TypeError, ValueError):
        context_fallback = 200_000

    config_max_raw = ctx.get("max_tokens_estimate")
    config_max: int | None = None
    if config_max_raw is not None:
        try:
            config_max = max(10_000, int(config_max_raw))
        except (TypeError, ValueError):
            config_max = None

    from llgraph.core.llm_settings import resolve_effective_model
    from llgraph.core.model_context_window import resolve_model_context_window

    model_id = resolve_effective_model(workspace)
    if budget_source == "config":
        max_tokens = config_max if config_max is not None else 120_000
        effective_source = "config"
    else:
        max_tokens, _window_src = resolve_model_context_window(
            workspace,
            model_id,
            fallback=context_fallback,
        )
        effective_source = "model"

    compress_strategy = normalize_compress_strategy(ctx.get("compress_strategy", "auto"))

    default_ratio = 0.85 if is_auto_compress_strategy(compress_strategy) else 0.65
    ratio = ctx.get("auto_compress_ratio", default_ratio)
    try:
        ratio = min(0.95, max(0.4, float(ratio)))
    except (TypeError, ValueError):
        ratio = default_ratio

    default_keep = 1 if is_auto_compress_strategy(compress_strategy) else 4
    keep = ctx.get("keep_recent_turns", default_keep)
    try:
        keep = max(2, int(keep))
    except (TypeError, ValueError):
        keep = 4

    keep_token_ratio = ctx.get("keep_recent_token_ratio", 0.25)
    try:
        keep_token_ratio = min(0.5, max(0.1, float(keep_token_ratio)))
    except (TypeError, ValueError):
        keep_token_ratio = 0.25

    model = ctx.get("compress_model")
    compress_model = str(model).strip() if model else None

    archive = ctx.get("session_archive_on_compress", True)
    if isinstance(archive, str):
        archive = archive.strip().lower() not in ("0", "false", "no")

    default_max_tool_chars = 12_000 if is_auto_compress_strategy(compress_strategy) else 40_000
    max_tool_chars = ctx.get("tool_result_max_chars", default_max_tool_chars)
    try:
        max_tool_chars = max(500, int(max_tool_chars))
    except (TypeError, ValueError):
        max_tool_chars = default_max_tool_chars

    default_read_max = 36_000 if is_auto_compress_strategy(compress_strategy) else max_tool_chars
    read_max_raw = ctx.get("read_tool_result_max_chars", default_read_max)
    try:
        read_max_chars = max(max_tool_chars, int(read_max_raw))
    except (TypeError, ValueError):
        read_max_chars = default_read_max

    default_read_file_bytes = (
        600_000 if is_auto_compress_strategy(compress_strategy) else 200_000
    )
    read_file_bytes_raw = ctx.get("read_file_max_bytes", default_read_file_bytes)
    try:
        read_file_max_bytes = max(50_000, int(read_file_bytes_raw))
    except (TypeError, ValueError):
        read_file_max_bytes = default_read_file_bytes

    default_read_file_lines = (
        6000 if is_auto_compress_strategy(compress_strategy) else 2000
    )
    read_file_lines_raw = ctx.get("read_file_max_lines", default_read_file_lines)
    try:
        read_file_max_lines = max(200, int(read_file_lines_raw))
    except (TypeError, ValueError):
        read_file_max_lines = default_read_file_lines

    preview_head = ctx.get("tool_result_preview_head_lines", 25)
    try:
        preview_head = max(0, min(80, int(preview_head)))
    except (TypeError, ValueError):
        preview_head = 25

    preview_lines = ctx.get("tool_result_preview_lines", 40)
    try:
        preview_lines = max(5, int(preview_lines))
    except (TypeError, ValueError):
        preview_lines = 40

    spill_dir = str(ctx.get("spill_dir", ".llgraph/context/tool-results")).strip()
    if not spill_dir:
        spill_dir = ".llgraph/context/tool-results"

    spill_on = ctx.get("spill_enabled", True)
    if isinstance(spill_on, str):
        spill_on = spill_on.strip().lower() not in ("0", "false", "no")

    spill_exempt_tools = _parse_spill_exempt_tools(ctx)

    retrieval_on = ctx.get("compress_retrieval_enabled", True)
    if isinstance(retrieval_on, str):
        retrieval_on = retrieval_on.strip().lower() not in ("0", "false", "no")

    retrieval_top_k = ctx.get("compress_retrieval_top_k", 5)
    try:
        retrieval_top_k = max(1, min(15, int(retrieval_top_k)))
    except (TypeError, ValueError):
        retrieval_top_k = 5

    mask_chars = ctx.get("compress_tool_mask_max_chars", 2000)
    try:
        mask_chars = max(200, min(max_tool_chars, int(mask_chars)))
    except (TypeError, ValueError):
        mask_chars = 2000

    incremental_prune = ctx.get("incremental_tool_prune", True)
    if isinstance(incremental_prune, str):
        incremental_prune = incremental_prune.strip().lower() not in ("0", "false", "no")

    default_keep_tools = 4 if is_auto_compress_strategy(compress_strategy) else 12
    keep_tools = ctx.get("keep_recent_tool_messages", default_keep_tools)
    try:
        keep_tools = max(2, int(keep_tools))
    except (TypeError, ValueError):
        keep_tools = default_keep_tools

    dispatch_chain_compress = ctx.get("dispatch_tool_chain_compress", True)
    if isinstance(dispatch_chain_compress, str):
        dispatch_chain_compress = dispatch_chain_compress.strip().lower() not in (
            "0",
            "false",
            "no",
        )

    default_dispatch_keep_tools = 4 if is_auto_compress_strategy(compress_strategy) else 2
    dispatch_keep_tools = ctx.get("dispatch_keep_full_tool_messages", default_dispatch_keep_tools)
    try:
        dispatch_keep_tools = max(1, int(dispatch_keep_tools))
    except (TypeError, ValueError):
        dispatch_keep_tools = default_dispatch_keep_tools

    dedupe_reads = ctx.get("dispatch_dedupe_read_paths", is_auto_compress_strategy(compress_strategy))
    if isinstance(dedupe_reads, str):
        dispatch_dedupe_read_paths = dedupe_reads.strip().lower() not in ("0", "false", "no")
    else:
        dispatch_dedupe_read_paths = bool(dedupe_reads)

    default_min_tool_rounds = 12 if is_auto_compress_strategy(compress_strategy) else 24
    min_tool_rounds_raw = ctx.get("dispatch_min_tool_rounds", default_min_tool_rounds)
    try:
        dispatch_min_tool_rounds = max(4, int(min_tool_rounds_raw))
    except (TypeError, ValueError):
        dispatch_min_tool_rounds = default_min_tool_rounds

    grep_ctx_raw = ctx.get("grep_context_lines", 5)
    try:
        grep_context_lines = max(0, min(20, int(grep_ctx_raw)))
    except (TypeError, ValueError):
        grep_context_lines = 5

    spill_hit_raw = ctx.get("spill_hit_context_lines", 100)
    try:
        spill_hit_context_lines = max(0, min(300, int(spill_hit_raw)))
    except (TypeError, ValueError):
        spill_hit_context_lines = 100

    trigger_cap: int | None = None
    trigger_raw = ctx.get("compress_trigger_max_tokens")
    if trigger_raw is not None:
        try:
            trigger_cap = max(10_000, int(trigger_raw))
        except (TypeError, ValueError):
            trigger_cap = None

    history_search_on = ctx.get("session_history_search_enabled", True)
    if isinstance(history_search_on, str):
        history_search_on = history_search_on.strip().lower() not in ("0", "false", "no")

    history_top_k = ctx.get("session_history_search_top_k", 8)
    try:
        history_top_k = max(1, min(20, int(history_top_k)))
    except (TypeError, ValueError):
        history_top_k = 8

    # 0 = 按 token 自动扩展保留轮数；>0 = 固定 N 轮
    default_dispatch_keep = 0 if is_auto_compress_strategy(compress_strategy) else 4
    dispatch_keep = ctx.get("dispatch_keep_user_turns", default_dispatch_keep)
    try:
        dispatch_keep = max(0, min(32, int(dispatch_keep)))
    except (TypeError, ValueError):
        dispatch_keep = default_dispatch_keep

    default_dispatch_min = 1 if is_auto_compress_strategy(compress_strategy) else 2
    dispatch_min = ctx.get("dispatch_min_user_turns", default_dispatch_min)
    try:
        dispatch_min = max(1, min(16, int(dispatch_min)))
    except (TypeError, ValueError):
        dispatch_min = default_dispatch_min

    dispatch_max = ctx.get("dispatch_max_user_turns", 8)
    try:
        dispatch_max = max(dispatch_min, min(32, int(dispatch_max)))
    except (TypeError, ValueError):
        dispatch_max = 8

    dispatch_ratio = ctx.get("dispatch_window_token_ratio", 0.35)
    try:
        dispatch_ratio = min(0.6, max(0.1, float(dispatch_ratio)))
    except (TypeError, ValueError):
        dispatch_ratio = 0.35

    during_react = ctx.get("compress_during_react", is_auto_compress_strategy(compress_strategy))
    if isinstance(during_react, str):
        compress_during_react = during_react.strip().lower() not in ("0", "false", "no")
    else:
        compress_during_react = bool(during_react)

    chunk_raw = ctx.get("compress_summary_chunk_chars", 120_000)
    try:
        compress_summary_chunk_chars = max(20_000, int(chunk_raw))
    except (TypeError, ValueError):
        compress_summary_chunk_chars = 120_000

    return ContextSettings(
        max_tokens_estimate=max_tokens,
        auto_compress_ratio=ratio,
        keep_recent_turns=keep,
        keep_recent_token_ratio=keep_token_ratio,
        compress_model=compress_model,
        session_archive_on_compress=bool(archive),
        compress_retrieval_enabled=bool(retrieval_on),
        compress_retrieval_top_k=retrieval_top_k,
        compress_tool_mask_max_chars=mask_chars,
        tool_result_max_chars=max_tool_chars,
        read_tool_result_max_chars=read_max_chars,
        read_file_max_bytes=read_file_max_bytes,
        read_file_max_lines=read_file_max_lines,
        tool_result_preview_lines=preview_lines,
        tool_result_preview_head_lines=preview_head,
        spill_dir=spill_dir,
        spill_enabled=bool(spill_on),
        spill_exempt_tools=spill_exempt_tools,
        budget_source=effective_source,
        context_model_id=model_id,
        context_fallback_max_tokens=context_fallback,
        incremental_tool_prune=bool(incremental_prune),
        keep_recent_tool_messages=keep_tools,
        compress_trigger_max_tokens=trigger_cap,
        session_history_search_enabled=bool(history_search_on),
        session_history_search_top_k=history_top_k,
        dispatch_keep_user_turns=dispatch_keep,
        dispatch_min_user_turns=dispatch_min,
        dispatch_max_user_turns=dispatch_max,
        dispatch_window_token_ratio=dispatch_ratio,
        compress_strategy=compress_strategy,
        compress_during_react=compress_during_react,
        compress_summary_chunk_chars=compress_summary_chunk_chars,
        dispatch_tool_chain_compress=bool(dispatch_chain_compress),
        dispatch_keep_full_tool_messages=dispatch_keep_tools,
        dispatch_dedupe_read_paths=dispatch_dedupe_read_paths,
        dispatch_min_tool_rounds=dispatch_min_tool_rounds,
        grep_context_lines=grep_context_lines,
        spill_hit_context_lines=spill_hit_context_lines,
    )


def resolve_spill_settings(workspace: Path) -> SpillSettings:
    """
    解析工具结果落盘配置。

    @param workspace 工作区根
    @return SpillSettings
    """
    ctx = resolve_context_settings(workspace)
    return SpillSettings(
        enabled=ctx.spill_enabled,
        tool_result_max_chars=ctx.tool_result_max_chars,
        read_tool_result_max_chars=ctx.read_tool_result_max_chars,
        tool_result_preview_lines=ctx.tool_result_preview_lines,
        tool_result_preview_head_lines=ctx.tool_result_preview_head_lines,
        spill_dir=ctx.spill_dir,
        spill_exempt_tools=ctx.spill_exempt_tools,
        spill_hit_context_lines=ctx.spill_hit_context_lines,
    )
