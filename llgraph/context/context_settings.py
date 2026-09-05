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
    compress_tool_mask_max_chars: int
    read_tool_mask_max_chars: int
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
    compress_strategy: str
    compress_during_react: bool
    compress_summary_chunk_chars: int
    dispatch_tool_chain_compress: bool
    dispatch_keep_full_tool_messages: int
    dispatch_full_tool_hysteresis: float
    dispatch_full_tool_budget_tokens: int
    dispatch_compact_low_ratio: float
    read_tool_result_max_chars: int
    read_file_max_bytes: int
    read_file_max_lines: int
    tool_result_preview_head_lines: int
    dispatch_dedupe_read_paths: bool
    grep_context_lines: int
    grep_max_inline_chars: int
    spill_hit_context_lines: int
    tool_prune_token_ratio: float
    protect_cited_tool_messages: bool
    max_protected_cited_tool_messages: int


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
        "压缩策略。可选值：\n"
        "  auto（默认）— 接近满窗时用 LLM 将远早对话摘要为 <conversation-anchor>；"
        "出站不裁 user 轮，超长工具结果用指针/预览。\n"
        "  legacy — 不滚动 LLM 摘要，按 keep_recent_turns / token 比例保留最近对话尾段。\n"
        "  cursor — 已废弃别名，等同 auto。"
    ),
    "compress_during_react": (
        "ReAct 单轮内工具链过长时是否中途压缩（auto 默认 true，legacy 默认 false）。"
    ),
    "auto_compress_ratio": "自动触发压缩的上下文占用比例阈值（auto 默认 0.85，legacy 默认 0.65）。",
    "keep_recent_turns": "legacy 策略压缩后至少保留的 user 轮数下限。",
    "incremental_tool_prune": "是否将较早 ToolMessage 超长输出替换为指针（默认 true）。",
    "keep_recent_tool_messages": "incremental_tool_prune 落盘会话时保留全文 ToolMessage 条数（auto 默认 6）。",
    "dispatch_tool_chain_compress": (
        "发往模型前是否压缩 tool 链：较早 grep/read 替换为指针，仅最近 N 条重结果保留全文（默认 true）。"
        "出站始终按 recency 压缩（不对齐满窗压力）；写入成功快照按路径钉住最新一份。"
    ),
    "dispatch_keep_full_tool_messages": (
        "dispatch_tool_chain_compress 触发压缩后，出站保留全文的**重** ToolMessage 条数"
        "（低水位，auto 默认 6）；短指针/重复拦截不占名额。"
    ),
    "dispatch_full_tool_hysteresis": (
        "出站压缩的条数滞回倍数（默认 2.5）：全文重结果达到 "
        "dispatch_keep_full_tool_messages × 该倍数才压缩一次，压到低水位。\n"
        "  Anthropic prompt cache 按精确前缀命中，每步改写历史都会击穿缓存；"
        "滞回让压缩变成低频纪元事件，纪元内出站字节不变、整段命中缓存。\n"
        "  设为 1.0 即退化为旧的 recency 滑窗（每步都压、每步都击穿）。"
    ),
    "dispatch_full_tool_budget_tokens": (
        "出站保留全文的**重** ToolMessage 估算 token 高水位；"
        "默认按模型窗口取 12%（夹在 8k~48k）。超过即压缩到 "
        "dispatch_compact_low_ratio × 该值。"
    ),
    "dispatch_compact_low_ratio": (
        "触发压缩后压到的低水位比例（默认 0.4）；比例越小压缩越狠、纪元越长、缓存命中越久。"
    ),
    "spill_exempt_tools": "不参与落盘的工具名；默认空（read_file/read_files 超长也会落盘+指针）。",
    "tool_result_max_chars": "grep/shell 等工具 spill 阈值（auto 默认 12000）。",
    "read_tool_result_max_chars": (
        "read_file/read_files 单独 spill 阈值（auto 默认 36000）；"
        "高于 tool_result_max_chars，减少「仅尾部预览→反复 read」。"
    ),
    "read_file_max_bytes": (
        "read_file/read_files 单文件磁盘读取字节上限（auto 默认 600000）；"
        "超过需 start_line/end_line 分段。"
    ),
    "read_file_max_lines": (
        "read_file/read_files 明确行段时单次返回最大行数（默认 2000，对齐 Claude Code）；"
        "未指定行段的大文件会先折叠为大纲+命中窗，不走到此上限。"
    ),
    "tool_result_preview_head_lines": "read 落盘/归档时保留的开头预览行数（auto 默认 25，含 package/import）。",
    "tool_result_preview_lines": "read 落盘/归档时保留的末尾预览行数（auto 默认 40）。",
    "compress_tool_mask_max_chars": "incremental_tool_prune 时非 read 工具超过此长度则替换为指针（auto 默认 6000）。",
    "read_tool_mask_max_chars": (
        "read_file/read_files 在 incremental 与出站归档前的保留全文阈值（auto 默认 12000）；"
        "超过后压缩为带 head/tail 预览的指针，避免仅一行路径导致误判。"
    ),
    "dispatch_dedupe_read_paths": "出站时同路径旧 read 替换为短指针，仅保留最新一次全文（auto 默认 true）。",
    "protect_cited_tool_messages": (
        "被后续 AI 结论/推理以 path:line 或文件名引用过的历史 ToolMessage，"
        "优先保留全文、延后裁剪（非满窗压力时生效；默认 true）。"
    ),
    "max_protected_cited_tool_messages": (
        "protect_cited_tool_messages 时，超出 recency 窗口额外保护的被引用条数上限（默认 8）。"
    ),
    "grep_context_lines": "grep_files / ripgrep 每条命中上下附加上下文行数（auto 默认 5）。",
    "grep_max_inline_chars": (
        "grep/search 类工具结果内联字符上限（auto 默认 48000）；"
        "检索工具不落盘，超长时保留命中块预览并内联截断。"
    ),
    "spill_hit_context_lines": "read 落盘时，对历史 grep/parallel 命中行在源文件 ±N 行嵌入预览（auto 默认 100）。",
    "tool_prune_token_ratio": (
        "历史 ToolMessage **checkpoint/落盘** mask 起始比例（相对 LLM 压缩阈值，auto 默认 0.7）；"
        "低于该比例时不改写会话里的历史 tool。出站 dispatch 裁剪不受此项门控。"
    ),
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
        "compress_during_react",
        "auto_compress_ratio",
        "keep_recent_turns",
        "incremental_tool_prune",
        "keep_recent_tool_messages",
        "dispatch_tool_chain_compress",
        "dispatch_keep_full_tool_messages",
        "dispatch_full_tool_hysteresis",
        "dispatch_full_tool_budget_tokens",
        "dispatch_compact_low_ratio",
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
        lines.extend(
            [
                "当前生效值:",
                f"  compress_strategy: {settings.compress_strategy}",
                f"  compress_during_react: {settings.compress_during_react}",
                f"  auto_compress_ratio: {settings.auto_compress_ratio}",
                f"  dispatch_tool_chain_compress: {settings.dispatch_tool_chain_compress}",
                f"  dispatch_keep_full_tool_messages: {settings.dispatch_keep_full_tool_messages}",
                f"  dispatch_full_tool_hysteresis: {settings.dispatch_full_tool_hysteresis}",
                f"  dispatch_full_tool_budget_tokens: {settings.dispatch_full_tool_budget_tokens}",
                f"  dispatch_compact_low_ratio: {settings.dispatch_compact_low_ratio}",
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
    grep_max_inline_chars: int
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

    default_read_file_lines = 2000
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

    default_mask_chars = 6000 if is_auto_compress_strategy(compress_strategy) else 2000
    mask_chars = ctx.get("compress_tool_mask_max_chars", default_mask_chars)
    try:
        mask_chars = max(200, min(max_tool_chars, int(mask_chars)))
    except (TypeError, ValueError):
        mask_chars = default_mask_chars

    default_read_mask = 12_000 if is_auto_compress_strategy(compress_strategy) else max(mask_chars, 8000)
    read_mask_raw = ctx.get("read_tool_mask_max_chars", default_read_mask)
    try:
        read_tool_mask_max_chars = max(mask_chars, int(read_mask_raw))
    except (TypeError, ValueError):
        read_tool_mask_max_chars = default_read_mask

    incremental_prune = ctx.get("incremental_tool_prune", True)
    if isinstance(incremental_prune, str):
        incremental_prune = incremental_prune.strip().lower() not in ("0", "false", "no")

    default_keep_tools = 6 if is_auto_compress_strategy(compress_strategy) else 12
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

    default_dispatch_keep_tools = 6 if is_auto_compress_strategy(compress_strategy) else 2
    dispatch_keep_tools = ctx.get("dispatch_keep_full_tool_messages", default_dispatch_keep_tools)
    try:
        dispatch_keep_tools = max(1, int(dispatch_keep_tools))
    except (TypeError, ValueError):
        dispatch_keep_tools = default_dispatch_keep_tools

    hysteresis_raw = ctx.get("dispatch_full_tool_hysteresis", 2.5)
    try:
        dispatch_full_tool_hysteresis = min(8.0, max(1.0, float(hysteresis_raw)))
    except (TypeError, ValueError):
        dispatch_full_tool_hysteresis = 2.5

    default_full_tool_budget = min(48_000, max(8_000, int(max_tokens * 0.12)))
    budget_raw = ctx.get("dispatch_full_tool_budget_tokens", default_full_tool_budget)
    try:
        dispatch_full_tool_budget_tokens = max(2_000, int(budget_raw))
    except (TypeError, ValueError):
        dispatch_full_tool_budget_tokens = default_full_tool_budget

    low_ratio_raw = ctx.get("dispatch_compact_low_ratio", 0.4)
    try:
        dispatch_compact_low_ratio = min(0.95, max(0.05, float(low_ratio_raw)))
    except (TypeError, ValueError):
        dispatch_compact_low_ratio = 0.4

    dedupe_reads = ctx.get("dispatch_dedupe_read_paths", is_auto_compress_strategy(compress_strategy))
    if isinstance(dedupe_reads, str):
        dispatch_dedupe_read_paths = dedupe_reads.strip().lower() not in ("0", "false", "no")
    else:
        dispatch_dedupe_read_paths = bool(dedupe_reads)

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

    grep_inline_raw = ctx.get("grep_max_inline_chars", 48_000)
    try:
        grep_max_inline_chars = max(8_000, min(200_000, int(grep_inline_raw)))
    except (TypeError, ValueError):
        grep_max_inline_chars = 48_000

    tool_prune_ratio_raw = ctx.get("tool_prune_token_ratio", 0.7)
    try:
        tool_prune_token_ratio = min(0.95, max(0.0, float(tool_prune_ratio_raw)))
    except (TypeError, ValueError):
        tool_prune_token_ratio = 0.7

    protect_cited_raw = ctx.get("protect_cited_tool_messages", True)
    if isinstance(protect_cited_raw, str):
        protect_cited_tool_messages = protect_cited_raw.strip().lower() not in ("0", "false", "no")
    else:
        protect_cited_tool_messages = bool(protect_cited_raw)

    max_protected_cited_raw = ctx.get("max_protected_cited_tool_messages", 8)
    try:
        max_protected_cited_tool_messages = max(0, min(64, int(max_protected_cited_raw)))
    except (TypeError, ValueError):
        max_protected_cited_tool_messages = 8

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
        compress_tool_mask_max_chars=mask_chars,
        read_tool_mask_max_chars=read_tool_mask_max_chars,
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
        compress_strategy=compress_strategy,
        compress_during_react=compress_during_react,
        compress_summary_chunk_chars=compress_summary_chunk_chars,
        dispatch_tool_chain_compress=bool(dispatch_chain_compress),
        dispatch_keep_full_tool_messages=dispatch_keep_tools,
        dispatch_full_tool_hysteresis=dispatch_full_tool_hysteresis,
        dispatch_full_tool_budget_tokens=dispatch_full_tool_budget_tokens,
        dispatch_compact_low_ratio=dispatch_compact_low_ratio,
        dispatch_dedupe_read_paths=dispatch_dedupe_read_paths,
        grep_context_lines=grep_context_lines,
        grep_max_inline_chars=grep_max_inline_chars,
        spill_hit_context_lines=spill_hit_context_lines,
        tool_prune_token_ratio=tool_prune_token_ratio,
        protect_cited_tool_messages=protect_cited_tool_messages,
        max_protected_cited_tool_messages=max_protected_cited_tool_messages,
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
        grep_max_inline_chars=ctx.grep_max_inline_chars,
        spill_hit_context_lines=ctx.spill_hit_context_lines,
    )
