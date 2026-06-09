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
    compress_strategy: str
    compress_during_react: bool
    compress_summary_chunk_chars: int


@dataclass(frozen=True)
class SpillSettings:
    """工具结果落盘参数（P6）。"""

    enabled: bool
    tool_result_max_chars: int
    tool_result_preview_lines: int
    spill_dir: str
    spill_exempt_tools: tuple[str, ...]


_DEFAULT_SPILL_EXEMPT_TOOLS = ("read_file", "read_files")


def _parse_spill_exempt_tools(ctx: dict) -> tuple[str, ...]:
    """
    解析不参与落盘的工具名列表。

    @param ctx agent.json context 段
    @return 工具名元组
    """
    raw = ctx.get("spill_exempt_tools", list(_DEFAULT_SPILL_EXEMPT_TOOLS))
    if not isinstance(raw, list):
        return _DEFAULT_SPILL_EXEMPT_TOOLS
    names = tuple(str(item).strip() for item in raw if str(item).strip())
    return names if names else _DEFAULT_SPILL_EXEMPT_TOOLS


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

    strategy_raw = ctx.get("compress_strategy", "cursor")
    if isinstance(strategy_raw, str):
        compress_strategy = strategy_raw.strip().lower()
    else:
        compress_strategy = "cursor"
    if compress_strategy not in ("cursor", "legacy"):
        compress_strategy = "cursor"

    default_ratio = 0.85 if compress_strategy == "cursor" else 0.65
    ratio = ctx.get("auto_compress_ratio", default_ratio)
    try:
        ratio = min(0.95, max(0.4, float(ratio)))
    except (TypeError, ValueError):
        ratio = default_ratio

    default_keep = 1 if compress_strategy == "cursor" else 4
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

    max_tool_chars = ctx.get("tool_result_max_chars", 40_000)
    try:
        max_tool_chars = max(500, int(max_tool_chars))
    except (TypeError, ValueError):
        max_tool_chars = 40_000

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

    keep_tools = ctx.get("keep_recent_tool_messages", 12)
    try:
        keep_tools = max(2, int(keep_tools))
    except (TypeError, ValueError):
        keep_tools = 12

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

    default_dispatch_keep = 2 if compress_strategy == "cursor" else 4
    dispatch_keep = ctx.get("dispatch_keep_user_turns", default_dispatch_keep)
    try:
        dispatch_keep = max(0, min(32, int(dispatch_keep)))
    except (TypeError, ValueError):
        dispatch_keep = default_dispatch_keep

    during_react = ctx.get("compress_during_react", compress_strategy == "cursor")
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
        tool_result_preview_lines=preview_lines,
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
        compress_strategy=compress_strategy,
        compress_during_react=compress_during_react,
        compress_summary_chunk_chars=compress_summary_chunk_chars,
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
        tool_result_preview_lines=ctx.tool_result_preview_lines,
        spill_dir=ctx.spill_dir,
        spill_exempt_tools=ctx.spill_exempt_tools,
    )
