"""策略执行日志：压缩、token、工具、索引缓存等（JSONL 落盘）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from llgraph.context.context_compressor import CompressReport, estimate_tokens
from llgraph.config.edit_settings import load_agent_config
from llgraph.core.llm_settings import resolve_effective_model
from llgraph.display.log_retention import run_log_retention
from llgraph.session.user_storage import workspace_context_dir

EXECUTION_LOG_FILENAME = "execution.jsonl"
_DEFAULT_TAIL_LINES = 8


def execution_log_path(workspace: Path) -> Path:
    """
    执行日志路径。

    @param workspace 工作区根
    @return ~/.llgraph/context/<slug>/logs/execution.jsonl
    """
    return workspace_context_dir(workspace) / "logs" / EXECUTION_LOG_FILENAME


def resolve_execution_log_enabled(workspace: Path | None) -> bool:
    """
    是否写入执行日志。

    @param workspace 工作区根
    @return 默认 True
    """
    if workspace is None:
        return True
    cfg = load_agent_config(workspace)
    logging_cfg = cfg.get("logging") if isinstance(cfg.get("logging"), dict) else {}
    if "execution_log" in logging_cfg:
        return bool(logging_cfg.get("execution_log"))
    return True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def append_execution_event(workspace: Path, event: dict[str, Any]) -> None:
    """
    追加一条执行事件。

    @param workspace 工作区根
    @param event 事件字段（会自动补 ts、workspace）
    """
    if not resolve_execution_log_enabled(workspace):
        return
    payload = {
        "ts": _utc_now_iso(),
        "workspace": str(workspace.expanduser().resolve()),
        **event,
    }
    _append_jsonl(execution_log_path(workspace), payload)


def _usage_dict_from_mapping(usage: Any) -> dict[str, int]:
    """
    归一化网关 usage（含 LangChain input_token_details 内的 cache 字段）。

    @param usage usage_metadata 或 dict
    @return 字段 → 整型计数；含 _cache_reported=1 表示响应里出现过 cache 相关键
    """
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = dict(usage)
    else:
        raw = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            val = getattr(usage, key, None)
            if val is not None:
                raw[key] = val

    out: dict[str, int] = {}
    cache_reported = False

    def _add(key: str, val: Any) -> None:
        nonlocal cache_reported
        if isinstance(val, (int, float)) and val >= 0:
            out[key] = out.get(key, 0) + int(val)
            if key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                cache_reported = True

    for key, val in raw.items():
        if key == "input_token_details" and isinstance(val, dict):
            if "cache_read" in val:
                cache_reported = True
                _add("cache_read_input_tokens", val.get("cache_read"))
            if "cache_creation" in val:
                cache_reported = True
                _add("cache_creation_input_tokens", val.get("cache_creation"))
            for sub_key, sub_val in val.items():
                if sub_key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                    if sub_val:
                        cache_reported = True
                        _add("cache_creation_input_tokens", sub_val)
            continue
        if key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if val is not None:
                cache_reported = True
                _add(key, val)
            continue
        if isinstance(val, (int, float)) and val >= 0 and not key.startswith("_"):
            out[key] = int(val)

    if cache_reported:
        out["_cache_reported"] = 1
    return out


def extract_usage_from_messages(messages: list[Any]) -> dict[str, Any]:
    """
    从本轮 AIMessage 汇总网关 usage（含 prompt 缓存字段）。

    @param messages 完整消息列表
    @return usage 汇总与来源标记
    """
    totals: dict[str, int] = {}
    rounds = 0
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        usage_raw = getattr(msg, "usage_metadata", None)
        if usage_raw is None:
            meta = getattr(msg, "response_metadata", None) or {}
            if isinstance(meta, dict):
                usage_raw = meta.get("usage_metadata") or meta.get("usage")
        part = _usage_dict_from_mapping(usage_raw)
        if not part:
            continue
        rounds += 1
        for key, val in part.items():
            totals[key] = totals.get(key, 0) + val

    if not totals:
        return {"source": "none", "rounds": 0, "totals": {}}
    return {"source": "gateway", "rounds": rounds, "totals": totals}


def _compress_payload(report: CompressReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "ran": True,
        "before_count": report.before_count,
        "after_count": report.after_count,
        "before_tokens_est": report.before_tokens,
        "after_tokens_est": report.after_tokens,
        "saved_ratio": round(report.saved_ratio, 4),
        "archive_path": report.archive_path,
        "anchor_path": report.anchor_path,
    }


def _spill_payload(
    spill: Any | None,
    *,
    spill_count_at_start: int,
) -> dict[str, Any] | None:
    if spill is None:
        return None
    count = spill.spill_count()
    return {
        "count": count,
        "delta_count": max(0, count - spill_count_at_start),
        "bytes_on_disk": spill.spilled_bytes_on_disk(),
    }


def log_compress_event(
    workspace: Path,
    *,
    thread_id: str,
    report: CompressReport,
    trigger: str = "manual",
) -> None:
    """
    记录压缩事件。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param report 压缩报告
    @param trigger auto | manual | command
    """
    append_execution_event(
        workspace,
        {
            "event": "compress",
            "thread_id": thread_id,
            "trigger": trigger,
            "compress": _compress_payload(report),
        },
    )


def log_index_event(
    workspace: Path,
    *,
    mode: str,
    files_scanned: int,
    files_updated: int,
    chunks_written: int,
    embed_cache_enabled: bool,
    log_path: str | None = None,
    error_count: int = 0,
) -> None:
    """
    记录索引任务摘要。

    @param workspace 工作区根
    @param mode full/incremental/rebuild
    @param files_scanned 扫描文件数
    @param files_updated 更新文件数
    @param chunks_written 写入 chunk 数
    @param embed_cache_enabled 是否启用 embed SQLite 缓存
    @param log_path 索引过程日志路径
    @param error_count 错误数
    """
    append_execution_event(
        workspace,
        {
            "event": "index",
            "mode": mode,
            "files_scanned": files_scanned,
            "files_updated": files_updated,
            "chunks_written": chunks_written,
            "embed_cache_enabled": embed_cache_enabled,
            "index_log_path": log_path,
            "error_count": error_count,
        },
    )


def log_turn_end(
    workspace: Path,
    *,
    thread_id: str,
    with_memory: bool,
    agent: Any,
    tool_names: list[str],
    duration_sec: float,
    compress_report: CompressReport | None = None,
    spill: Any | None = None,
    spill_count_at_start: int = 0,
    trace_mode: str | None = None,
) -> None:
    """
    记录单轮对话结束（token、压缩、工具、落盘）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param with_memory 是否持久化会话
    @param agent LangGraph agent
    @param tool_names 本轮调用的工具名（顺序保留）
    @param duration_sec 本轮耗时
    @param compress_report 本轮前自动压缩报告
    @param spill 工具结果落盘器
    @param spill_count_at_start 本轮开始前 spill 次数
    @param trace_mode 过程展示模式
    """
    if not resolve_execution_log_enabled(workspace):
        return

    messages: list[Any] = []
    if with_memory:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = agent.get_state(config)
            messages = list((state.values or {}).get("messages") or [])
        except Exception:
            messages = []

    messages_tokens_est = estimate_tokens(messages) if messages else 0
    usage = extract_usage_from_messages(messages)
    cache_read = usage.get("totals", {}).get("cache_read_input_tokens", 0)
    cache_create = usage.get("totals", {}).get("cache_creation_input_tokens", 0)
    prompt_cache_used = cache_read > 0 or cache_create > 0

    unique_tools = list(dict.fromkeys(tool_names))

    append_execution_event(
        workspace,
        {
            "event": "turn",
            "thread_id": thread_id,
            "model": resolve_effective_model(workspace),
            "duration_sec": round(duration_sec, 3),
            "trace_mode": trace_mode,
            "tools": unique_tools,
            "tool_call_count": len(tool_names),
            "messages_tokens_est": messages_tokens_est,
            "usage": usage,
            "prompt_cache_used": prompt_cache_used,
            "compress": _compress_payload(compress_report),
            "spill": _spill_payload(
                spill,
                spill_count_at_start=spill_count_at_start,
            ),
        },
    )


def read_execution_tail(
    workspace: Path,
    *,
    limit: int = _DEFAULT_TAIL_LINES,
) -> list[dict[str, Any]]:
    """
    读取执行日志末尾若干条。

    @param workspace 工作区根
    @param limit 条数上限
    @return 解析后的事件列表
    """
    path = execution_log_path(workspace)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def format_execution_record(record: dict[str, Any]) -> str:
    """
    单行摘要。

    @param record JSON 事件
    @return 可读一行
    """
    ts = record.get("ts", "?")
    event = record.get("event", "?")
    if event == "turn":
        tools = record.get("tools") or []
        tool_part = ",".join(tools[:4]) if tools else "—"
        if len(tools) > 4:
            tool_part += f"+{len(tools) - 4}"
        usage = record.get("usage") or {}
        totals = usage.get("totals") or {}
        in_tok = totals.get("input_tokens")
        out_tok = totals.get("output_tokens")
        est = record.get("messages_tokens_est")
        cache_flag = "缓存✓" if record.get("prompt_cache_used") else "缓存—"
        usage_part = (
            f"网关 in={in_tok} out={out_tok}"
            if in_tok is not None
            else f"估算≈{est}tok"
        )
        compress = record.get("compress")
        cmp_part = ""
        if compress and compress.get("ran"):
            cmp_part = (
                f" | 压缩 {compress.get('before_tokens_est')}"
                f"→{compress.get('after_tokens_est')}"
            )
        spill = record.get("spill") or {}
        spill_part = ""
        if spill.get("delta_count"):
            spill_part = f" | spill+{spill['delta_count']}"
        return (
            f"{ts} turn [{record.get('thread_id', '?')[:8]}] "
            f"{usage_part} {cache_flag} tools={tool_part}{cmp_part}{spill_part}"
        )
    if event == "compress":
        c = record.get("compress") or {}
        return (
            f"{ts} compress [{record.get('thread_id', '?')[:8]}] "
            f"{c.get('before_tokens_est')}→{c.get('after_tokens_est')} "
            f"({record.get('trigger', '?')})"
        )
    if event == "index":
        cache = "embed缓存开" if record.get("embed_cache_enabled") else "embed缓存关"
        return (
            f"{ts} index {record.get('mode')} "
            f"chunks={record.get('chunks_written')} "
            f"更新={record.get('files_updated')} {cache}"
        )
    return f"{ts} {event}"


def format_execution_tail(workspace: Path, *, limit: int = _DEFAULT_TAIL_LINES) -> str:
    """
    格式化末尾执行日志。

    @param workspace 工作区根
    @param limit 条数
    @return 多行文本
    """
    path = execution_log_path(workspace)
    records = read_execution_tail(workspace, limit=limit)
    if not records:
        return f"执行日志为空\n路径: {path}"
    lines = [f"执行日志（最近 {len(records)} 条）: {path}", ""]
    lines.extend(format_execution_record(r) for r in records)
    return "\n".join(lines)


def startup_logging_maintenance(workspace: Path) -> None:
    """
    启动时：过期清理 + 确保日志目录存在。

    @param workspace 工作区根
    """
    execution_log_path(workspace).parent.mkdir(parents=True, exist_ok=True)
    run_log_retention(workspace, quiet=True)
