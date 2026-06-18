"""Web 会话 trace 落盘（多轮累积，供右侧 Trace 面板回放）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.session.user_storage import session_thread_dir

LAST_WEB_TRACE_FILENAME = "last_web_trace.json"
WEB_TRACE_HISTORY_FILENAME = "web_trace_history.json"
_MAX_TURNS = 80


def last_web_trace_path(workspace: Path, thread_id: str) -> Path:
    """
    @param workspace 工作区根
    @param thread_id 会话 thread
    @return last_web_trace.json 路径（兼容旧读）
    """
    return session_thread_dir(workspace, thread_id) / LAST_WEB_TRACE_FILENAME


def web_trace_history_path(workspace: Path, thread_id: str) -> Path:
    """
    @param workspace 工作区根
    @param thread_id 会话 thread
    @return web_trace_history.json 路径
    """
    return session_thread_dir(workspace, thread_id) / WEB_TRACE_HISTORY_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_turn_separator_label(ts: str, idx: int) -> str:
    """
    多轮 trace 合并时的轮次分隔标题（与 Web live turn_start 一致，本地 HH:MM:SS）。

    @param ts 落盘 UTC ISO 时间
    @param idx 轮次序号（0-based）
    @return 如「本轮 17:14:04」
    """
    raw = (ts or "").strip()
    if raw:
        try:
            normalized = raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local = dt.astimezone()
            return f"本轮 {local.strftime('%H:%M:%S')}"
        except ValueError:
            pass
    return f"本轮 {idx + 1}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_last_web_trace(
    workspace: Path,
    thread_id: str,
    *,
    log_lines: list[str],
    steps: list[dict[str, Any]],
) -> None:
    """
    追加一轮 Web trace（并更新 last_web_trace 兼容字段）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param log_lines 逐行日志
    @param steps 结构化步骤（TraceStepRecord 序列化）
    """
    if not thread_id.strip():
        return
    clean_lines = [line for line in log_lines if str(line).strip()]
    if not clean_lines and not steps:
        return

    turn = {
        "ts": _utc_now_iso(),
        "log_lines": clean_lines,
        "steps": steps,
    }

    history_path = web_trace_history_path(workspace, thread_id)
    history = _read_json(history_path) or {"turns": []}
    turns = history.get("turns")
    if not isinstance(turns, list):
        turns = []
    turns.append(turn)
    if len(turns) > _MAX_TURNS:
        turns = turns[-_MAX_TURNS:]
    _write_json(history_path, {"turns": turns})

    # 兼容旧 API / 单轮读取
    _write_json(
        last_web_trace_path(workspace, thread_id),
        {"log_lines": clean_lines, "steps": steps},
    )


def _merge_turns(turns: list[dict[str, Any]]) -> dict[str, Any]:
    merged_lines: list[str] = []
    merged_steps: list[dict[str, Any]] = []
    step_offset = 0
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        ts = str(turn.get("ts") or "").strip()
        label = _format_turn_separator_label(ts, idx)
        merged_lines.append(f"─── {label} ───")
        for line in turn.get("log_lines") or []:
            text = str(line).strip()
            if text:
                merged_lines.append(text)

        turn_steps = turn.get("steps") or []
        if not isinstance(turn_steps, list):
            continue
        local_max = 0
        for raw in turn_steps:
            if not isinstance(raw, dict):
                continue
            step = dict(raw)
            local_id = int(step.get("step_id") or 0)
            step["step_id"] = step_offset + local_id
            merged_steps.append(step)
            local_max = max(local_max, local_id)
        if local_max:
            step_offset += local_max

    return {"log_lines": merged_lines, "steps": merged_steps}


def load_last_web_trace(workspace: Path, thread_id: str) -> dict[str, Any] | None:
    """
    读取 Web trace（优先多轮累积，回退 last_web_trace.json）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @return 合并后的 log_lines + steps；无数据则 None
    """
    history = _read_json(web_trace_history_path(workspace, thread_id))
    turns = history.get("turns") if isinstance(history, dict) else None
    if isinstance(turns, list) and turns:
        merged = _merge_turns(turns)
        if merged["log_lines"] or merged["steps"]:
            return merged

    legacy = _read_json(last_web_trace_path(workspace, thread_id))
    if not legacy:
        return None
    log_lines = legacy.get("log_lines")
    steps = legacy.get("steps")
    return {
        "log_lines": log_lines if isinstance(log_lines, list) else [],
        "steps": steps if isinstance(steps, list) else [],
    }
