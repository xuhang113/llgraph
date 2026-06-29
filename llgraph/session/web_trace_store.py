"""Web 会话 trace 落盘（多轮累积，供右侧 Trace 面板回放）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.session.user_storage import session_thread_dir

LAST_WEB_TRACE_FILENAME = "last_web_trace.json"
WEB_TRACE_HISTORY_FILENAME = "web_trace_history.json"
LIVE_WEB_TRACE_FILENAME = "live_web_trace.json"
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


def live_web_trace_path(workspace: Path, thread_id: str) -> Path:
    """
    @param workspace 工作区根
    @param thread_id 会话 thread
    @return live_web_trace.json 路径（执行中增量 trace）
    """
    return session_thread_dir(workspace, thread_id) / LIVE_WEB_TRACE_FILENAME


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
    incomplete: bool = False,
    stop_reason: str | None = None,
    outcome: str | None = None,
) -> None:
    """
    追加一轮 Web trace（并更新 last_web_trace 兼容字段）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param log_lines 逐行日志
    @param steps 结构化步骤（TraceStepRecord 序列化）
    @param incomplete True 表示轮次未正常完成（用户停止/异常）
    @param stop_reason 终止原因摘要（写入 history，供排查；非 UI 必需）
    @param outcome ok | cancelled | error | timeout
    """
    if not thread_id.strip():
        return
    clean_lines = [line for line in log_lines if str(line).strip()]
    if not clean_lines and not steps:
        return

    turn: dict[str, Any] = {
        "ts": _utc_now_iso(),
        "log_lines": clean_lines,
        "steps": steps,
    }
    if incomplete:
        turn["incomplete"] = True
    if stop_reason:
        turn["stop_reason"] = stop_reason
    if outcome:
        turn["outcome"] = outcome

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
    clear_live_web_trace(workspace, thread_id)


def update_live_web_trace(
    workspace: Path,
    thread_id: str,
    *,
    log_lines: list[str],
    steps: list[dict[str, Any]],
) -> None:
    """
    执行中增量落盘 trace（切换会话后可从 last-trace API 恢复）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param log_lines 当前轮逐行日志
    @param steps 当前轮结构化步骤
    """
    if not thread_id.strip():
        return
    clean_lines = [line for line in log_lines if str(line).strip()]
    if not clean_lines and not steps:
        return
    _write_json(
        live_web_trace_path(workspace, thread_id),
        {"log_lines": clean_lines, "steps": steps, "ts": _utc_now_iso()},
    )


def clear_live_web_trace(workspace: Path, thread_id: str) -> None:
    """@param workspace 工作区根 @param thread_id 会话 thread"""
    path = live_web_trace_path(workspace, thread_id)
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


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


def _append_live_to_base(
    base: dict[str, Any],
    *,
    live_lines: list[Any],
    live_steps: list[Any],
    live_ts: str,
    turn_index: int,
) -> dict[str, Any]:
    """
    将执行中的 live trace 接到已合并的历史之后（不再把整段 history 当作一轮重包）。

    @param base 已合并的 log_lines + steps
    @param live_lines 当前轮逐行日志
    @param live_steps 当前轮结构化步骤
    @param live_ts live 落盘 UTC ISO
    @param turn_index 当前轮序号（0-based，与 history turns 长度一致）
    @return 合并后的 log_lines + steps
    """
    base_lines = [str(x).strip() for x in (base.get("log_lines") or []) if str(x).strip()]
    base_steps: list[dict[str, Any]] = []
    step_offset = 0
    for raw in base.get("steps") or []:
        if isinstance(raw, dict):
            base_steps.append(dict(raw))
            step_offset = max(step_offset, int(raw.get("step_id") or 0))

    clean_live = [str(x).strip() for x in live_lines if str(x).strip()]
    label = _format_turn_separator_label(live_ts, turn_index)
    sep = f"─── {label} ───"

    out_lines = list(base_lines)
    while out_lines and out_lines[-1].startswith("─── 本轮"):
        out_lines.pop()

    if clean_live and len(out_lines) >= len(clean_live) and out_lines[-len(clean_live) :] == clean_live:
        out_steps = list(base_steps)
        if isinstance(live_steps, list):
            for raw in live_steps:
                if not isinstance(raw, dict):
                    continue
                step = dict(raw)
                sid = step_offset + int(step.get("step_id") or 0)
                if not any(int(s.get("step_id") or 0) == sid for s in out_steps):
                    step["step_id"] = sid
                    out_steps.append(step)
        return {"log_lines": out_lines, "steps": out_steps}

    if clean_live or live_steps:
        out_lines.append(sep)
        for line in clean_live:
            if not out_lines or out_lines[-1] != line:
                out_lines.append(line)

    out_steps = list(base_steps)
    if isinstance(live_steps, list):
        for raw in live_steps:
            if not isinstance(raw, dict):
                continue
            step = dict(raw)
            local_id = int(step.get("step_id") or 0)
            step["step_id"] = step_offset + local_id
            if not any(int(s.get("step_id") or 0) == step["step_id"] for s in out_steps):
                out_steps.append(step)

    return {"log_lines": out_lines, "steps": out_steps}


def load_web_trace_turns(workspace: Path, thread_id: str) -> list[dict[str, Any]]:
    """
    按用户轮次返回 trace（不合并 step_id）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @return 轮次列表，每项含 turn_index / label / steps / log_lines / live
    """
    turns_out: list[dict[str, Any]] = []

    history = _read_json(web_trace_history_path(workspace, thread_id))
    history_turns = history.get("turns") if isinstance(history, dict) else None
    if isinstance(history_turns, list):
        for idx, turn in enumerate(history_turns):
            if not isinstance(turn, dict):
                continue
            ts = str(turn.get("ts") or "").strip()
            time_label = _format_turn_separator_label(ts, idx).replace("本轮 ", "")
            steps = turn.get("steps") if isinstance(turn.get("steps"), list) else []
            log_lines = turn.get("log_lines") if isinstance(turn.get("log_lines"), list) else []
            if not steps and not log_lines:
                continue
            turns_out.append(
                {
                    "turn_index": idx + 1,
                    "label": f"第 {idx + 1} 轮 · {time_label}",
                    "ts": ts,
                    "steps": steps,
                    "log_lines": log_lines,
                    "live": False,
                }
            )

    live = _read_json(live_web_trace_path(workspace, thread_id))
    if isinstance(live, dict):
        live_steps = live.get("steps") if isinstance(live.get("steps"), list) else []
        live_lines = live.get("log_lines") if isinstance(live.get("log_lines"), list) else []
        if live_steps or live_lines:
            ts = str(live.get("ts") or "").strip()
            idx = len(turns_out)
            time_label = _format_turn_separator_label(ts, idx).replace("本轮 ", "")
            turns_out.append(
                {
                    "turn_index": idx + 1,
                    "label": f"第 {idx + 1} 轮 · {time_label}",
                    "ts": ts,
                    "steps": live_steps,
                    "log_lines": live_lines,
                    "live": True,
                }
            )

    if turns_out:
        return turns_out

    legacy = _read_json(last_web_trace_path(workspace, thread_id))
    if not legacy:
        return []
    log_lines = legacy.get("log_lines") if isinstance(legacy.get("log_lines"), list) else []
    steps = legacy.get("steps") if isinstance(legacy.get("steps"), list) else []
    if not log_lines and not steps:
        return []
    return [
        {
            "turn_index": 1,
            "label": "第 1 轮",
            "ts": "",
            "steps": steps,
            "log_lines": log_lines,
            "live": False,
        }
    ]


def load_last_web_trace(workspace: Path, thread_id: str) -> dict[str, Any] | None:
    """
    读取 Web trace（优先多轮累积，回退 last_web_trace.json）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @return 合并后的 log_lines + steps；无数据则 None
    """
    history = _read_json(web_trace_history_path(workspace, thread_id))
    turns = history.get("turns") if isinstance(history, dict) else None
    base: dict[str, Any] | None = None
    if isinstance(turns, list) and turns:
        merged = _merge_turns(turns)
        if merged["log_lines"] or merged["steps"]:
            base = merged

    if base is None:
        legacy = _read_json(last_web_trace_path(workspace, thread_id))
        if legacy:
            log_lines = legacy.get("log_lines")
            steps = legacy.get("steps")
            base = {
                "log_lines": log_lines if isinstance(log_lines, list) else [],
                "steps": steps if isinstance(steps, list) else [],
            }

    live = _read_json(live_web_trace_path(workspace, thread_id))
    live_ts = ""
    if isinstance(live, dict):
        live_ts = str(live.get("ts") or "").strip()
        live_lines = live.get("log_lines") if isinstance(live.get("log_lines"), list) else []
        live_steps = live.get("steps") if isinstance(live.get("steps"), list) else []
        if live_lines or live_steps:
            if base is None:
                out = {"log_lines": live_lines, "steps": live_steps}
                if live_ts:
                    out["live_ts"] = live_ts
                return out
            merged = _append_live_to_base(
                base,
                live_lines=live_lines,
                live_steps=live_steps,
                live_ts=live_ts,
                turn_index=len(turns) if isinstance(turns, list) else 0,
            )
            if merged["log_lines"] or merged["steps"]:
                if live_ts:
                    merged["live_ts"] = live_ts
                return merged

    if base is not None and live_ts:
        base = dict(base)
        base["live_ts"] = live_ts
    return base
