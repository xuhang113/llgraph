"""会话级运行结果落盘（与 trace 分离：trace 供 UI 恢复，本模块供排查终止原因）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.session.user_storage import session_thread_dir

LAST_RUN_FILENAME = "last_run.json"
RUN_LOG_FILENAME = "run_log.jsonl"


class UserCancelledError(RuntimeError):
    """Web Stop / 用户主动停止当前 ReAct 轮次。"""


def last_run_path(workspace: Path, thread_id: str) -> Path:
    """
    @param workspace 工作区根
    @param thread_id 会话 thread
    @return sessions/<thread>/last_run.json
    """
    return session_thread_dir(workspace, thread_id) / LAST_RUN_FILENAME


def run_log_path(workspace: Path, thread_id: str) -> Path:
    """
    @param workspace 工作区根
    @param thread_id 会话 thread
    @return sessions/<thread>/run_log.jsonl
    """
    return session_thread_dir(workspace, thread_id) / RUN_LOG_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _message_preview(text: str, *, limit: int = 200) -> str:
    one_line = " ".join(str(text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def trace_run_context(trace: Any | None) -> dict[str, Any]:
    """
    从 TraceSession 提取本轮运行上下文（用于日志，非 trace 回放）。

    @param trace TraceSession 或 None
    @return trace_step_count / last_trace_step_id / last_trace_step_summary / tools
    """
    if trace is None:
        return {}

    steps: list[Any] = list(getattr(trace, "last_turn_steps", None) or [])
    printer = getattr(trace, "active_printer", None)
    if not steps and printer is not None:
        steps = list(getattr(printer, "_steps", None) or [])

    tool_names: list[str] = []
    if printer is not None and getattr(printer, "_tool_names", None):
        tool_names = list(printer._tool_names)
    elif steps:
        for step in steps:
            name = getattr(step, "tool_name", None) or getattr(step, "tool", None)
            if name and name not in tool_names:
                tool_names.append(str(name))

    ctx: dict[str, Any] = {"trace_step_count": len(steps)}
    if tool_names:
        ctx["tools"] = tool_names

    if not steps:
        return ctx

    last = steps[-1]
    step_id = getattr(last, "step_id", None)
    if step_id is not None:
        ctx["last_trace_step_id"] = step_id
    summary = (
        getattr(last, "summary", None)
        or getattr(last, "title", None)
        or getattr(last, "label", None)
    )
    if summary:
        ctx["last_trace_step_summary"] = str(summary).strip()[:500]
    return ctx


def read_session_last_run(workspace: Path, thread_id: str) -> dict[str, Any] | None:
    """
    @param workspace 工作区根
    @param thread_id 会话 thread
    @return 最近一次运行结果；无文件则 None
    """
    path = last_run_path(workspace, thread_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _append_run_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def write_session_last_run(
    workspace: Path,
    thread_id: str,
    *,
    outcome: str,
    duration_sec: float,
    model: str | None = None,
    user_message: str | None = None,
    error: BaseException | None = None,
    trace_context: dict[str, Any] | None = None,
    source: str = "web",
) -> dict[str, Any]:
    """
    写入会话最近一次运行结果（last_run.json + run_log.jsonl 追加）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param outcome ok | cancelled | error | timeout
    @param duration_sec 本轮耗时
    @param model 模型 id
    @param user_message 用户输入（仅预览）
    @param error 异常（error/timeout/cancelled 时）
    @param trace_context trace_run_context 返回值
    @param source web | cli
    @return 写入的 payload
    """
    if not thread_id.strip():
        return {}

    ts = _utc_now_iso()
    payload: dict[str, Any] = {
        "ts": ts,
        "thread_id": thread_id,
        "outcome": outcome,
        "duration_sec": round(duration_sec, 3),
        "source": source,
    }
    if model:
        payload["model"] = model
    if user_message:
        payload["user_message_preview"] = _message_preview(user_message)
    if trace_context:
        payload.update(trace_context)

    if error is not None:
        message = str(error).strip() or type(error).__name__
        payload["error_type"] = type(error).__name__
        payload["error_message"] = message[:2000]

    path = last_run_path(workspace, thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_run_log(run_log_path(workspace, thread_id), payload)
    return payload
