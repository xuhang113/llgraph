"""终端追踪输出：/trace 四档（all / steps / reply / none）。"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from llgraph.display.execution_log import _usage_dict_from_mapping

DEFAULT_PREVIEW_LINES = 4
STEP_INLINE_PREVIEW_LINES = 3
_ALL_TOOL_OUTPUT_LINES = 12
_MAX_LINE_WIDTH = 120
_TOOL_ARGS_PREVIEW = 200
_TOOL_ARGS_PREVIEW_ALL = 500


class TraceMode(str, Enum):
    """过程展示档位。"""

    ALL = "all"
    """完整过程（规划详情、工具参数与输出，对应截图效果）。"""

    STEPS = "steps"
    """展示步骤（折叠摘要，默认）。"""

    REPLY = "reply"
    """不展示步骤，仅流式输出最终回复。"""

    NONE = "none"
    """都不展示（无过程行，仅最终回复文本）。"""


TRACE_MODE_LABELS: dict[TraceMode, str] = {
    TraceMode.ALL: "完整过程（规划+工具详情）",
    TraceMode.STEPS: "展示步骤（折叠摘要）",
    TraceMode.REPLY: "仅回复（不展示步骤）",
    TraceMode.NONE: "都不展示",
}


def parse_trace_mode(name: str) -> TraceMode | None:
    """
    解析 /trace 参数。

    @param name 模式名或别名
    @return 对应 TraceMode，无法识别时返回 None
    """
    key = name.strip().lower()
    aliases = {
        "all": TraceMode.ALL,
        "full": TraceMode.ALL,
        "完整": TraceMode.ALL,
        "全部": TraceMode.ALL,
        "steps": TraceMode.STEPS,
        "step": TraceMode.STEPS,
        "步骤": TraceMode.STEPS,
        "展示步骤": TraceMode.STEPS,
        "reply": TraceMode.REPLY,
        "off": TraceMode.REPLY,
        "回复": TraceMode.REPLY,
        "仅回复": TraceMode.REPLY,
        "不展示": TraceMode.REPLY,
        "none": TraceMode.NONE,
        "quiet": TraceMode.NONE,
        "静默": TraceMode.NONE,
        "都不展示": TraceMode.NONE,
    }
    return aliases.get(key)


from llgraph.terminal.style import indent_line, sty, sty_sgr as _c

_TRACE_L1 = indent_line(1)
_TRACE_L2 = indent_line(2)
_TRACE_L3 = indent_line(3)


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def _clip_line(line: str, width: int = _MAX_LINE_WIDTH) -> str:
    line = line.replace("\r", "").replace("\t", " ")
    if len(line) <= width:
        return line
    return line[: width - 1] + "…"


def _format_tool_args(args: Any, *, verbose: bool = False) -> str:
    try:
        raw = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        raw = str(args)
    limit = _TOOL_ARGS_PREVIEW_ALL if verbose else _TOOL_ARGS_PREVIEW
    if len(raw) > limit:
        return raw[: limit - 1] + "…"
    return raw


_PATH_TRACE_MAX_LEN = 48


def _short_path_for_trace(path_text: str, *, max_len: int = _PATH_TRACE_MAX_LEN) -> str:
    """
    Trace 摘要中的 path 缩短：保留仓库名前缀 + 末段，避免 …/b 与 ../b 混淆。

    @param path_text 完整 path
    @param max_len 超过则缩短
    @return 缩短后的 path
    """
    text = path_text.strip()
    if len(text) <= max_len:
        return text
    normalized = text.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return text[: max_len - 1] + "…"
    name = parts[-1]
    if len(parts) >= 2:
        return f"{parts[0]}/…/{name}"
    return f"…/{name}"


def _short_tool_target(args: Any) -> str:
    """
    工具调用摘要中的关键参数（路径、查询等）。

    @param args 工具参数字典
    @return 短描述；无则空串
    """
    if not isinstance(args, dict):
        return ""
    for key in ("path", "query", "pattern", "command", "url"):
        val = args.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        text = val.strip()
        if key == "path" and len(text) > _PATH_TRACE_MAX_LEN:
            shortened = _short_path_for_trace(text)
            if shortened:
                return shortened
        if len(text) > 48:
            return text[:45] + "…"
        return text
    return ""


def _format_planned_tools_summary(tool_calls: list, *, verbose: bool = False) -> str:
    """
    模型决策步摘要：强调「拟调用」、尚未执行。

    @param tool_calls AIMessage.tool_calls
    @param verbose 完整模式用计数摘要
    @return 如 拟调用 read_file(…/sandbox.json)
    """
    if not tool_calls:
        return "无工具调用"
    if verbose:
        return f"{len(tool_calls)} 个工具"
    parts: list[str] = []
    for call in tool_calls:
        name = call.get("name", "?")
        target = _short_tool_target(call.get("args") or {})
        if target:
            parts.append(f"{name}({target})")
        else:
            parts.append(str(name))
    if len(parts) == 1:
        return f"拟调用 {parts[0]}"
    joined = ", ".join(parts[:3])
    if len(parts) > 3:
        joined += f" 等 {len(parts)} 个"
    return f"拟调用 {joined}"


def _format_turn_skills_line(
    workspace: Path | None,
    context_session: Any | None,
    user_message: str,
) -> str:
    """
    本回合 /skill 手动启用技能一行摘要。

    @param workspace 工作区根
    @param context_session Rule/Skill 会话状态
    @param user_message 保留参数（自动匹配已默认关闭）
    @return 如 ⭐ 本回合技能: tracking（/skill）；无则空串
    """
    if workspace is None or context_session is None:
        return ""
    if not context_session.active_skills:
        return ""
    labels = [f"{name}（/skill）" for name in context_session.active_skills]
    return "⭐ 本会话技能: " + ", ".join(labels)


def _short_tool_target(args: Any) -> str:
    """
    工具调用摘要中的关键参数（路径、查询等）。

    @param args 工具参数字典
    @return 短描述；无则空串
    """
    if not isinstance(args, dict):
        return ""
    for key in ("path", "query", "pattern", "command", "url"):
        val = args.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        text = val.strip()
        if key == "path" and len(text) > _PATH_TRACE_MAX_LEN:
            shortened = _short_path_for_trace(text)
            if shortened:
                return shortened
        if len(text) > 44:
            return text[:41] + "…"
        return text
    return ""


def _format_planned_tools_summary(tool_calls: list, *, verbose: bool = False) -> str:
    """
    模型决策步摘要：强调「拟调用」、尚未执行。

    @param tool_calls LangGraph tool_calls
    @param verbose 是否 verbose 模式
    @return 折叠行摘要
    """
    if not tool_calls:
        return "无工具调用"
    if verbose:
        return f"{len(tool_calls)} 个工具"
    parts: list[str] = []
    for call in tool_calls:
        name = call.get("name", "?")
        target = _short_tool_target(call.get("args") or {})
        if target:
            parts.append(f"{name}({target})")
        else:
            parts.append(str(name))
    if len(parts) == 1:
        return f"拟调用 {parts[0]}"
    joined = ", ".join(parts[:3])
    if len(parts) > 3:
        joined += f" 等 {len(parts)} 个"
    return f"拟调用 {joined}"


def _tool_output_looks_like_error(text: str) -> bool:
    """工具返回是否像失败/校验错误（trace 展示全文）。"""
    if not text.strip():
        return False
    lowered = text.lower()
    markers = (
        "错误:",
        "缺少必填",
        "validation error",
        "field required",
        "未找到 old_string",
    )
    return any(m in text or m in lowered for m in markers)


@dataclass
class StepUsage:
    """单步模型调用 token 与 prompt cache 用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_reported: bool = False


@dataclass
class TraceStepRecord:
    """折叠模式下可展开的步骤记录。"""

    step_id: int
    kind: str
    title: str
    elapsed: float
    summary: str
    body_lines: list[str] = field(default_factory=list)
    usage: StepUsage | None = None


def _format_token_amount(tokens: int) -> str:
    """
    格式化 token 数量（K / M）。

    @param tokens token 数
    @return 如 456、12.3K、1.05M
    """
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}K"
    return str(tokens)


def _extract_usage_from_ai_message(msg: AIMessage | AIMessageChunk) -> StepUsage | None:
    """
    从单条 AIMessage 提取网关 usage（含 cache 字段）。

    @param msg 模型消息
    @return 本步用量，无 metadata 时返回 None
    """
    usage_raw = getattr(msg, "usage_metadata", None)
    if usage_raw is None:
        meta = getattr(msg, "response_metadata", None) or {}
        if isinstance(meta, dict):
            usage_raw = meta.get("usage_metadata") or meta.get("usage")
    part = _usage_dict_from_mapping(usage_raw)
    if not part:
        return None
    cache_reported = bool(part.pop("_cache_reported", 0))
    return StepUsage(
        input_tokens=part.get("input_tokens", 0),
        output_tokens=part.get("output_tokens", 0),
        cache_read_input_tokens=part.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=part.get("cache_creation_input_tokens", 0),
        cache_reported=cache_reported,
    )


def _format_step_usage_inline(usage: StepUsage) -> str:
    """
    步骤摘要行上的 token / cache 文案。

    @param usage 本步用量
    @return 单行摘要，如 token in 12.3K out 456 · cache 读 8.2K
    """
    parts: list[str] = []
    if usage.input_tokens or usage.output_tokens:
        parts.append(
            "token in "
            f"{_format_token_amount(usage.input_tokens)} "
            f"out {_format_token_amount(usage.output_tokens)}"
        )
    cache_read = usage.cache_read_input_tokens
    cache_create = usage.cache_creation_input_tokens
    if cache_read or cache_create:
        cache_bits: list[str] = []
        if cache_read:
            cache_bits.append(f"读 {_format_token_amount(cache_read)}")
        if cache_create:
            cache_bits.append(f"写 {_format_token_amount(cache_create)}")
        parts.append(f"cache {' '.join(cache_bits)}")
    elif usage.input_tokens or usage.output_tokens:
        if usage.cache_reported:
            parts.append("cache 无命中")
        else:
            parts.append("cache 未上报")
    return " · ".join(parts)


@dataclass
class TraceSession:
    """交互会话的过程展示配置（/trace 可运行时切换）。"""

    mode: TraceMode = TraceMode.STEPS
    preview_lines: int = DEFAULT_PREVIEW_LINES
    show_step_tokens: bool = True
    last_turn_steps: list[TraceStepRecord] = field(default_factory=list)
    last_turn_raw_reply: str = ""
    trace_sink: Any = None

    def shows_process(self) -> bool:
        """是否展示规划/工具过程（steps 折叠或 all 完整）。"""
        return self.mode in (TraceMode.STEPS, TraceMode.ALL)

    def is_verbose(self) -> bool:
        """是否完整展示（截图同款）。"""
        return self.mode == TraceMode.ALL

    def shows_reply_stream(self) -> bool:
        """是否流式打印最终回复。"""
        return self.mode in (TraceMode.STEPS, TraceMode.ALL, TraceMode.REPLY)

    def is_silent(self) -> bool:
        """是否完全不展示过程。"""
        return self.mode == TraceMode.NONE


def print_invoke_prelude(trace_session: TraceSession | None) -> None:
    """
    Agent 轮次开始前立即输出，避免大会话压缩/修链期间终端看似无响应。

    @param trace_session 过程展示配置
    """
    trace = trace_session or TraceSession()
    if trace.is_silent():
        return
    if trace.mode == TraceMode.REPLY:
        print(_c(f"[{_timestamp()}] ▶ 处理中…", "33"), flush=True)
        return
    print(
        _c(f"[{_timestamp()}] ", "90")
        + _c("▶ 准备中…", "33")
        + _c("  加载历史 / 压缩上下文 / 修链（大会话可能需 1～2 分钟）", "90"),
        flush=True,
    )


def print_command_prelude(label: str, *, detail: str = "") -> None:
    """
    内置命令（/review 等）阻塞执行前的即时提示。

    @param label 命令简述
    @param detail 补充说明
    """
    suffix = f"  {detail}" if detail else ""
    print(
        _c(f"[{_timestamp()}] ", "90") + _c(f"▶ {label}", "33") + _c(suffix, "90"),
        flush=True,
    )


def _trace_sink_is_terminal(session: TraceSession) -> bool:
    """
    当前是否为经典终端 sink（保留 ANSI）。

    @param session 追踪会话
    @return 是否终端模式
    """
    sink = session.trace_sink
    if sink is None:
        return True
    return bool(getattr(sink, "preserves_ansi", False))


def _trace_line(session: TraceSession, text: str) -> None:
    """
    输出一行到 trace sink（终端保留 ANSI）。

    @param session 追踪会话
    @param text 文本
    """
    sink = session.trace_sink
    payload = text
    if sink is not None and not getattr(sink, "preserves_ansi", False):
        from llgraph.display.trace_sink import strip_ansi

        payload = strip_ansi(text)
    if sink is None:
        from llgraph.terminal.style import color_enabled
        from llgraph.terminal.terminal_theme import _has_ansi, colorize_terminal_text

        if color_enabled() and payload.strip() and not _has_ansi(payload):
            payload = colorize_terminal_text(payload)
        print(payload, flush=True)
        return
    sink.line(payload)


def _step_expand_hint(session: TraceSession, step_id: int) -> str:
    """
    步骤行末尾展开提示。

    @param session 追踪会话
    @param step_id 步骤编号
    @return 带 ANSI 的提示片段
    """
    del session
    return _c(f"  · /trace step {step_id}", "90")


def _trace_stream(session: TraceSession, text: str) -> None:
    """
    流式输出到 trace sink。

    @param session 追踪会话
    @param text 文本块
    """
    sink = session.trace_sink
    if sink is None:
        if text:
            print(text, end="", flush=True)
        return
    sink.stream(text)


def _trace_stream_end(session: TraceSession) -> None:
    """
    流式段落结束。

    @param session 追踪会话
    """
    sink = session.trace_sink
    if sink is None:
        print(flush=True)
        return
    sink.stream_end()


LAST_TRACE_SESSION: TraceSession | None = None


@dataclass
class TurnRunResult:
    """单轮执行结果（供执行日志与调用方）。"""

    text: str
    raw_text: str = ""
    tool_names: list[str] = field(default_factory=list)
    duration_sec: float = 0.0


class TurnTracePrinter:
    """按 TraceMode 打印单轮追踪。"""

    def __init__(self, session: TraceSession) -> None:
        self._session = session
        self._turn_start = time.perf_counter()
        self._step_start = time.perf_counter()
        self._step_index = 0
        self._final_text: str = ""
        self._last_thinking_text: str = ""
        self._printed_final_header = False
        self._streamed_reply = False
        self._tool_names: list[str] = []
        self._steps: list[TraceStepRecord] = []
        self._pending_usage: StepUsage | None = None
        from llgraph.survey.survey_prompt import SurveyStreamFilter

        self._survey_filter = SurveyStreamFilter()
        self._stream_open = False

    def _line(self, text: str = "") -> None:
        # 步骤行须独占一行：先结束未换行的流式片段（避免【规划】与 ▶ #N 窜行）
        if self._stream_open:
            self._stream_end()
        _trace_line(self._session, text)

    def _stream(self, text: str) -> None:
        if text:
            self._stream_open = True
        _trace_stream(self._session, text)

    def _stream_end(self) -> None:
        if self._stream_open:
            _trace_stream_end(self._session)
            self._stream_open = False

    @property
    def preview_lines(self) -> int:
        return self._session.preview_lines

    def _register_step(
        self,
        kind: str,
        title: str,
        elapsed: float,
        summary: str,
        *,
        body: str = "",
        body_lines: list[str] | None = None,
        usage: StepUsage | None = None,
    ) -> int:
        self._step_index += 1
        lines = body_lines if body_lines is not None else (body.splitlines() if body else [])
        self._steps.append(
            TraceStepRecord(
                step_id=self._step_index,
                kind=kind,
                title=title,
                elapsed=elapsed,
                summary=summary,
                body_lines=lines,
                usage=usage,
            )
        )
        return self._step_index

    def absorb_usage_from_chunk(self, msg_chunk: Any) -> None:
        """
        从流式 AIMessageChunk 合并 usage（网关常在末包返回）。

        @param msg_chunk 流式块
        """
        usage = _extract_usage_from_ai_message(msg_chunk)
        if usage is not None:
            self._pending_usage = usage

    def _resolve_step_usage(self, usage: StepUsage | None) -> StepUsage | None:
        """
        解析本步 token：显式传入优先，否则用流式缓存。

        @param usage 调用方传入
        @return 本步用量
        """
        if usage is not None:
            self._pending_usage = None
            return usage
        pending = self._pending_usage
        self._pending_usage = None
        return pending

    def _format_step_token_suffix(self, usage: StepUsage | None) -> str:
        """
        步骤行 token 后缀（默认展示）。

        @param usage 本步用量
        @return ANSI 后缀或空
        """
        if not self._session.show_step_tokens:
            return ""
        if usage is not None:
            inline = _format_step_usage_inline(usage)
            if inline:
                return "  " + _c(inline, "35")
        return ""

    def _print_step_inline_detail(self, step_id: int) -> None:
        """
        折叠模式下步骤行下方缩进预览（无需再输 /trace step）。

        @param step_id 步骤编号
        """
        if self._session.is_verbose():
            return
        step = next((s for s in self._steps if s.step_id == step_id), None)
        if step is None or not step.body_lines:
            return
        limit = STEP_INLINE_PREVIEW_LINES
        for line in step.body_lines[:limit]:
            clipped = _clip_line(line)
            if clipped.lstrip().startswith("【规划】"):
                self._line(
                    _c(f"{_TRACE_L2}│ ", "90")
                    + _c("规划 ", "35")
                    + _c(clipped, "37"),
                )
            else:
                self._line(_c(f"{_TRACE_L2}│ {_clip_line(line)}", "90"))
        rest = len(step.body_lines) - min(len(step.body_lines), limit)
        if rest > 0:
            self._line(
                _c(
                    f"{_TRACE_L2}│ … 还有 {rest} 行 · 输入 {step_id} 或 /trace step {step_id} 看全部",
                    "90",
                ),
            )

    def _print_preview_block(self, lines: list[str], *, limit: int | None = None) -> None:
        max_lines = limit if limit is not None else self.preview_lines
        for line in lines[:max_lines]:
            self._line(_c(f"{_TRACE_L3}{_clip_line(line)}", "90"))
        hidden = len(lines) - min(len(lines), max_lines)
        if hidden > 0:
            self._line(_c(f"{_TRACE_L3}… 还有 {hidden} 行", "90"))

    def _print_step_summary(
        self,
        step_id: int,
        title: str,
        elapsed: float,
        summary: str,
        *,
        usage: StepUsage | None = None,
    ) -> None:
        resolved = usage
        if resolved is None and self._steps:
            last = self._steps[-1]
            if last.step_id == step_id:
                resolved = last.usage
        self._line(
            _c(f"[{_timestamp()}] ", "90")
            + _c(f"▶ #{step_id} {title}", "32")
            + _c(f"  ({_format_duration(elapsed)})", "90")
            + f"  {summary}"
            + self._format_step_token_suffix(resolved)
            + _step_expand_hint(self._session, step_id),
        )
        self._print_step_inline_detail(step_id)
        sink = self._session.trace_sink
        if sink is not None and self._steps:
            sink.step_added(self._steps[-1])

    def on_turn_start(
        self,
        user_message: str,
        *,
        workspace: Path | None = None,
        context_session: Any | None = None,
    ) -> None:
        self._survey_filter.reset()
        if self._session.is_silent():
            return
        if self._session.mode == TraceMode.REPLY:
            return
        preview = _clip_line(user_message.replace("\n", " "), 120)
        self._line(
            _c(f"[{_timestamp()}] ", "90")
            + _c("▶ 用户消息", "1")
            + f"  {preview}",
        )
        skills_line = _format_turn_skills_line(workspace, context_session, user_message)
        if skills_line:
            self._line(
                _c(f"[{_timestamp()}] ", "90")
                + _c("▶ 技能", "35")
                + f"  {skills_line}",
            )
        self._line(
            _c(f"{_TRACE_L1}提示: 大段文本用 /paste；/trace /rule /skill；/help", "90"),
        )
        expand_tip = (
            "  实时: /trace all  ·  展开: 输入步骤号 或 /trace step <#>"
            if _trace_sink_is_terminal(self._session)
            else "  实时: /trace all  ·  Ctrl+O 步骤侧栏  ·  /trace step <#>"
        )
        self._line(
            _c(f"[{_timestamp()}] ", "90")
            + _c("▶ 思考中…", "33")
            + _c(expand_tip, "90"),
        )
        self._step_start = time.perf_counter()

    def on_agent_update(self, messages: list) -> None:
        if not self._session.shows_process():
            self._step_start = time.perf_counter()
            return
        if not messages:
            return
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return

        elapsed = time.perf_counter() - self._step_start
        tool_calls = getattr(last, "tool_calls", None) or []
        text = _message_text(last.content).strip()
        verbose = self._session.is_verbose()
        step_usage = _extract_usage_from_ai_message(last)

        if tool_calls:
            for call in tool_calls:
                name = call.get("name", "?")
                if isinstance(name, str):
                    self._tool_names.append(name)
            if verbose and text:
                plan_label = "💭 Thought" if text.lstrip().startswith("【规划】") else "规划摘要"
                self._line(
                    _c(f"{_TRACE_L2}{plan_label}: ", "35")
                    + _c(f"{text[:400]}{'…' if len(text) > 400 else ''}", "90"),
                )
            plan_body: list[str] = []
            if text:
                plan_body.extend(text.splitlines())
            if verbose:
                for call in tool_calls:
                    name = call.get("name", "?")
                    args = call.get("args", {})
                    plan_body.append(
                        f"拟调用 {name}({_format_tool_args(args, verbose=True)})"
                    )
            plan_summary = _format_planned_tools_summary(tool_calls, verbose=verbose)
            resolved_usage = self._resolve_step_usage(step_usage)
            step_id = self._register_step(
                "plan",
                "模型决策",
                elapsed,
                plan_summary,
                body_lines=plan_body,
                usage=resolved_usage,
            )
            if verbose:
                self._line(
                    _c(f"[{_timestamp()}] ", "90")
                    + _c(f"🤔 模型决策 #{step_id}", "33")
                    + _c(f"  ({_format_duration(elapsed)})", "90"),
                )
                for call in tool_calls:
                    name = call.get("name", "?")
                    args = call.get("args", {})
                    self._line(
                        _c(f"{_TRACE_L2}└ ", "90")
                        + _c(f"拟调用 {name}", "36")
                        + f"({_format_tool_args(args, verbose=True)})",
                    )
            else:
                self._print_step_summary(
                    step_id,
                    "模型决策",
                    elapsed,
                    plan_summary,
                    usage=resolved_usage,
                )
                if any(call.get("name") == "web_search" for call in tool_calls):
                    self._line(
                        _c(f"[{_timestamp()}] ", "90")
                        + _c("⏳ web_search 等待 Tavily 响应（通常 5～25 秒）…", "33"),
                    )
            self._streamed_reply = False
        elif text and not self._streamed_reply:
            self._final_text = text
            if self._session.shows_process() and not self._session.is_verbose():
                self._emit_final_reply_block(text)
        elif not tool_calls and not text:
            # 最终轮 thinking-only：暂存，on_turn_end 无正文时降级展示
            thinking_only = _extract_thinking_from_message_chunk(last)
            if thinking_only:
                self._last_thinking_text = thinking_only

        self._step_start = time.perf_counter()

    def on_tools_update(self, messages: list) -> None:
        if not self._session.shows_process():
            self._step_start = time.perf_counter()
            return
        elapsed = time.perf_counter() - self._step_start
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            return

        verbose = self._session.is_verbose()
        if verbose:
            step_id = self._register_step(
                "tools",
                "工具执行",
                elapsed,
                f"{len(tool_msgs)} 个工具",
            )
            self._line(
                _c(f"[{_timestamp()}] ", "90")
                + _c(f"🔧 工具执行 #{step_id}", "32")
                + _c(f"  ({_format_duration(elapsed)})", "90"),
            )

        for msg in tool_msgs:
            name = msg.name or "tool"
            full_text = _message_text(msg.content)
            lines = full_text.splitlines() if full_text else ["(无输出)"]
            is_error = _tool_output_looks_like_error(full_text)
            preview_limit = 80 if is_error and verbose else _ALL_TOOL_OUTPUT_LINES
            if verbose:
                self._line(
                    _c(f"{_TRACE_L2}└ {name}", "36")
                    + _c(f"  ({len(lines)} 行输出)", "90"),
                )
                self._print_preview_block(lines, limit=preview_limit)
            else:
                output_summary = f"{len(lines)} 行输出"
                step_id = self._register_step(
                    "tool",
                    f"执行 {name}",
                    elapsed,
                    output_summary,
                    body_lines=lines,
                )
                self._print_step_summary(
                    step_id,
                    f"执行 {name}",
                    elapsed,
                    f"· {output_summary}",
                )

        self._step_start = time.perf_counter()
        self._streamed_reply = False
        if verbose:
            self._printed_final_header = False

    def _emit_final_reply_block(self, text: str) -> None:
        """
        steps 折叠模式下整段输出最终回复（中间轮次不走流式，避免与步骤头窜行）。

        @param text 助手最终正文
        """
        if not text.strip():
            return
        elapsed = time.perf_counter() - self._step_start
        self._step_index += 1
        if self._session.shows_process():
            label = f"💬 #{self._step_index} 助手回复"
            self._line(
                _c(f"[{_timestamp()}] ", "90")
                + _c(label, "1")
                + _c(f"  ({_format_duration(elapsed)})", "90"),
            )
        self._printed_final_header = True
        self._streamed_reply = True
        self._survey_filter.reset()
        visible = self._survey_filter.feed(text)
        if visible:
            self._stream(visible)
        self._stream_end()

    def on_text_chunk(self, chunk_text: str) -> None:
        if not self._session.shows_reply_stream():
            self._final_text += chunk_text
            return
        if not chunk_text:
            return
        # steps 折叠：中间轮次只累积，最终回复在 on_agent_update 整段输出
        if self._session.shows_process() and not self._session.is_verbose():
            self._final_text += chunk_text
            return
        if not self._printed_final_header:
            elapsed = time.perf_counter() - self._step_start
            self._step_index += 1
            if self._session.shows_process():
                label = (
                    f"💬 助手回复（流式）#{self._step_index}"
                    if self._session.is_verbose()
                    else f"💬 #{self._step_index} 助手回复"
                )
                self._line(
                    _c(f"[{_timestamp()}] ", "90")
                    + _c(label, "1")
                    + _c(f"  ({_format_duration(elapsed)})", "90"),
                )
            self._printed_final_header = True
        self._streamed_reply = True
        self._final_text += chunk_text
        visible = self._survey_filter.feed(chunk_text)
        if visible:
            self._stream(visible)

    def on_turn_end(self, *, last_step_body: str = "") -> tuple[str, str]:
        if (
            self._session.mode == TraceMode.STEPS
            and last_step_body
        ):
            lines = last_step_body.splitlines()
            if lines:
                self._line(_c(f"{_TRACE_L1}▼ 最近步骤预览", "90"))
                self._print_preview_block(lines)

        if not self._session.is_silent():
            if self._printed_final_header:
                tail = self._survey_filter.flush()
                if tail:
                    self._stream(tail)
                self._stream_end()
            total = time.perf_counter() - self._turn_start
            self._line(
                _c(f"[{_timestamp()}] ", "90")
                + _c("✓ 本轮完成", "32")
                + _c(f"  {_format_duration(total)}", "90"),
            )
            if self._session.mode == TraceMode.STEPS and self._steps:
                total_usage = _sum_steps_usage(self._steps)
                if total_usage is not None and self._session.show_step_tokens:
                    self._line(
                        _c(
                            f"{_TRACE_L1}本轮 token 合计: "
                            f"{_format_step_usage_inline(total_usage)}",
                            "35",
                        ),
                    )
                if _trace_sink_is_terminal(self._session):
                    hint = (
                        f"{_TRACE_L1}提示: 输入步骤号或 /trace step <#> 展开"
                        " · /trace all 看完整过程 · /trace token 关步骤 token"
                    )
                else:
                    hint = (
                        f"{_TRACE_L1}提示: 点击左侧步骤展开 · /trace step 列表"
                        " · /trace token 关步骤 token"
                    )
                self._line(_c(hint, "90"))
        from llgraph.survey.survey_prompt import strip_survey_for_display

        if not self._final_text.strip() and self._last_thinking_text.strip():
            fallback = self._last_thinking_text.strip()
            wrapped = (
                "（模型未输出可见正文，以下为 thinking 降级展示）\n\n"
                + fallback
            )
            self._final_text = wrapped
            if (
                not self._session.is_silent()
                and not self._printed_final_header
            ):
                self._emit_final_reply_block(wrapped)

        raw = self._final_text.strip()
        display = strip_survey_for_display(self._final_text).strip()
        self._session.last_turn_raw_reply = raw
        return display, raw


def _sum_steps_usage(steps: list[TraceStepRecord]) -> StepUsage | None:
    """
    汇总本轮各「模型决策」步 token（工具步无独立 LLM 用量）。

    @param steps 步骤列表
    @return 合计用量
    """
    total = StepUsage()
    found = False
    for step in steps:
        if step.usage is None:
            continue
        found = True
        total.input_tokens += step.usage.input_tokens
        total.output_tokens += step.usage.output_tokens
        total.cache_read_input_tokens += step.usage.cache_read_input_tokens
        total.cache_creation_input_tokens += step.usage.cache_creation_input_tokens
        total.cache_reported = total.cache_reported or step.usage.cache_reported
    return total if found else None


def _find_trace_step(
    session: TraceSession,
    target: str,
) -> TraceStepRecord | None:
    """
    按编号或 last 查找步骤。

    @param session 追踪会话
    @param target 步骤号、#96 或 last/-
    @return 步骤记录，未找到返回 None
    """
    steps = session.last_turn_steps
    if not steps:
        return None
    key = target.strip().lower()
    if key in ("last", "-", "latest"):
        return steps[-1]
    try:
        step_id = int(key.lstrip("#"))
    except ValueError:
        return None
    for step in steps:
        if step.step_id == step_id:
            return step
    return None


def print_trace_step_list(session: TraceSession) -> None:
    """
    列出上一轮各步骤摘要（折叠态）。

    @param session 追踪会话
    """
    steps = session.last_turn_steps
    if not steps:
        _trace_line(session, "本轮暂无步骤记录（需在 steps/all 模式下执行一轮）")
        return
    if _trace_sink_is_terminal(session):
        hint = "输入步骤号或 /trace step <#> 展开"
    else:
        hint = "点击左侧或 /trace step <#> 展开"
    _trace_line(session, f"本轮共 {len(steps)} 步（{hint}）")
    for step in steps:
        token_part = ""
        if session.show_step_tokens:
            if step.usage is not None:
                inline = _format_step_usage_inline(step.usage)
                if inline:
                    token_part = f"  [{inline}]"
        _trace_line(session, 
            f"  ▶ #{step.step_id} {step.title}  ({_format_duration(step.elapsed)})"
            f"  {step.summary}{token_part}"
            f"  · 点击左侧展开",
        )


def print_trace_step_detail(session: TraceSession, target: str) -> None:
    """
    展开指定步骤的完整 trace。

    @param session 追踪会话
    @param target 步骤号或 last
    """
    step = _find_trace_step(session, target)
    if step is None:
        steps = session.last_turn_steps
        if not steps:
            _trace_line(session, "本轮暂无步骤记录（需在 steps/all 模式下执行一轮）")
            return
        _trace_line(session, 
            f"未找到步骤 {target!r}，可用 /trace step 查看列表（1～{steps[-1].step_id}）",
        )
        return
    _trace_line(session, 
        _c(f"▼ #{step.step_id} {step.title}", "32")
        + _c(f"  ({_format_duration(step.elapsed)})", "90")
        + f"  {step.summary}",
    )
    if step.usage is not None:
        _trace_line(session, _c(f"{_TRACE_L2}{_format_step_usage_inline(step.usage)}", "35"))
    if step.body_lines:
        for line in step.body_lines:
            _trace_line(session, _c(f"{_TRACE_L2}{_clip_line(line)}", "90"))
    else:
        _trace_line(session, _c(f"{_TRACE_L2}(无详情)", "90"))


def set_trace_step_tokens(session: TraceSession, enabled: bool | None = None) -> bool:
    """
    开关步骤行 token/cache 展示。

    @param session 追踪会话
    @param enabled True/False 显式设置；None 则切换
    @return 切换后的状态
    """
    if enabled is None:
        session.show_step_tokens = not session.show_step_tokens
    else:
        session.show_step_tokens = enabled
    return session.show_step_tokens


def _extract_text_from_message_chunk(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _extract_thinking_from_message_chunk(chunk: Any) -> str:
    """从流式 chunk 提取 thinking 文本（仅作正文为空时的降级展示）。"""
    content = getattr(chunk, "content", None)
    extra = getattr(chunk, "additional_kwargs", None) or {}
    meta = extra.get("llgraph") if isinstance(extra, dict) else None
    if isinstance(meta, dict):
        stored = meta.get("thinking_text")
        if isinstance(stored, str) and stored.strip():
            return stored.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type", "")).lower()
            if kind not in ("thinking", "reasoning", "reasoning_text", "redacted_thinking"):
                continue
            text = (
                block.get("thinking")
                or block.get("reasoning")
                or block.get("text")
                or block.get("data")
            )
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


def _print_trace_usage(session: TraceSession) -> None:
    token_state = "开" if session.show_step_tokens else "关"
    _trace_line(session, 
        f"当前过程展示: {TRACE_MODE_LABELS[session.mode]} ({session.mode.value})",
    )
    _trace_line(session, f"步骤 token 显示: {token_state}（/trace token 切换）")
    _trace_line(session, "用法: /trace <模式|子命令>")
    _trace_line(session, "  all    完整过程（规划+工具参数+输出，同截图）")
    _trace_line(
        session,
        "  steps  折叠步骤摘要（默认，左侧列表可点击展开）",
    )
    _trace_line(session, "  reply  仅流式最终回复")
    _trace_line(session, "  none   都不展示")
    _trace_line(session, "  step         列出本轮步骤")
    _trace_line(session, "  step <#>     展开指定步骤详情")
    _trace_line(session, "  step last    展开最近一步")
    _trace_line(session, "  token        开关每步 token/cache 摘要")
    _trace_line(session, "  token on|off 显式开关步骤 token")
    _trace_line(session, "  stats        token 估算与工具落盘统计（含执行日志路径）")


def _collect_tool_names_from_updates(payload: dict) -> list[str]:
    """从 updates 载荷提取本轮工具名。"""
    names: list[str] = []
    if not isinstance(payload, dict):
        return names
    for _node, state_update in payload.items():
        messages = (state_update or {}).get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in getattr(msg, "tool_calls", None) or []:
                    name = call.get("name", "?")
                    if isinstance(name, str):
                        names.append(name)
    return names


def _stream_collect_silent(
    agent,
    user_message: str,
    *,
    thread_id: str,
    with_memory: bool,
    effective_message: str | None = None,
    trace_session: TraceSession | None = None,
) -> TurnRunResult:
    """NONE 模式：不打印过程，只收集最终回复。"""
    turn_start = time.perf_counter()
    config = {"configurable": {"thread_id": thread_id}} if with_memory else None
    final_parts: list[str] = []
    tool_names: list[str] = []
    payload = effective_message if effective_message is not None else user_message
    input_state = {"messages": [{"role": "user", "content": payload}]}

    for item in agent.stream(
        input_state,
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        mode, chunk = item
        if mode == "updates" and isinstance(chunk, dict):
            tool_names.extend(_collect_tool_names_from_updates(chunk))
            continue
        if mode != "messages" or not isinstance(chunk, tuple):
            continue
        msg_chunk, metadata = chunk
        meta = metadata if isinstance(metadata, dict) else {}
        if meta.get("langgraph_node", "") != "agent":
            continue
        if getattr(msg_chunk, "tool_calls", None) or []:
            continue
        text = _extract_text_from_message_chunk(msg_chunk)
        if text:
            final_parts.append(text)

    result = "".join(final_parts).strip()
    session = trace_session or TraceSession()
    if result:
        _trace_stream(session, result)
        _trace_stream_end(session)
    return TurnRunResult(
        text=result,
        tool_names=tool_names,
        duration_sec=time.perf_counter() - turn_start,
    )


def stream_agent_turn(
    agent,
    user_message: str,
    *,
    thread_id: str = "default",
    with_memory: bool = False,
    trace_session: TraceSession | None = None,
    effective_message: str | None = None,
    write_failure_tracker=None,
    workspace: Path | str | None = None,
    context_session: Any | None = None,
) -> TurnRunResult:
    """
    流式执行一轮对话并按 TraceMode 展示。

    @param user_message 用于过程展示的用户消息原文
    @param effective_message 实际发给模型的消息（含 workspace-context 时传入）
    @param trace_session 展示配置，默认 steps
    @param write_failure_tracker 写工具失败跟踪（可选）
    @param workspace 工作区根（展示本回合技能）
    @param context_session Rule/Skill 会话状态
    @return 助手最终文本与本轮指标
    """
    global LAST_TRACE_SESSION

    from llgraph.context.runtime_context import set_active_thread_id

    set_active_thread_id(thread_id if with_memory else None)

    session = trace_session or TraceSession()
    LAST_TRACE_SESSION = session
    session.last_turn_steps = []
    payload = effective_message if effective_message is not None else user_message
    turn_start = time.perf_counter()

    if session.is_silent():
        return _stream_collect_silent(
            agent,
            user_message,
            thread_id=thread_id,
            with_memory=with_memory,
            effective_message=payload,
            trace_session=session,
        )

    config = {"configurable": {"thread_id": thread_id}} if with_memory else None
    printer = TurnTracePrinter(session)
    ws_path = Path(workspace).expanduser().resolve() if workspace is not None else None
    printer.on_turn_start(
        user_message,
        workspace=ws_path,
        context_session=context_session,
    )

    input_state = {"messages": [{"role": "user", "content": payload}]}
    saw_tool_round = False
    streaming_reply = False
    last_tool_body = ""

    for item in agent.stream(
        input_state,
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        mode, payload = item

        if mode == "updates" and isinstance(payload, dict):
            for node_name, state_update in payload.items():
                messages = (state_update or {}).get("messages", [])
                if node_name == "agent":
                    if any(
                        isinstance(m, AIMessage) and (m.tool_calls or [])
                        for m in messages
                    ):
                        saw_tool_round = True
                        streaming_reply = False
                    printer.on_agent_update(messages)
                elif node_name == "tools":
                    saw_tool_round = True
                    streaming_reply = False
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            last_tool_body = _message_text(msg.content)
                    if write_failure_tracker is not None:
                        write_failure_tracker.inspect_tool_messages(messages)
                    printer.on_tools_update(messages)
                    if with_memory and ws_path is not None:
                        from llgraph.context.context_compressor import (
                            format_compress_report,
                            maybe_compress_during_react,
                        )
                        from llgraph.terminal.ops_notice import ops_notice

                        react_compress = maybe_compress_during_react(
                            agent,
                            thread_id=thread_id,
                            workspace=ws_path,
                        )
                        if react_compress is not None:
                            ops_notice(
                                "ReAct 中途压缩: " + format_compress_report(react_compress)
                            )

        elif mode == "messages" and isinstance(payload, tuple) and len(payload) == 2:
            msg_chunk, metadata = payload
            meta = metadata if isinstance(metadata, dict) else {}
            if meta.get("langgraph_node", "") != "agent":
                continue
            if isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                printer.absorb_usage_from_chunk(msg_chunk)
            thinking = _extract_thinking_from_message_chunk(msg_chunk)
            if thinking:
                printer._last_thinking_text = thinking
            if getattr(msg_chunk, "tool_calls", None) or []:
                streaming_reply = False
                continue
            text = _extract_text_from_message_chunk(msg_chunk)
            if not text:
                continue
            if saw_tool_round or not streaming_reply:
                streaming_reply = True
            printer.on_text_chunk(text)

    display_text, raw_text = printer.on_turn_end(last_step_body=last_tool_body)
    session.last_turn_steps = list(printer._steps)
    return TurnRunResult(
        text=display_text,
        raw_text=raw_text,
        tool_names=list(printer._tool_names),
        duration_sec=time.perf_counter() - turn_start,
    )
