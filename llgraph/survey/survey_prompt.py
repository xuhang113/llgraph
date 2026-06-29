"""结构化确认问卷：解析 Agent 输出并走终端菜单（避免手写长段回复）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_SURVEY_START = "<<<llgraph-survey>>>"
_SURVEY_END = "<<<end-survey>>>"
_SURVEY_BLOCK_RE = re.compile(
    re.escape(_SURVEY_START) + r"[\s\S]*?"
    + r"(?:" + re.escape(_SURVEY_END) + r"|$)",
)


class SurveyStreamFilter:
    """流式输出时隐藏 survey JSON 块，避免与向导双显。"""

    _HOLD_BACK = len(_SURVEY_START) - 1

    def __init__(self) -> None:
        self._buffer = ""
        self._in_survey = False

    def _hold_back_suffix(self, text: str) -> tuple[str, str]:
        """
        _emit 时保留可能是 <<<llgraph-survey>>> 前缀的尾部，避免跨 chunk 漏出标记。

        @param text 待输出片段
        @return (可安全输出, 留待下一片段拼接的尾部)
        """
        if not text or self._HOLD_BACK <= 0:
            return text, ""
        marker = _SURVEY_START
        for size in range(min(len(text), self._HOLD_BACK), 0, -1):
            tail = text[-size:]
            if marker.startswith(tail):
                return text[:-size], tail
        return text, ""

    def feed(self, chunk: str) -> str:
        """
        喂入流式片段，返回可展示给用户的文本。

        @param chunk 模型输出片段
        @return 应打印的正文（survey/确认块内返回空串）
        """
        if not chunk:
            return ""
        self._buffer += chunk
        visible_parts: list[str] = []

        while self._buffer:
            if not self._in_survey:
                survey_idx = self._buffer.find(_SURVEY_START)
                if survey_idx >= 0:
                    if survey_idx > 0:
                        visible_parts.append(self._buffer[:survey_idx])
                    self._buffer = self._buffer[survey_idx + len(_SURVEY_START) :]
                    self._in_survey = True
                    continue
                safe, hold = self._hold_back_suffix(self._buffer)
                if hold:
                    visible_parts.append(safe)
                    self._buffer = hold
                else:
                    visible_parts.append(self._buffer)
                    self._buffer = ""
                break

            end_idx = self._buffer.find(_SURVEY_END)
            if end_idx < 0:
                self._buffer = ""
                break
            self._buffer = self._buffer[end_idx + len(_SURVEY_END) :]
            self._in_survey = False

        return "".join(visible_parts)

    def flush(self) -> str:
        """
        流式结束时刷出 hold-back 尾部（survey 块内则丢弃）。

        @return 剩余可展示文本
        """
        if not self._buffer:
            return ""
        if self._in_survey:
            self._buffer = ""
            return ""
        tail = self._buffer
        self._buffer = ""
        if _SURVEY_START in tail:
            return strip_survey_for_display(tail)
        return tail

    def reset(self) -> None:
        """新一轮对话前重置。"""
        self._buffer = ""
        self._in_survey = False


def strip_survey_for_display(text: str) -> str:
    """
    从展示用正文中移除 survey 块（解析仍用原文）。

    @param text 助手全文
    @return 去掉 survey 后的文本
    """
    if _SURVEY_START not in text:
        return text
    cleaned = _SURVEY_BLOCK_RE.sub("", text)
    return cleaned.strip()


def _normalize_survey_payload(data: dict) -> dict | None:
    """
    修复 Agent 常见畸形 survey JSON（根级仅有 step_label + option_hints）。

    @param data 解析后的对象
    @return 含 questions 列表的对象；无法修复时 None
    """
    raw_questions = data.get("questions")
    if isinstance(raw_questions, list) and raw_questions:
        return data

    title = str(data.get("title") or data.get("step_label") or "请确认").strip()
    prompt = str(
        data.get("prompt") or data.get("step_label") or title or "请选择一项",
    ).strip()
    options_raw = data.get("options")
    hints_raw = data.get("option_hints")
    options: list[str] = []
    if isinstance(options_raw, list):
        options = [str(o).strip() for o in options_raw if str(o).strip()]
    if len(options) < 2 and isinstance(hints_raw, list):
        options = [str(h).strip() for h in hints_raw if str(h).strip()]
    if len(options) < 2:
        return None

    hints: list[str] = []
    if isinstance(hints_raw, list):
        hints = [str(h).strip() for h in hints_raw]
    if hints == options:
        hints = []

    default = 1
    if "default" in data:
        try:
            default = int(data["default"])
        except (TypeError, ValueError):
            default = 1

    step_label = str(data.get("step_label") or "确认").strip()
    return {
        "title": title,
        "questions": [
            {
                "id": str(data.get("id") or "choice"),
                "prompt": prompt,
                "options": options,
                "default": default,
                "step_label": step_label,
                "option_hints": hints,
            },
        ],
    }


def _try_parse_survey_json_relaxed(chunk: str) -> dict | None:
    """
    宽松解析 survey JSON（提取首个 {...} 对象）。

    @param chunk survey 块正文
    @return 字典或 None
    """
    match = re.search(r"\{[\s\S]*\}", chunk)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass(frozen=True)
class SurveyQuestion:
    """单道确认题。"""

    question_id: str
    prompt: str
    options: tuple[str, ...]
    default_index: int
    allow_free_text: bool
    step_label: str = ""
    option_hints: tuple[str, ...] = ()
    multi_select: bool = False
    default_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class SurveySpec:
    """问卷规格。"""

    title: str
    questions: tuple[SurveyQuestion, ...]


def extract_survey_block(text: str) -> SurveySpec | None:
    """
    从助手回复中解析 <<<llgraph-survey>>> 块。

    @param text 助手全文
    @return 问卷规格；无块时 None
    """
    if _SURVEY_START not in text:
        return None
    start = text.find(_SURVEY_START)
    end = text.find(_SURVEY_END, start)
    if end < 0:
        chunk = text[start + len(_SURVEY_START) :].strip()
    else:
        chunk = text[start + len(_SURVEY_START) : end].strip()
    if chunk.startswith("```"):
        chunk = re.sub(r"^```(?:json)?\s*", "", chunk)
        chunk = re.sub(r"\s*```$", "", chunk)
    try:
        data = json.loads(chunk)
    except json.JSONDecodeError:
        data = _try_parse_survey_json_relaxed(chunk)
        if data is None:
            return None
    if not isinstance(data, dict):
        return None
    normalized = _normalize_survey_payload(data)
    if normalized is None:
        return None
    data = normalized
    title = str(data.get("title") or "请确认以下项").strip()
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        return None

    questions: list[SurveyQuestion] = []
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id") or f"q{len(questions) + 1}").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        opts_raw = item.get("options")
        if not isinstance(opts_raw, list) or len(opts_raw) < 2:
            continue
        options = tuple(str(o).strip() for o in opts_raw if str(o).strip())
        if len(options) < 2:
            continue
        default = 0
        default_indices: tuple[int, ...] = ()
        multi_select = bool(item.get("multi_select") or item.get("multiple"))
        if "default" in item:
            raw_default = item["default"]
            if multi_select and isinstance(raw_default, list):
                picked: list[int] = []
                for entry in raw_default:
                    try:
                        idx = int(entry) - 1
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(options):
                        picked.append(idx)
                default_indices = tuple(dict.fromkeys(picked))
            else:
                try:
                    default = max(0, int(raw_default) - 1)
                except (TypeError, ValueError):
                    default = 0
        default = min(default, len(options) - 1)
        allow_free = bool(item.get("allow_free_text")) or any(
            "其他" in o or "手动" in o for o in options
        )
        if multi_select and not default_indices and default >= 0:
            default_indices = (default,)
        hints_raw = item.get("option_hints")
        hints: tuple[str, ...] = ()
        if isinstance(hints_raw, list):
            hint_list = [str(h).strip() for h in hints_raw]
            if hint_list and list(options) != hint_list:
                hints = tuple(hint_list)
        step_label = str(item.get("step_label") or "").strip()
        if not step_label:
            step_label = prompt[:16] if len(prompt) > 16 else prompt
        questions.append(
            SurveyQuestion(
                question_id=qid,
                prompt=prompt,
                options=options,
                default_index=default,
                allow_free_text=allow_free,
                step_label=step_label,
                option_hints=hints,
                multi_select=multi_select,
                default_indices=default_indices,
            )
        )
    if not questions:
        return None
    return SurveySpec(title=title, questions=tuple(questions))


def run_survey_interactive(spec: SurveySpec) -> dict[str, str] | None:
    """
    终端问卷向导。

    @param spec 问卷
    @return 题 id → 答案；取消返回 None
    """
    from llgraph.terminal.survey_wizard import run_survey_tty

    return run_survey_tty(spec)


def format_survey_answers_for_agent(
    answers: dict[str, str],
    *,
    allow_write: bool = True,
) -> str:
    """
    将问卷结果整理为发给 Agent 的简短确认消息。

    @param answers 题 id → 选项
    @param allow_write 当前是否可写（只读时附加无法落盘提醒）
    @return 用户消息正文
    """
    lines = ["【用户确认】"]
    for key, value in answers.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    if not allow_write:
        lines.append(
            "【系统提醒】当前会话为**只读模式**，无法 write_file / 落盘 tmp 模式（.tmp.md）；"
            "请勿按上述确认中的写文件项执行；在正文输出完整梳理，或明确提示用户先 `/write on`。"
        )
        lines.append("")
    lines.append("请按以上确认继续执行，无需重复提问。")
    return "\n".join(lines)


def resolve_survey_from_assistant(text: str) -> SurveySpec | None:
    """
    解析助手回复中的 <<<llgraph-survey>>> JSON 块。

    问卷是否弹出、题目内容均由 Agent 显式输出 survey 块决定；不做正文关键词推断。

    @param text 助手全文
    @return 问卷规格
    """
    return extract_survey_block(text)


def try_run_survey_followup(
    assistant_text: str,
    *,
    workspace: Path | None = None,
    context_session: object | None = None,
    allow_write: bool = True,
) -> str | None:
    """
    若助手回复需确认，则走终端向导并返回可提交的用户消息。

    @param assistant_text 助手回复
    @param workspace 工作区根
    @param context_session 会话状态
    @return 整理后的用户消息；无问卷或取消时 None
    """
    from llgraph.config.survey_settings import survey_followup_enabled

    if not survey_followup_enabled(workspace, context_session):
        return None
    if not allow_write:
        return None
    has_block = _SURVEY_START in assistant_text
    spec = resolve_survey_from_assistant(assistant_text)
    if spec is None:
        if has_block:
            from llgraph.terminal.notify import notify

            notify(
                "survey",
                "JSON 格式无法解析，请用对话说明选择",
            )
        return None
    from llgraph.terminal.notify import notify

    notify("survey", "请在下方菜单中确认（已隐藏选项正文）")
    answers = run_survey_interactive(spec)
    if answers is None:
        from llgraph.terminal.output import emit_hint

        emit_hint(
            "[survey] 已取消确认。直接说明你的选择继续对话。"
        )
        return None
    return format_survey_answers_for_agent(answers, allow_write=allow_write)
