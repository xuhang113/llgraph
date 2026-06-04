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
_CONFIRMATION_HEADER_RE = re.compile(
    r"(请确认|确认你的需求|确认以下|请选择以下|请在下方确认)",
    re.IGNORECASE,
)
_NUMBERED_OPTION_RE = re.compile(
    r"^\s*(\d+)[\.\)、]\s*(?:\*\*)?(.+?)(?:\*\*)?\s*$",
)


class SurveyStreamFilter:
    """流式输出时隐藏 survey JSON 块与重复确认列表，避免与向导双显。"""

    _HOLD_BACK = len(_SURVEY_START) - 1

    def __init__(self) -> None:
        self._buffer = ""
        self._in_survey = False
        self._suppress_confirmation = False

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
            if self._suppress_confirmation and not self._in_survey:
                survey_idx = self._buffer.find(_SURVEY_START)
                if survey_idx >= 0:
                    self._buffer = self._buffer[survey_idx + len(_SURVEY_START) :]
                    self._in_survey = True
                    continue
                self._buffer = ""
                break

            if not self._in_survey:
                survey_idx = self._buffer.find(_SURVEY_START)
                header_match = _CONFIRMATION_HEADER_RE.search(self._buffer)
                if survey_idx >= 0 and (
                    header_match is None or survey_idx < header_match.start()
                ):
                    if survey_idx > 0:
                        visible_parts.append(self._buffer[:survey_idx])
                    self._buffer = self._buffer[survey_idx + len(_SURVEY_START) :]
                    self._in_survey = True
                    continue
                if header_match is not None:
                    if header_match.start() > 0:
                        visible_parts.append(self._buffer[: header_match.start()])
                    self._buffer = self._buffer[header_match.start() :]
                    self._suppress_confirmation = True
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
        if self._in_survey or self._suppress_confirmation:
            self._buffer = ""
            return ""
        tail = self._buffer
        self._buffer = ""
        if _SURVEY_START in tail or _CONFIRMATION_HEADER_RE.search(tail):
            return strip_survey_for_display(tail)
        return tail

    def reset(self) -> None:
        """新一轮对话前重置。"""
        self._buffer = ""
        self._in_survey = False
        self._suppress_confirmation = False


def _strip_confirmation_markdown(text: str) -> str:
    """
    去掉「请确认 + 编号选项」类 Markdown（向导会接管交互）。

    @param text 助手正文
    @return 去掉确认列表后的文本
    """
    if not _CONFIRMATION_HEADER_RE.search(text):
        return text
    lines = text.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        if not skipping and _CONFIRMATION_HEADER_RE.search(line):
            match = _CONFIRMATION_HEADER_RE.search(line)
            if match and match.start() > 0:
                prefix = line[: match.start()].strip()
                if prefix:
                    out.append(prefix)
            skipping = True
            continue
        if skipping:
            if _NUMBERED_OPTION_RE.match(line.strip()):
                continue
            if line.strip() == "":
                continue
            if line.strip().startswith("---"):
                skipping = False
                out.append(line)
                continue
            skipping = False
        out.append(line)
    return "\n".join(out).strip()


def strip_survey_for_display(text: str) -> str:
    """
    从展示用正文中移除 survey 块（解析仍用原文）。

    @param text 助手全文
    @return 去掉 survey 后的文本
    """
    if _SURVEY_START not in text and not _CONFIRMATION_HEADER_RE.search(text):
        return text
    cleaned = _SURVEY_BLOCK_RE.sub("", text)
    cleaned = _strip_confirmation_markdown(cleaned)
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
        if "default" in item:
            try:
                default = max(0, int(item["default"]) - 1)
            except (TypeError, ValueError):
                default = 0
        default = min(default, len(options) - 1)
        allow_free = bool(item.get("allow_free_text")) or any(
            "其他" in o or "手动" in o for o in options
        )
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
            )
        )
    if not questions:
        return None
    return SurveySpec(title=title, questions=tuple(questions))


def run_survey_interactive(spec: SurveySpec) -> dict[str, str] | None:
    """
    TUI 问卷向导。

    @param spec 问卷
    @return 题 id → 答案；取消返回 None
    """
    from llgraph.ui.prompts import run_survey_wizard

    return run_survey_wizard(spec)


def format_survey_answers_for_agent(answers: dict[str, str]) -> str:
    """
    将问卷结果整理为发给 Agent 的简短确认消息。

    @param answers 题 id → 选项
    @return 用户消息正文
    """
    lines = ["【用户确认】"]
    for key, value in answers.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("请按以上确认继续执行，无需重复提问。")
    return "\n".join(lines)


def infer_survey_from_markdown(text: str) -> SurveySpec | None:
    """
    从「请确认 + 编号列表」推断单题问卷（Agent 未输出 survey 块时的兜底）。

    @param text 助手回复
    @return 问卷规格
    """
    if not re.search(r"请确认|确认你的需求|确认以下|请选择", text):
        return None
    options: list[str] = []
    for line in text.splitlines():
        matched = re.match(
            r"^\s*(\d+)[\.\)、]\s*(?:\*\*)?(.+?)(?:\*\*)?\s*$",
            line.strip(),
        )
        if not matched:
            continue
        body = matched.group(2).strip()
        if len(body) < 4:
            continue
        if body.startswith("---"):
            break
        options.append(body[:240])
    if len(options) < 2 or len(options) > 8:
        return None
    opts = tuple(options) + ("其他（手动输入）",)
    return SurveySpec(
        title="请确认你的需求",
        questions=(
            SurveyQuestion(
                question_id="choice",
                prompt="请选择一项（↑↓ · Enter，无需输入序号）",
                options=opts,
                default_index=0,
                allow_free_text=True,
                step_label="需求",
            ),
        ),
    )


def resolve_survey_from_assistant(text: str) -> SurveySpec | None:
    """
    解析助手回复中的问卷（JSON 块优先，其次 Markdown 编号列表）。

    @param text 助手全文
    @return 问卷规格
    """
    spec = extract_survey_block(text)
    if spec is not None:
        return spec
    return infer_survey_from_markdown(text)


def try_run_survey_followup(
    assistant_text: str,
    *,
    workspace: Path | None = None,
    context_session: object | None = None,
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
    has_block = _SURVEY_START in assistant_text
    spec = resolve_survey_from_assistant(assistant_text)
    if spec is None:
        if has_block:
            from llgraph.ui.context import ui_notify

            ui_notify(
                "survey",
                "JSON 格式无法解析，请用对话说明选择",
            )
        return None
    from llgraph.ui.context import ui_notify

    ui_notify("survey", "请在下方菜单中确认（已隐藏选项正文）")
    answers = run_survey_interactive(spec)
    if answers is None:
        return None
    return format_survey_answers_for_agent(answers)


_ORGANIZE_USER_PATTERN = re.compile(
    r"(梳理|整理|整理下|帮我.*?整理|帮我.*?梳理|看看.*?业务)",
    re.IGNORECASE,
)


def user_message_needs_organize_preflight(user_message: str) -> bool:
    """
    是否像 project-organize 的「梳理/整理业务」首轮请求。

    @param user_message 用户输入
    @return 是否建议先走确认向导
    """
    text = user_message.strip()
    if len(text) > 120:
        return False
    if _ORGANIZE_USER_PATTERN.search(text):
        return True
    return False


def build_organize_preflight_survey(user_message: str) -> SurveySpec:
    """
    根据用户消息生成 project-organize 默认向导（顶栏三题 + Submit）。

    @param user_message 用户原始输入
    @return 问卷规格
    """
    title = "业务梳理 — 请确认"
    if "优惠券" in user_message:
        title = "优惠券业务梳理 — 请确认"
    elif "charbi" in user_message.lower():
        title = "charbi 业务梳理 — 请确认"
    spec = default_project_organize_survey()
    return SurveySpec(title=title, questions=spec.questions)


def maybe_preflight_survey_for_user_message(
    user_message: str,
    *,
    preflight_done: bool,
    workspace: Path | None = None,
    context_session: object | None = None,
) -> tuple[str, bool] | None:
    """
    梳理类请求在调用 Agent 前先走向导（对齐 Claude：先确认再搜代码）。

    @param user_message 用户输入
    @param preflight_done 本会话是否已做过前置确认
    @param workspace 工作区根
    @param context_session 会话状态
    @return (增强后的用户消息, 已确认标记)；无需前置时 None
    """
    from llgraph.config.survey_settings import survey_preflight_enabled

    if not survey_preflight_enabled(workspace, context_session):
        return None
    if preflight_done or not user_message_needs_organize_preflight(user_message):
        return None
    spec = build_organize_preflight_survey(user_message)
    answers = run_survey_interactive(spec)
    if answers is None:
        return None
    payload = format_survey_answers_for_agent(answers)
    merged = f"{user_message.strip()}\n\n{payload}"
    return merged, True


def default_project_organize_survey() -> SurveySpec:
    """
    project-organize 技能默认三项确认（/survey 无参数时使用）。

    @return 问卷规格
    """
    return SurveySpec(
        title="业务梳理 — 请确认",
        questions=(
            SurveyQuestion(
                question_id="mode",
                prompt="梳理业务时按哪种方式？",
                options=(
                    "按现有文档梳理",
                    "结合代码重新梳理（推荐）",
                    "其他（手动输入）",
                ),
                default_index=1,
                allow_free_text=True,
                step_label="梳理方式",
                option_hints=(
                    "以 docs 已有文档为主，必要时核对代码",
                    "重读相关代码，文档作参考",
                    "",
                ),
            ),
            SurveyQuestion(
                question_id="write_docs",
                prompt="是否落盘到 docs？",
                options=(
                    "仅对话，不落盘",
                    "落盘到 docs/（工作区 + 各仓 docs）",
                    "两者都落（工作区总览 + 各仓 docs）",
                ),
                default_index=1,
                allow_free_text=False,
                step_label="是否落盘",
            ),
            SurveyQuestion(
                question_id="edit_existing",
                prompt="是否修改已有 doc？",
                options=(
                    "否，仅新增",
                    "是，可覆盖/补充已有文档",
                    "不适用（尚无 doc）",
                ),
                default_index=0,
                allow_free_text=False,
                step_label="改已有 doc",
            ),
        ),
    )
