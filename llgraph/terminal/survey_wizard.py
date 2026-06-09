"""终端多步确认向导（Tab 步骤条 · 可回退 · 配色）。"""

from __future__ import annotations

import sys

from llgraph.survey.survey_prompt import SurveySpec, SurveyQuestion
from llgraph.terminal.interactive_prompt import prompt_menu_tty
from llgraph.terminal.redraw import redraw_tty_block, reset_tty_redraw_slot
from llgraph.terminal.menu_option import MenuOption
from llgraph.terminal.style import color_enabled, sty

# --- TTY 控制（ANSI 转义序列，见 ECMA-48 / xterm DECSCUSR）---
# 向导在原地重绘菜单时隐藏光标，避免光标停在中间一行闪烁；结束或输入补充说明时再显示。
_ANSI_HIDE_CURSOR = "\033[?25l"  # DECSCUSR: 隐藏文本光标
_ANSI_SHOW_CURSOR = "\033[?25h"  # DECSCUSR: 显示文本光标

# redraw_tty_block 的槽位 id：与 trace、其它菜单的 "default" 槽位分开，各自记录「上一帧占几行」。
_SURVEY_SLOT = "llgraph_survey"

# --- 按键 raw 读入后的返回值 ---
_KEY_ESC = "\x1b"           # Esc 单独按下
_KEY_ENTER = "\r"           # Enter（raw 模式下也可能是 \n）
_KEY_TAB = "\t"             # Tab → 下一题
_KEY_SHIFT_TAB = "\x1b[Z"   # Shift+Tab → 上一题
_KEY_ARROW_UP = "\x1b[A"
_KEY_ARROW_DOWN = "\x1b[B"
_KEY_ARROW_LEFT = "\x1b[D"
_KEY_SPACE = " "


def _is_select_all_option(option: str) -> bool:
    """选项是否为「全部」类快捷项。"""
    text = option.strip()
    return text == "全部" or text.startswith("全部（") or text.startswith("全部(")


def _format_multi_answer(question: SurveyQuestion, picked_indices: set[int]) -> str:
    """
    多选答案序列化。

    @param question 题目
    @param picked_indices 选中下标集合
    @return 答案字符串
    """
    if not picked_indices:
        return ""
    ordered = sorted(picked_indices)
    labels = [question.options[idx] for idx in ordered]
    return "、".join(labels)
_KEY_ARROW_RIGHT = "\x1b[C"


def _stdin_is_tty() -> bool:
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _read_key_tty() -> str:
    """
    读取单键（TTY raw 模式）。

    关闭行缓冲与 echo，便于 ↑↓/Tab 一次读一个「逻辑键」；
    方向键等会先读 ESC，再读 [A/B/C/D 组成完整序列。
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                # 方向键：ESC [ A/B/C/D
                ch3 = sys.stdin.read(1)
                return ch + ch2 + ch3
            if ch2 == "Z":
                # Shift+Tab 在部分终端上报为 ESC Z
                return _KEY_SHIFT_TAB
            return ch + ch2
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _step_label(question: SurveyQuestion, idx: int) -> str:
    """
    步骤 Tab 标签。

    @param question 题目
    @param idx 0-based 序号
    @return 短标签
    """
    label = (question.step_label or "").strip()
    if label:
        return label
    prompt = (question.prompt or "").strip()
    if len(prompt) > 8:
        return prompt[:8] + "…"
    return prompt or f"步骤{idx + 1}"


def _tab_current(text: str) -> str:
    """当前步骤 Tab：粗体 + 反色（1;7;36）。"""
    if not color_enabled():
        return text
    return f"\033[1;7;36m{text}\033[0m"


def _render_progress_bar(
    spec: SurveySpec,
    *,
    step: int,
    answers: dict[str, str],
) -> str:
    """
    渲染 Tab 式步骤条（当前高亮 · 已完成绿色 ✓）。

    @param spec 问卷
    @param step 当前步骤（0..len(questions)，最后一格为 Submit）
    @param answers 已确认答案
    @return 一行文本
    """
    parts: list[str] = []
    total = len(spec.questions)
    for idx, question in enumerate(spec.questions):
        label = _step_label(question, idx)
        if idx == step:
            parts.append(_tab_current(f" ▣ {label} "))
        elif question.question_id in answers:
            parts.append(sty(f"✓ {label}", "ok"))
        else:
            parts.append(sty(label, "dim"))
    if step >= total:
        parts.append(_tab_current(" ▣ Submit "))
    else:
        parts.append(sty("Submit", "dim"))
    return _join_colored(sty(" · ", "hint"), parts)


def _join_colored(sep: str, parts: list[str]) -> str:
    """拼接已着色片段。"""
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        out += sep + part
    return out


def _render_survey_frame(
    spec: SurveySpec,
    *,
    step: int,
    selections: list[int],
    multi_selections: list[set[int]],
    answers: dict[str, str],
    option_index: int,
) -> str:
    """
    渲染整屏 survey 内容。

    @param spec 问卷
    @param step 当前步骤
    @param selections 各题选中下标
    @param answers 已确认答案
    @param option_index 当前题选项光标
    @return 多行文本
    """
    lines: list[str] = [
        sty(spec.title, "title"),
        "",
        _render_progress_bar(spec, step=step, answers=answers),
        "",
    ]
    total = len(spec.questions)
    if step >= total:
        # 最后一屏：汇总各题答案，Enter 提交
        lines.append(sty("确认并提交？", "label"))
        lines.append("")
        for question in spec.questions:
            val = answers.get(question.question_id, "—")
            key = question.step_label or question.question_id
            lines.append(sty(f"  · {key}: ", "dim") + sty(val, "value"))
        lines.append("")
        lines.append(sty("Enter 提交 · ← 返回修改 · Esc 取消", "hint"))
        return "\n".join(lines)

    question = spec.questions[step]
    step_title = question.step_label or f"步骤 {step + 1}/{total}"
    lines.append(sty(step_title, "accent"))
    lines.append(sty(question.prompt, "label"))
    lines.append("")
    hints = getattr(question, "option_hints", None)
    picked_multi = multi_selections[step] if step < len(multi_selections) else set()
    for idx, opt in enumerate(question.options):
        if question.multi_select:
            box = sty("☑", "ok") if idx in picked_multi else sty("☐", "dim")
            cursor = sty("›", "brand") if idx == option_index else " "
            label = sty(opt, "brand") if idx == option_index else sty(opt, "value")
            if "推荐" in opt and idx != option_index:
                label = sty(opt, "ok")
        else:
            box = ""
            cursor = sty("›", "brand") if idx == option_index else " "
            label = sty(opt, "brand") if idx == option_index else sty(opt, "value")
            if "推荐" in opt and idx != option_index:
                label = sty(opt, "ok")
        hint = ""
        if isinstance(hints, tuple) and idx < len(hints) and hints[idx]:
            hint = sty(f"  ({hints[idx]})", "hint")
        prefix = f" {cursor} {box} " if question.multi_select else f" {cursor} "
        lines.append(f"{prefix}{label}{hint}")
    lines.append("")
    if question.multi_select:
        footer = (
            "Space 勾选/取消 · Enter 确认/下一题 · Tab/→ 下一题 · "
            "←/Shift+Tab 上一题 · ↑↓ 移动 · Esc 取消"
        )
    else:
        footer = (
            "Enter 确认/下一题 · Tab/→ 下一题 · ←/Shift+Tab 上一题 · "
            "↑↓ 移动 · Esc 取消"
        )
    lines.append(sty(footer, "hint"))
    return "\n".join(lines)


def _commit_step(
    spec: SurveySpec,
    step: int,
    selections: list[int],
    multi_selections: list[set[int]],
    answers: dict[str, str],
) -> None:
    """
    将当前步骤选中项写入 answers。

    @param spec 问卷
    @param step 步骤下标
    @param selections 各题选中下标
    @param multi_selections 各题多选集合
    @param answers 答案 dict（就地更新）
    """
    if step >= len(spec.questions):
        return
    question = spec.questions[step]
    if question.multi_select:
        answers[question.question_id] = _format_multi_answer(
            question,
            multi_selections[step],
        )
        return
    picked = selections[step]
    answers[question.question_id] = question.options[picked]


def _toggle_multi_selection(question: SurveyQuestion, picked: set[int], idx: int) -> None:
    """
    切换多选勾选项；「全部」与其它项互斥。

    @param question 题目
    @param picked 当前选中集合（就地更新）
    @param idx 选项下标
    """
    option = question.options[idx]
    if _is_select_all_option(option):
        if idx in picked:
            picked.clear()
        else:
            picked.clear()
            picked.add(idx)
        return
    if idx in picked:
        picked.discard(idx)
        return
    for other_idx, other in enumerate(question.options):
        if other_idx in picked and _is_select_all_option(other):
            picked.discard(other_idx)
    picked.add(idx)


def _maybe_free_text(question: SurveyQuestion, choice: str) -> str:
    """
    「其他」类选项的补充说明。

    @param question 题目
    @param choice 当前选项文案
    @return 最终答案
    """
    if not question.allow_free_text:
        return choice
    if "其他" not in choice and "手动" not in choice:
        return choice
    # 需要用户打字：恢复光标；勿 reset slot，否则下一帧无法擦除本屏
    sys.stdout.write("\n")
    sys.stdout.write(_ANSI_SHOW_CURSOR)
    sys.stdout.flush()
    try:
        extra = input(sty("补充说明（可回车跳过）: ", "hint")).strip()
    except EOFError:
        return choice
    finally:
        sys.stdout.write(_ANSI_HIDE_CURSOR)
        sys.stdout.flush()
    if extra:
        return f"{choice}：{extra}"
    return choice


def _run_survey_wizard_tty(spec: SurveySpec) -> dict[str, str] | None:
    """
    Tab 式多步 survey（可回退修改上一步）。

    @param spec 问卷
    @return 题 id → 答案；取消返回 None
    """
    total = len(spec.questions)
    if total == 0:
        return {}
    if not _stdin_is_tty():
        return _run_survey_linear_fallback(spec)

    selections = [q.default_index for q in spec.questions]
    multi_selections: list[set[int]] = []
    for question in spec.questions:
        if question.multi_select:
            if question.default_indices:
                multi_selections.append(set(question.default_indices))
            else:
                multi_selections.append(set())
        else:
            multi_selections.append(set())
    answers: dict[str, str] = {}
    step = 0  # 0..total-1 为题目，total 为 Submit 汇总屏
    option_index = selections[0]

    slot = _SURVEY_SLOT
    reset_tty_redraw_slot(slot)
    sys.stdout.write(_ANSI_HIDE_CURSOR)
    sys.stdout.flush()
    try:
        while True:
            if step < total:
                option_index = min(option_index, len(spec.questions[step].options) - 1)
                selections[step] = option_index
            frame = _render_survey_frame(
                spec,
                step=step,
                selections=selections,
                multi_selections=multi_selections,
                answers=answers,
                option_index=option_index,
            )
            # 在同一屏幕区域覆盖上一帧，避免多步 survey 滚屏堆叠
            redraw_tty_block(frame, slot=slot)
            key = _read_key_tty()

            if key in (_KEY_ESC, "q", "Q"):
                reset_tty_redraw_slot(slot)
                return None
            if key in (_KEY_ENTER, "\n"):
                if step >= total:
                    # Submit 屏：补齐未 Tab 跳过的题，再返回
                    for idx, question in enumerate(spec.questions):
                        if question.question_id not in answers:
                            _commit_step(
                                spec,
                                idx,
                                selections,
                                multi_selections,
                                answers,
                            )
                    reset_tty_redraw_slot(slot)
                    return dict(answers)
                question = spec.questions[step]
                if question.multi_select:
                    if not multi_selections[step]:
                        continue
                    choice = _format_multi_answer(question, multi_selections[step])
                else:
                    selections[step] = option_index
                    choice = question.options[option_index]
                choice = _maybe_free_text(question, choice)
                answers[question.question_id] = choice
                step = min(step + 1, total)
                if step < total:
                    option_index = selections[step]
                continue
            if key in (_KEY_TAB, _KEY_ARROW_RIGHT):
                # 下一题：暂存当前选项（不触发「其他」补充输入）
                if step < total:
                    question = spec.questions[step]
                    if question.multi_select and not multi_selections[step]:
                        continue
                    _commit_step(
                        spec,
                        step,
                        selections,
                        multi_selections,
                        answers,
                    )
                step = min(step + 1, total)
                if step < total:
                    option_index = selections[step]
                continue
            if key in (_KEY_SHIFT_TAB, _KEY_ARROW_LEFT):
                # 上一题；在 Submit 屏则回到最后一题
                if step >= total:
                    step = total - 1
                    option_index = selections[step]
                elif step > 0:
                    step -= 1
                    option_index = selections[step]
                continue
            if step < total:
                opt_count = len(spec.questions[step].options)
                question = spec.questions[step]
                if key == _KEY_ARROW_UP:
                    option_index = (option_index - 1) % opt_count
                elif key == _KEY_ARROW_DOWN:
                    option_index = (option_index + 1) % opt_count
                elif key == _KEY_SPACE and question.multi_select:
                    _toggle_multi_selection(
                        question,
                        multi_selections[step],
                        option_index,
                    )
    finally:
        # 无论确认、取消还是异常，都要恢复光标
        sys.stdout.write(_ANSI_SHOW_CURSOR)
        sys.stdout.flush()


def _run_survey_linear_fallback(spec: SurveySpec) -> dict[str, str] | None:
    """非 TTY 回退：逐题序号输入。"""
    answers: dict[str, str] = {}
    total = len(spec.questions)
    for idx, question in enumerate(spec.questions, start=1):
        title = spec.title if total == 1 else f"{spec.title} ({idx}/{total})"
        header = f"{title}\n{question.step_label or idx}\n{question.prompt}"
        options = [MenuOption(label=opt, hint="") for opt in question.options]
        picked = prompt_menu_tty(header, options, default_index=question.default_index)
        if picked is None:
            return None
        choice = _maybe_free_text(question, question.options[picked])
        answers[question.question_id] = choice
    return answers


def run_survey_tty(spec: SurveySpec) -> dict[str, str] | None:
    """
    终端问卷：多步 Tab 向导（可回退）；非 TTY 时逐题菜单。

    @param spec 问卷
    @return 题 id → 答案；取消返回 None
    """
    if not _stdin_is_tty():
        return _run_survey_linear_fallback(spec)
    return _run_survey_wizard_tty(spec)
