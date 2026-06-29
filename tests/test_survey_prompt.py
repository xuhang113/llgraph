"""survey 解析：仅认 Agent 显式输出的 <<<llgraph-survey>>> 块。"""

from llgraph.survey.survey_prompt import (
    extract_survey_block,
    resolve_survey_from_assistant,
    strip_survey_for_display,
)


def test_resolve_survey_only_from_explicit_block() -> None:
    text = (
        "请先确认范围。\n"
        "判定失效后需重建 Client；@RefreshScope 监听；覆盖 localMap。\n"
        "<<<llgraph-survey>>>\n"
        '{"title":"请确认","questions":[{"id":"q1","prompt":"选一项",'
        '"options":["A","B"]}]}\n'
        "<<<end-survey>>>"
    )
    spec = resolve_survey_from_assistant(text)
    assert spec is not None
    assert spec.title == "请确认"
    assert len(spec.questions) == 1


def test_resolve_survey_rejects_heuristic_keywords_without_block() -> None:
    text = (
        "须用户确认改动范围与覆盖策略。\n"
        "判定失效：连续 ping 失败；@RefreshScope；覆盖 localMap。\n"
        "1. 仅 docs/\n"
        "2. 结合代码\n"
    )
    assert resolve_survey_from_assistant(text) is None
    assert extract_survey_block(text) is None


def test_strip_survey_for_display_keeps_plain_confirmation_text() -> None:
    text = "请确认以下选项：\n1. 方案 A\n2. 方案 B"
    assert strip_survey_for_display(text) == text


def test_strip_survey_for_display_removes_survey_block() -> None:
    text = (
        "正文在前。\n"
        "<<<llgraph-survey>>>{\"title\":\"T\",\"questions\":[]}"
        "<<<end-survey>>>\n"
        "正文在后。"
    )
    cleaned = strip_survey_for_display(text)
    assert "<<<llgraph-survey>>>" not in cleaned
    assert "正文在前" in cleaned
    assert "正文在后" in cleaned
