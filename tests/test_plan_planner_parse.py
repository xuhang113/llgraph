"""Planner plan.json 解析与修订降级策略。"""

from __future__ import annotations

from llgraph.plan.nodes import planner as planner_mod


def test_extract_plan_json_from_prose_and_fence() -> None:
    text = (
        "说明：汇总由系统自动完成。\n"
        "```json\n"
        '{"title": "demo", "tasks": [{"id": "w1", "title": "a"}, {"id": "w2", "title": "b"}]}\n'
        "```"
    )
    raw = planner_mod._extract_plan_json_raw(text)
    plan = planner_mod._parse_plan_from_text(raw, plan_id="p1", goal="g")
    assert len(plan["tasks"]) == 2


def test_revision_parse_fallback_keeps_previous_plan() -> None:
    previous = {
        "plan_id": "p1",
        "title": "旧计划",
        "tasks": [{"id": "w1", "title": "A"}, {"id": "w2", "title": "B"}],
    }
    plan = planner_mod._parse_plan_from_text("只有解释，没有 JSON。", plan_id="p1", goal="g")
    kept, err = planner_mod._apply_planner_parse_fallback(
        plan,
        parse_fallback=True,
        revision="为什么没有汇总节点",
        previous_plan=previous,
        goal="g",
        version=2,
    )
    assert err
    assert len(kept["tasks"]) == 2
    assert kept["title"] == "旧计划"
    assert kept["version"] == 2


def test_first_plan_parse_fallback_uses_w1() -> None:
    plan = planner_mod._parse_plan_from_text("无 JSON", plan_id="p1", goal="目标")
    kept, err = planner_mod._apply_planner_parse_fallback(
        plan,
        parse_fallback=True,
        revision="",
        previous_plan=None,
        goal="目标",
        version=1,
    )
    assert err
    assert len(kept["tasks"]) == 1
    assert kept["tasks"][0]["id"] == "w1"
