"""会话级 Rule / Skill 配置（/rule、/skill 命令修改）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextSession:
    """
    每轮对话注入模型的规则与技能状态。

    active_skills: 用户通过 /skill 启用的技能名（整会话有效；manifest 中 ⭐ 标记）
    disabled_rules: 用户临时禁用的规则 id（/rule off）
    forced_rules: 用户强制启用的 glob 规则 id（/rule on）
    write_failure_hint: 写工具连续失败后注入下一轮的提醒（由 WriteFailureTracker 设置）
  """

    active_skills: list[str] = field(default_factory=list)
    disabled_rules: set[str] = field(default_factory=set)
    forced_rules: set[str] = field(default_factory=set)
    write_failure_hint: str = ""
    survey_enabled: bool | None = None

    def activate_skill(self, name: str) -> None:
        """
        启用技能（去重保序）。

        @param name 技能目录名
        """
        key = name.strip().lower()
        if not key:
            return
        lowered = [s.lower() for s in self.active_skills]
        if key not in lowered:
            self.active_skills.append(name.strip())

    def deactivate_skill(self, name: str) -> bool:
        """
        关闭指定技能。

        @param name 技能名
        @return 是否曾启用
        """
        key = name.strip().lower()
        before = len(self.active_skills)
        self.active_skills = [s for s in self.active_skills if s.lower() != key]
        return len(self.active_skills) < before

    def clear_skills(self) -> None:
        """清空已启用技能。"""
        self.active_skills.clear()

    def fork(self) -> ContextSession:
        """
        复制会话级 Rule/Skill 状态（并行 Worker 独立 ctx）。

        @return 新 ContextSession 快照
        """
        return ContextSession(
            active_skills=list(self.active_skills),
            disabled_rules=set(self.disabled_rules),
            forced_rules=set(self.forced_rules),
            write_failure_hint=self.write_failure_hint,
            survey_enabled=self.survey_enabled,
        )
