"""发往 LLM 前敏感信息脱敏（规则来自 ~/.llgraph/agent.json 或工作区 agent.json → context.outbound_redact）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# 旧 agent.json 布尔开关 → 规则 id 或特殊块名
_LEGACY_FLAG_TO_RULE_IDS: dict[str, tuple[str, ...]] = {
    "redact_ipv4": ("ipv4",),
    "redact_ip_in_hostname": ("ip_in_hostname",),
    "redact_url_with_creds": ("url_with_creds",),
    "redact_kv_secrets": ("kv_secret", "bearer_token", "ak_sk"),
    "redact_internal_hosts": ("internal_prod_host",),
    "redact_kafka_topics": ("kafka_topic",),
}
_LEGACY_FLAG_TO_BLOCK: dict[str, str] = {
    "redact_internal_domains": "domain_rules",
    "redact_internal_paths": "path_rules",
}


@dataclass(frozen=True)
class RegexRedactRule:
    """单条正则脱敏规则。"""

    rule_id: str
    enabled: bool
    pattern: re.Pattern[str]
    replacement: str
    replacement_uses_groups: bool


@dataclass(frozen=True)
class DomainRedactRules:
    """内网域名后缀匹配。"""

    enabled: bool
    match_urls: bool
    match_hostnames: bool
    suffixes: tuple[str, ...]


@dataclass(frozen=True)
class PathRedactRules:
    """内网路径前缀匹配。"""

    enabled: bool
    prefixes: tuple[str, ...]


@dataclass(frozen=True)
class OutboundRedactSettings:
    """出站脱敏配置（解析后）。"""

    enabled: bool
    placeholder: str
    regex_rules: tuple[RegexRedactRule, ...]
    domain_rules: DomainRedactRules
    path_rules: PathRedactRules


def _parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no")
    return bool(value)


def _str_list(raw: object, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace(";", ",").split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw]
    else:
        return default
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out) if out else default


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _merge_regex_rules(
    defaults: list[dict[str, Any]],
    override: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in defaults:
        rid = str(item.get("id", "")).strip()
        if rid:
            by_id[rid] = dict(item)
    if override:
        for item in override:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("id", "")).strip()
            if not rid:
                continue
            merged = dict(by_id.get(rid, {"id": rid}))
            merged.update(item)
            by_id[rid] = merged
    return list(by_id.values())


def _apply_legacy_flags(block: dict[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
    """将旧版 redact_* 布尔开关应用到配置。"""
    result = json.loads(json.dumps(merged))
    for flag, rule_ids in _LEGACY_FLAG_TO_RULE_IDS.items():
        if flag not in block:
            continue
        enabled = _parse_bool(block.get(flag), True)
        rules = result.get("regex_rules")
        if not isinstance(rules, list):
            continue
        id_set = set(rule_ids)
        result["regex_rules"] = [
            {**rule, "enabled": enabled if str(rule.get("id")) in id_set else rule.get("enabled", True)}
            for rule in rules
            if isinstance(rule, dict)
        ]
    for flag, block_name in _LEGACY_FLAG_TO_BLOCK.items():
        if flag not in block:
            continue
        enabled = _parse_bool(block.get(flag), True)
        section = result.get(block_name)
        if isinstance(section, dict):
            section["enabled"] = enabled
        elif block_name == "domain_rules":
            result["domain_rules"] = {"enabled": enabled}
        elif block_name == "path_rules":
            result["path_rules"] = {"enabled": enabled}
    legacy_suffixes = block.get("internal_domain_suffixes")
    if legacy_suffixes is not None:
        domain = result.setdefault("domain_rules", {})
        if isinstance(domain, dict):
            domain["suffixes"] = _str_list(legacy_suffixes, tuple(domain.get("suffixes") or ()))
    legacy_prefixes = block.get("internal_path_prefixes")
    if legacy_prefixes is not None:
        paths = result.setdefault("path_rules", {})
        if isinstance(paths, dict):
            paths["prefixes"] = _str_list(legacy_prefixes, tuple(paths.get("prefixes") or ()))
    return result


def resolve_outbound_redact_config(workspace: Path | None) -> dict[str, Any]:
    """
    从 agent.json 读取 context.outbound_redact：用户 ~/.llgraph 为底，工作区按字段覆盖。

    @param workspace 工作区根
    @return 脱敏配置 dict；未配置时 enabled=false
    """
    from llgraph.core.agent_config import load_user_agent_config, load_workspace_agent_config

    user_cfg = load_user_agent_config()
    user_ctx = user_cfg.get("context") if isinstance(user_cfg.get("context"), dict) else {}
    user_block = user_ctx.get("outbound_redact")
    if not isinstance(user_block, dict):
        user_block = {}

    ws_block: dict[str, Any] = {}
    if workspace is not None:
        ws_cfg = load_workspace_agent_config(workspace.expanduser().resolve())
        ws_ctx = ws_cfg.get("context") if isinstance(ws_cfg.get("context"), dict) else {}
        raw_ws = ws_ctx.get("outbound_redact")
        if isinstance(raw_ws, dict):
            ws_block = raw_ws

    if not user_block and not ws_block:
        return {"enabled": False}

    merged = _deep_merge_dict(user_block, ws_block)
    if user_block.get("regex_rules") or ws_block.get("regex_rules"):
        merged["regex_rules"] = _merge_regex_rules(
            list(user_block.get("regex_rules") or []),
            list(ws_block.get("regex_rules") or []) if ws_block else None,
        )
    if ws_block:
        merged = _apply_legacy_flags(ws_block, merged)
    merged.pop("legacy_flags", None)
    merged.pop("infra_tool_mask", None)
    return merged


def _compile_regex_rule(raw: dict[str, Any], placeholder: str) -> RegexRedactRule | None:
    rule_id = str(raw.get("id", "")).strip()
    pattern_text = str(raw.get("pattern", "")).strip()
    if not rule_id or not pattern_text:
        return None
    try:
        compiled = re.compile(pattern_text)
    except re.error:
        return None
    replacement = str(raw.get("replacement", placeholder))
    replacement = replacement.replace("[REDACTED]", placeholder)
    return RegexRedactRule(
        rule_id=rule_id,
        enabled=_parse_bool(raw.get("enabled"), True),
        pattern=compiled,
        replacement=replacement,
        replacement_uses_groups=_parse_bool(raw.get("replacement_uses_groups"), False),
    )


def _build_domain_rules(raw: dict[str, Any] | None) -> DomainRedactRules:
    block = raw if isinstance(raw, dict) else {}
    return DomainRedactRules(
        enabled=_parse_bool(block.get("enabled"), True),
        match_urls=_parse_bool(block.get("match_urls"), True),
        match_hostnames=_parse_bool(block.get("match_hostnames"), True),
        suffixes=_str_list(block.get("suffixes")),
    )


def _build_path_rules(raw: dict[str, Any] | None) -> PathRedactRules:
    block = raw if isinstance(raw, dict) else {}
    return PathRedactRules(
        enabled=_parse_bool(block.get("enabled"), True),
        prefixes=_str_list(block.get("prefixes")),
    )


def resolve_outbound_redact_settings(workspace: Path | None) -> OutboundRedactSettings:
    """
    解析出站脱敏配置。

    @param workspace 工作区根
    @return 解析后的设置
    """
    cfg = resolve_outbound_redact_config(workspace)
    placeholder = str(cfg.get("placeholder", "[REDACTED]")).strip() or "[REDACTED]"
    rules: list[RegexRedactRule] = []
    for raw in cfg.get("regex_rules") or []:
        if not isinstance(raw, dict):
            continue
        compiled = _compile_regex_rule(raw, placeholder)
        if compiled is not None:
            rules.append(compiled)
    return OutboundRedactSettings(
        enabled=_parse_bool(cfg.get("enabled"), True),
        placeholder=placeholder,
        regex_rules=tuple(rules),
        domain_rules=_build_domain_rules(cfg.get("domain_rules")),
        path_rules=_build_path_rules(cfg.get("path_rules")),
    )


def _domain_alternation(suffixes: tuple[str, ...]) -> str:
    return "|".join(re.escape(suffix) for suffix in suffixes)


def _path_alternation(prefixes: tuple[str, ...]) -> str:
    return "|".join(re.escape(prefix.rstrip("/")) for prefix in prefixes)


def _apply_domain_rules(text: str, rules: DomainRedactRules, placeholder: str) -> str:
    if not rules.enabled or not rules.suffixes:
        return text
    out = text
    alts = _domain_alternation(rules.suffixes)
    if rules.match_urls:
        out = re.compile(rf"(?i)https?://[^\s\"']*\.(?:{alts})(?:/[^\s\"']*)?").sub(
            placeholder, out
        )
    if rules.match_hostnames:
        out = re.compile(rf"(?i)\b[\w.-]+\.(?:{alts})\b").sub(placeholder, out)
    return out


def _apply_path_rules(text: str, rules: PathRedactRules, placeholder: str) -> str:
    if not rules.enabled or not rules.prefixes:
        return text
    alts = _path_alternation(rules.prefixes)
    return re.compile(rf"(?:{alts})(?:/[^\s\"'`|,)]*)?").sub(placeholder, text)


def redact_sensitive_text(text: str, settings: OutboundRedactSettings) -> str:
    """
    对出站文本做敏感信息替换。

    @param text 原始文本
    @param settings 脱敏配置
    @return 脱敏后文本
    """
    if not settings.enabled or not text:
        return text

    out = text
    for rule in settings.regex_rules:
        if not rule.enabled:
            continue
        if rule.replacement_uses_groups:
            out = rule.pattern.sub(rule.replacement.replace("[REDACTED]", settings.placeholder), out)
        else:
            repl = rule.replacement.replace("[REDACTED]", settings.placeholder)
            out = rule.pattern.sub(repl, out)
    out = _apply_domain_rules(out, settings.domain_rules, settings.placeholder)
    out = _apply_path_rules(out, settings.path_rules, settings.placeholder)
    return out


def _redact_message_content(content: Any, settings: OutboundRedactSettings) -> Any:
    if isinstance(content, str):
        return redact_sensitive_text(content, settings)
    if isinstance(content, list):
        new_blocks: list[Any] = []
        changed = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                redacted = redact_sensitive_text(text, settings)
                if redacted != text:
                    changed = True
                    new_blocks.append({**block, "text": redacted})
                else:
                    new_blocks.append(block)
            elif isinstance(block, str):
                redacted = redact_sensitive_text(block, settings)
                if redacted != block:
                    changed = True
                new_blocks.append(redacted)
            else:
                new_blocks.append(block)
        return new_blocks if changed else content
    return content


def _redact_additional_kwargs(
    extra: dict[str, Any] | None,
    settings: OutboundRedactSettings,
) -> tuple[dict[str, Any], bool]:
    if not extra:
        return {}, False

    changed = False
    result: dict[str, Any] = {}
    for key, value in extra.items():
        if isinstance(value, str):
            redacted = redact_sensitive_text(value, settings)
            if redacted != value:
                changed = True
            result[key] = redacted
            continue
        if isinstance(value, dict):
            nested: dict[str, Any] = {}
            nested_changed = False
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, str):
                    redacted = redact_sensitive_text(nested_value, settings)
                    if redacted != nested_value:
                        nested_changed = True
                    nested[nested_key] = redacted
                else:
                    nested[nested_key] = nested_value
            if nested_changed:
                changed = True
            result[key] = nested
            continue
        result[key] = value
    return result, changed


def redact_messages_for_dispatch(
    messages: list[BaseMessage],
    settings: OutboundRedactSettings,
) -> list[BaseMessage]:
    """
    对即将发往网关的消息做脱敏（仅影响 API payload，不改落盘）。

    @param messages 出站消息
    @param settings 脱敏配置
    @return 脱敏后的消息列表
    """
    if not settings.enabled or not messages:
        return messages

    result: list[BaseMessage] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        new_content = _redact_message_content(content, settings)
        content_changed = new_content is not content

        if isinstance(msg, HumanMessage):
            if not content_changed:
                result.append(msg)
                continue
            result.append(HumanMessage(content=new_content))
            continue

        if isinstance(msg, AIMessage):
            extra, extra_changed = _redact_additional_kwargs(
                getattr(msg, "additional_kwargs", None) or {},
                settings,
            )
            if not content_changed and not extra_changed:
                result.append(msg)
                continue
            result.append(
                AIMessage(
                    content=new_content if content_changed else content,
                    tool_calls=getattr(msg, "tool_calls", None) or [],
                    additional_kwargs=extra,
                )
            )
            continue

        if isinstance(msg, ToolMessage):
            if not content_changed:
                result.append(msg)
                continue
            result.append(
                ToolMessage(
                    content=new_content if isinstance(new_content, str) else str(new_content),
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            )
            continue

        if isinstance(msg, BaseMessage):
            if not content_changed:
                result.append(msg)
                continue
            clone = msg.model_copy(deep=True)
            clone.content = new_content
            result.append(clone)
            continue

        result.append(msg)
    return result
