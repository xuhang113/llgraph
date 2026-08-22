"""写入后语法诊断：对齐 Cursor Apply 后把 lint/红线回注给 Agent。

search_replace / write_file 成功不等于代码可解析。商用 Agent 会把当前文件
诊断立刻塞进工具结果，模型在同一 ReAct 循环里修，而不是声称完成。
本模块只做**本地、同步、语法级**检查（不跑完整 LSP / 测试），失败则静默。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

DIAGNOSTIC_MARKER = "[语法诊断]"
_MAX_ISSUES = 6
_SPAN_PAD_LINES = 8
_MSG_DIGITS = re.compile(r"\d+")

_PYTHON_SUFFIX = frozenset({".py", ".pyw", ".pyi"})
_JSON_SUFFIX = frozenset({".json"})
_YAML_SUFFIX = frozenset({".yaml", ".yml"})
_TOML_SUFFIX = frozenset({".toml"})
_JS_SUFFIX = frozenset({".js", ".mjs", ".cjs", ".jsx"})
_JAVA_SUFFIX = frozenset({".java"})


@dataclass(frozen=True)
class SyntaxIssue:
    """单条语法问题。"""

    line: int
    column: int
    message: str
    kind: str = "syntax"


def collect_syntax_issues(path: str, text: str) -> list[SyntaxIssue]:
    """
    按后缀做语法解析；不支持的语言或空文件返回空。

    @param path 相对路径（用后缀选解析器）
    @param text 文件全文
    @return 问题列表（最多扫描，截断在格式化时做）
    """
    if not (text or "").strip():
        return []
    suffix = Path(path or "").suffix.lower()
    try:
        if suffix in _PYTHON_SUFFIX:
            return _python_issues(text)
        if suffix in _JSON_SUFFIX:
            return _json_issues(text)
        if suffix in _YAML_SUFFIX:
            return _yaml_issues(text)
        if suffix in _TOML_SUFFIX:
            return _toml_issues(text)
        if suffix in _JS_SUFFIX:
            return _tree_sitter_issues(text, "javascript") or _node_check_issues(text)
        if suffix in _JAVA_SUFFIX:
            return _tree_sitter_issues(text, "java")
    except Exception:
        return []
    return []


def format_edit_diagnostics(
    rel: str,
    new_text: str,
    *,
    old_text: str = "",
    change_span: tuple[int, int] | None = None,
) -> str:
    """
    生成可追加在写工具成功文案后的诊断块；无新问题则空串（避免浪费 token）。

    @param rel 相对路径
    @param new_text 写入后全文
    @param old_text 写入前全文（用于忽略既有问题）
    @param change_span 1-based 含端改动行区间；None 时根据 old/new 推断
    @return 诊断块或空
    """
    new_issues = collect_syntax_issues(rel, new_text)
    if not new_issues:
        return ""
    old_issues = collect_syntax_issues(rel, old_text) if old_text.strip() else []
    span = change_span
    if span is None and old_text:
        from llgraph.core.edit_apply import changed_line_span

        span = changed_line_span(old_text, new_text)
    selected = _select_new_issues(
        new_text,
        new_issues,
        old_text=old_text,
        old_issues=old_issues,
        change_span=span,
    )
    if not selected:
        return ""
    selected = selected[:_MAX_ISSUES]
    lines = [
        f"{DIAGNOSTIC_MARKER} 写入后发现 {len(selected)} 个新问题"
        "（对标 Cursor Apply 后的 lint 回注）。"
        "不要声称已完成；请立即 search_replace 修复：",
    ]
    for issue in selected:
        loc = f"{rel}:L{issue.line}"
        if issue.column > 0:
            loc += f":C{issue.column}"
        msg = issue.message.strip().replace("\n", " ")
        if len(msg) > 160:
            msg = msg[:157] + "…"
        lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)


def _select_new_issues(
    new_text: str,
    new_issues: list[SyntaxIssue],
    *,
    old_text: str,
    old_issues: list[SyntaxIssue],
    change_span: tuple[int, int] | None,
) -> list[SyntaxIssue]:
    if not old_issues:
        return list(new_issues)
    old_sigs = {_issue_signature(old_text, item) for item in old_issues}
    lo = hi = 0
    if change_span:
        lo, hi = change_span
        lo = max(1, lo - _SPAN_PAD_LINES)
        hi = hi + _SPAN_PAD_LINES
    out: list[SyntaxIssue] = []
    for issue in new_issues:
        sig = _issue_signature(new_text, issue)
        if sig in old_sigs:
            if change_span and lo <= issue.line <= hi:
                out.append(issue)
            continue
        out.append(issue)
    return out


def _issue_signature(text: str, issue: SyntaxIssue) -> tuple[str, str, str]:
    msg = _MSG_DIGITS.sub("N", issue.message).strip().lower()[:180]
    return (issue.kind, msg, _line_at(text, issue.line).strip()[:80])


def _line_at(text: str, line: int) -> str:
    lines = text.splitlines()
    idx = line - 1
    if 0 <= idx < len(lines):
        return lines[idx]
    return ""


def _python_issues(text: str) -> list[SyntaxIssue]:
    try:
        ast.parse(text)
        return []
    except SyntaxError as exc:
        return [
            SyntaxIssue(
                line=int(exc.lineno or 1),
                column=int(exc.offset or 0),
                message=str(exc.msg or "SyntaxError"),
            )
        ]


def _json_issues(text: str) -> list[SyntaxIssue]:
    try:
        json.loads(text)
        return []
    except json.JSONDecodeError as exc:
        return [
            SyntaxIssue(
                line=int(exc.lineno or 1),
                column=int(exc.colno or 0),
                message=str(exc.msg or "JSONDecodeError"),
            )
        ]


def _yaml_issues(text: str) -> list[SyntaxIssue]:
    try:
        import yaml
    except ImportError:
        return []
    try:
        yaml.safe_load(text)
        return []
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = int(getattr(mark, "line", 0) or 0) + 1
        col = int(getattr(mark, "column", 0) or 0) + 1
        msg = str(exc).split("\n", 1)[0].strip() or "YAMLError"
        return [SyntaxIssue(line=max(1, line), column=max(0, col), message=msg)]


def _toml_issues(text: str) -> list[SyntaxIssue]:
    import tomllib

    try:
        tomllib.loads(text)
        return []
    except tomllib.TOMLDecodeError as exc:
        msg = str(exc).strip() or "TOMLDecodeError"
        line = 1
        found = re.search(r"line\s+(\d+)", msg, re.IGNORECASE)
        if found:
            line = max(1, int(found.group(1)))
        return [SyntaxIssue(line=line, column=0, message=msg[:160])]


def _tree_sitter_issues(text: str, language: str) -> list[SyntaxIssue]:
    try:
        from tree_sitter import Parser
    except ImportError:
        return []
    lang_obj = _tree_sitter_language(language)
    if lang_obj is None:
        return []
    try:
        parser = Parser(lang_obj)
        tree = parser.parse(text.encode("utf-8"))
    except Exception:
        return []
    issues: list[SyntaxIssue] = []

    def walk(node) -> None:
        if len(issues) >= _MAX_ISSUES:
            return
        if node.type == "ERROR" or getattr(node, "is_missing", False):
            row, col = node.start_point
            snippet = text.splitlines()[row] if row < len(text.splitlines()) else ""
            label = "missing node" if getattr(node, "is_missing", False) else "parse error"
            if snippet.strip():
                label = f"{label} near {snippet.strip()[:40]!r}"
            issues.append(SyntaxIssue(line=row + 1, column=col + 1, message=label))
            return
        for child in node.children:
            walk(child)
            if len(issues) >= _MAX_ISSUES:
                return

    walk(tree.root_node)
    return issues


def _tree_sitter_language(language: str):
    try:
        from tree_sitter import Language
    except ImportError:
        return None
    try:
        if language == "javascript":
            import tree_sitter_javascript as tsjs

            return Language(tsjs.language())
        if language == "java":
            import tree_sitter_java as tsjava

            return Language(tsjava.language())
        if language == "python":
            import tree_sitter_python as tspy

            return Language(tspy.language())
    except Exception:
        return None
    return None


def _node_check_issues(text: str) -> list[SyntaxIssue]:
    import shutil
    import subprocess
    import tempfile

    if shutil.which("node") is None:
        return []
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".js",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(text)
            path = handle.name
        proc = subprocess.run(
            ["node", "--check", path],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    finally:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
    if proc.returncode == 0:
        return []
    err = (proc.stderr or proc.stdout or "").strip()
    if not err:
        return [SyntaxIssue(line=1, column=0, message="node --check failed")]
    line = 1
    col = 0
    found = re.search(r":(\d+):(\d+)", err)
    if found:
        line = max(1, int(found.group(1)))
        col = max(0, int(found.group(2)))
    msg = err.splitlines()[-1].strip()[:160] or "JavaScript syntax"
    return [SyntaxIssue(line=line, column=col, message=msg)]
