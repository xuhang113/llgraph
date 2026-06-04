"""内置 /review 命令（P4）。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

from llgraph.config.edit_settings import load_agent_config
from llgraph.core.llm import create_gateway_llm
from llgraph.session.session_edits import SessionEditTracker


@dataclass(frozen=True)
class ReviewSettings:
    """Review 配置。"""

    output_dir: Path
    spec_paths: tuple[str, ...]
    exclude_prefixes: tuple[str, ...]


def resolve_review_settings(workspace: Path) -> ReviewSettings:
    """
    解析 review 配置。

    @param workspace 工作区根
    @return ReviewSettings
    """
    cfg = load_agent_config(workspace)
    review = cfg.get("review") if isinstance(cfg.get("review"), dict) else {}

    out = review.get("output_dir", "~/llgraph-review")
    output_dir = Path(str(out)).expanduser()

    specs = review.get("spec_paths") or ["CONTRIBUTING.md"]
    if isinstance(specs, str):
        specs = [specs]
    spec_paths = tuple(str(s) for s in specs) if isinstance(specs, list) else ("CONTRIBUTING.md",)

    excludes = review.get("exclude_prefixes") or [
        ".cursor/", "node_modules/", ".llgraph/", ".git/",
    ]
    if isinstance(excludes, str):
        excludes = [excludes]
    exclude_prefixes = tuple(str(x) for x in excludes) if isinstance(excludes, list) else ()

    return ReviewSettings(
        output_dir=output_dir,
        spec_paths=spec_paths,
        exclude_prefixes=exclude_prefixes,
    )


def _resolve_spec_paths(workspace: Path, spec_paths: tuple[str, ...]) -> list[str]:
    """
    解析规范文件绝对路径（支持相对工作区根）。

    @param workspace 工作区根
    @param spec_paths 配置中的规范路径
    @return 存在的规范文件路径列表
    """
    resolved: list[str] = []
    for spec in spec_paths:
        candidate = Path(spec).expanduser()
        if not candidate.is_file():
            candidate = workspace / spec
        if candidate.is_file():
            resolved.append(str(candidate.resolve()))
    return resolved


def _find_git_root_for_path(workspace: Path, rel: str) -> tuple[Path, str] | None:
    """
    为工作区内相对路径查找所属 git 仓库根与仓库内相对路径。

    @param workspace 工作区根
    @param rel 工作区相对路径
    @return (git_root, path_in_repo) 或 None
    """
    ws = workspace.resolve()
    full = (ws / rel.strip().lstrip("/")).resolve()
    try:
        full.relative_to(ws)
    except ValueError:
        return None

    current = full.parent if full.is_file() else full
    while True:
        if (current / ".git").exists():
            if full.is_file():
                path_in_repo = str(full.relative_to(current))
            else:
                path_in_repo = rel.strip().lstrip("/")
            return current, path_in_repo
        if current == ws:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    if (ws / ".git").exists():
        return ws, rel.strip().lstrip("/")
    return None


def _git_branch(workspace: Path, paths: list[str] | None = None) -> str:
    """
    解析 git 分支名；工作区非 git 仓库时尝试从变更路径推断子仓库。

    @param workspace 工作区根
    @param paths 会话变更路径
    @return 分支名或 unknown
    """
    candidates: list[Path] = [workspace.resolve()]
    if paths:
        for rel in paths[:5]:
            found = _find_git_root_for_path(workspace, rel)
            if found is not None:
                candidates.append(found[0])
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
    return "unknown"


def _git_diff_at(root: Path, paths_in_repo: list[str], *, max_chars: int) -> str:
    """
    在指定 git 根目录执行 diff。

    @param root git 仓库根
    @param paths_in_repo 仓库内相对路径
    @param max_chars 字符上限
    @return diff 文本
    """
    if not paths_in_repo:
        for args in (["git", "diff", "--staged"], ["git", "diff"]):
            try:
                out = subprocess.run(
                    args,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if out.stdout.strip():
                    return out.stdout[:max_chars]
            except (OSError, subprocess.TimeoutExpired):
                continue
        return ""

    try:
        out = subprocess.run(
            ["git", "diff", "--", *paths_in_repo[:20]],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.stdout.strip():
            return out.stdout[:max_chars]
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return ""


def _git_diff(workspace: Path, paths: list[str], *, max_chars: int = 12000) -> str:
    """
    收集 git diff；支持 monorepo 根目录下嵌套 git 子仓库。

    @param workspace 工作区根
    @param paths 工作区相对路径
    @param max_chars 字符上限
    @return diff 文本
    """
    if not paths:
        return _git_diff_at(workspace, [], max_chars=max_chars)

    by_root: dict[Path, list[str]] = {}
    for rel in paths:
        found = _find_git_root_for_path(workspace, rel)
        if found is None:
            continue
        root, path_in_repo = found
        bucket = by_root.setdefault(root, [])
        if path_in_repo not in bucket:
            bucket.append(path_in_repo)

    chunks: list[str] = []
    remaining = max_chars
    for root, rels in by_root.items():
        part = _git_diff_at(root, rels, max_chars=remaining)
        if not part.strip():
            continue
        header = f"--- git repo: {root} ---\n"
        block = header + part
        if len(block) > remaining:
            block = block[:remaining] + "\n…（截断）\n"
        chunks.append(block)
        remaining -= len(block)
        if remaining <= 200:
            break
    return "\n".join(chunks).strip()


def _filter_paths(paths: list[str], settings: ReviewSettings) -> list[str]:
    filtered: list[str] = []
    for rel in paths:
        if any(rel.startswith(p.rstrip("/")) or f"/{p}" in f"/{rel}" for p in settings.exclude_prefixes):
            continue
        filtered.append(rel)
    return filtered


def _collect_review_diff_text(
    workspace: Path,
    edit_tracker: SessionEditTracker | None,
    paths: list[str],
    *,
    max_chars: int = 12000,
) -> tuple[str, str]:
    """
    收集评审用 diff：优先会话快照，其次嵌套/根 git diff。

    @param workspace 工作区根
    @param edit_tracker 会话编辑账本
    @param paths 评审文件列表
    @param max_chars 字符上限
    @return (diff_text, source_label)
    """
    if edit_tracker is not None and paths:
        session_diff = edit_tracker.collect_review_diff(paths, max_chars=max_chars)
        if session_diff.strip():
            return session_diff, "session_snapshot"

    git_diff = _git_diff(workspace, paths, max_chars=max_chars)
    if git_diff.strip():
        return git_diff, "git"

    if not paths:
        git_diff = _git_diff(workspace, [], max_chars=max_chars)
        if git_diff.strip():
            return git_diff, "git"
    return "", "empty"


def run_review(
    workspace: Path,
    *,
    topic: str,
    edit_tracker: SessionEditTracker | None = None,
    last_user_message: str = "",
) -> tuple[Path, str]:
    """
    执行代码评审并落盘。

    @param workspace 工作区根
    @param topic 评审主题
    @param edit_tracker 会话变更账本
    @param last_user_message 最近用户消息
    @return (review_file_path, chat_summary)
    """
    settings = resolve_review_settings(workspace)
    raw_paths = (
        edit_tracker.paths_for_review()
        if edit_tracker is not None
        else []
    )
    paths = _filter_paths(raw_paths, settings)
    diff_text, diff_source = _collect_review_diff_text(
        workspace,
        edit_tracker,
        paths,
    )
    branch = _git_branch(workspace, paths)
    repo_name = workspace.name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = settings.output_dir / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    review_path = out_dir / f"{branch}-{ts}.md"

    spec_notes = _resolve_spec_paths(workspace, settings.spec_paths)

    source_note = {
        "session_snapshot": "会话首次编辑快照 vs 当前磁盘（/diff 同源）",
        "git": "git diff（含嵌套子仓库）",
        "empty": "未收集到 diff",
    }.get(diff_source, diff_source)

    prompt = f"""你是代码评审助手。请对以下变更做代码评审，输出 Markdown。

## 主题
{topic or last_user_message or '（未指定）'}

## 评审范围文件（本会话 Agent 改动）
{chr(10).join(paths) if paths else '（无会话变更记录）'}

## diff 来源
{source_note}

## 规范参考
{chr(10).join(spec_notes) if spec_notes else '（未找到 review.spec_paths 中的规范文件，请按通用最佳实践评审）'}

## 变更 diff
```diff
{diff_text or '（无 diff）'}
```

请输出结构：
1. ## 结论摘要（2～4 句）
2. ## 问题列表（按严重级别：严重/一般/建议）
3. ## 规范对照（引用规范章节）
4. ## 测试建议
只基于给定 diff 与文件列表，不要臆造未出现的代码。
"""
    llm = create_gateway_llm(workspace)
    response = llm.invoke([HumanMessage(content=prompt)])
    body = getattr(response, "content", str(response))
    if isinstance(body, list):
        text_parts: list[str] = []
        for block in body:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                text_parts.append(block)
        body = "".join(text_parts)
    elif not isinstance(body, str):
        body = str(body)

    header = (
        f"# Code Review: {topic or repo_name}\n\n"
        f"- workspace: `{workspace}`\n"
        f"- branch: `{branch}`\n"
        f"- files: {len(paths)}\n"
        f"- diff_source: `{diff_source}`\n"
        f"- generated: {datetime.now(timezone.utc).isoformat()}\n\n"
    )
    review_path.write_text(header + str(body), encoding="utf-8")

    summary_lines = str(body).strip().splitlines()[:8]
    chat_summary = "\n".join(summary_lines)
    return review_path, chat_summary
