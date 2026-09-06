"""加载 llgraph 专用 API 凭据（与 Claude CLI / ANTHROPIC_* 完全独立）。"""

import os
from pathlib import Path

from dotenv import load_dotenv

# llgraph 独立配置目录（不读取 claude-code.env）
LLGRAPH_CONFIG_DIR = Path.home() / ".config" / "llgraph"
LLGRAPH_ENV_FILE = LLGRAPH_CONFIG_DIR / "llgraph.env"
LLGRAPH_ENV_EXAMPLE = LLGRAPH_CONFIG_DIR / "llgraph.env.example"

# 项目内 .env 可覆盖（开发调试用）
_PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"

# 环境变量名（唯一来源，禁止回退 ANTHROPIC_*）
ENV_API_BASE_URL = "LLGRAPH_API_BASE_URL"
ENV_API_KEY = "LLGRAPH_API_KEY"
ENV_MODEL = "LLGRAPH_MODEL"

# 置 1 时只认进程内已 export 的变量，不读磁盘 env 文件（单测隔离用）
ENV_IGNORE_ENV_FILES = "LLGRAPH_IGNORE_ENV_FILES"

DEFAULT_MODEL = "claude-opus-4-6"


def env_files_ignored() -> bool:
    """@return 是否跳过磁盘 env 文件（只认已 export 的变量）"""
    return os.getenv(ENV_IGNORE_ENV_FILES, "").strip().lower() in ("1", "true", "yes", "on")


def load_llgraph_env() -> None:
    """
    加载顺序（后者可覆盖前者）：
    1. ~/.config/llgraph/llgraph.env
    2. 项目根目录 .env（llgraph 仓库内，可选）
    3. 当前 shell 已 export 的变量保持不变

    置 ``LLGRAPH_IGNORE_ENV_FILES=1`` 时两个文件都不读。
    """
    if env_files_ignored():
        return
    if LLGRAPH_ENV_FILE.is_file():
        load_dotenv(LLGRAPH_ENV_FILE, override=False)
    if _PROJECT_ENV.is_file():
        load_dotenv(_PROJECT_ENV, override=True)


def resolve_configured_model() -> str:
    """
    读取配置的模型 id；缺网关凭据也不报错。

    模型名与凭据是两件事：上下文窗口、dispatch profile、缓存键、trace 展示
    都只需要模型名，不该因为没配 key 就崩。

    @return 模型 id；未配置时为 DEFAULT_MODEL
    """
    load_llgraph_env()
    return os.getenv(ENV_MODEL, DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_llgraph_settings() -> dict[str, str]:
    """
    读取 llgraph API 必填项；缺失时抛出清晰错误。

    @return base_url、api_key、model
    """
    model = resolve_configured_model()
    base_url = os.getenv(ENV_API_BASE_URL, "").strip()
    api_key = os.getenv(ENV_API_KEY, "").strip()

    missing: list[str] = []
    if not base_url:
        missing.append(ENV_API_BASE_URL)
    if not api_key:
        missing.append(ENV_API_KEY)
    if missing:
        hint = (
            f"请配置 llgraph 专用凭据：复制 {LLGRAPH_ENV_EXAMPLE} 为 {LLGRAPH_ENV_FILE} 并填写，"
            f"或在项目 .env 中设置 {ENV_API_BASE_URL} / {ENV_API_KEY}。"
            "（不复用 Claude CLI 的 claude-code.env / ANTHROPIC_*）"
        )
        raise RuntimeError(f"缺少环境变量: {', '.join(missing)}。{hint}")

    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model": model,
    }


def get_llgraph_api_credentials() -> tuple[str, str]:
    """
    返回 (base_url, api_key)，供 embedding remote 等复用。

    @return (base_url, api_key)
    """
    settings = get_llgraph_settings()
    return settings["base_url"], settings["api_key"]


# 兼容旧函数名（内部仍只读 LLGRAPH_*）
def load_gateway_env() -> None:
    """已废弃别名，请使用 load_llgraph_env。"""
    load_llgraph_env()


def get_gateway_settings() -> dict[str, str]:
    """已废弃别名，请使用 get_llgraph_settings。"""
    return get_llgraph_settings()


def get_embedding_settings() -> dict[str, str | int]:
    """
    读取远程 Embedding API 配置（provider=remote 时使用）。

    默认 embedding 已改为本地模型；见 .llgraph/embedding.json。
    """
    from llgraph.code_index.embedding_config import resolve_embedding_profile

    profile = resolve_embedding_profile(Path(".").resolve())
    if profile.provider != "remote":
        raise RuntimeError(
            "当前 embedding provider 为 local，无需远程 API。"
            "若需远程，请在 .llgraph/embedding.json 中设置 \"provider\": \"remote\"。"
        )
    return {
        "base_url": profile.base_url or "",
        "api_key": profile.api_key or "",
        "model": profile.model,
        "dimension": profile.dimension or 1536,
    }
