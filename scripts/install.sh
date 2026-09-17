#!/usr/bin/env bash
# 一键安装 llgraph（macOS / Linux）
#
#   curl -fsSL https://raw.githubusercontent.com/xuhang113/llgraph/cursor/auto_upgrade/scripts/install.sh | bash
#
# 本仓内执行则装到当前克隆；管道执行则克隆到 ~/.local/share/llgraph。
# 不读取、不打印、不写入 API Key。
set -euo pipefail

REPO_URL="${LLGRAPH_REPO:-https://github.com/xuhang113/llgraph.git}"
REPO_REF="${LLGRAPH_REF:-cursor/auto_upgrade}"
DEFAULT_HOME="${HOME}/.local/share/llgraph"
BIN_DIR="${LLGRAPH_BIN_DIR:-${HOME}/.local/bin}"
EXTRAS="${LLGRAPH_EXTRAS:-index,watch,mcp,models}"
WITH_WEB="${LLGRAPH_WITH_WEB:-0}"

usage() {
  cat <<EOF
用法: install.sh

环境变量:
  LLGRAPH_REPO       git 地址（默认 ${REPO_URL}）
  LLGRAPH_REF        分支或 tag（默认 ${REPO_REF}）
  LLGRAPH_HOME       安装目录（管道安装默认 ${DEFAULT_HOME}）
  LLGRAPH_BIN_DIR    可执行文件目录（默认 ${BIN_DIR}）
  LLGRAPH_EXTRAS     pip extras，逗号分隔（默认 ${EXTRAS}）
  LLGRAPH_WITH_WEB=1 额外安装 [web] 并构建 Web Console

在已有克隆里直接跑本脚本时，默认装到该克隆，不改 LLGRAPH_HOME。
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

log() { printf '%s\n' "$*"; }
die() { printf 'llgraph 安装失败: %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "找不到命令 $1"
}

python_ok() {
  local bin="$1"
  command -v "$bin" >/dev/null 2>&1 || return 1
  "$bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null
}

pick_python() {
  local bin
  for bin in python3.13 python3.12 python3; do
    if python_ok "$bin"; then
      printf '%s\n' "$bin"
      return 0
    fi
  done
  return 1
}

in_llgraph_checkout() {
  local here src
  [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]] || return 1
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  src="$(cd "${here}/.." && pwd)"
  [[ -f "${src}/pyproject.toml" ]] || return 1
  grep -q '^name = "llgraph"' "${src}/pyproject.toml" || return 1
  printf '%s\n' "$src"
}

clone_or_update() {
  local dest="$1"
  need_cmd git
  if [[ -d "${dest}/.git" ]]; then
    log "更新已有克隆 ${dest} （${REPO_REF}）…"
    git -C "$dest" fetch --depth 1 origin "$REPO_REF"
    git -C "$dest" checkout -B "$REPO_REF" FETCH_HEAD
    return
  fi
  if [[ -e "$dest" && ! -d "${dest}/.git" ]]; then
    die "${dest} 已存在且不是 git 仓库，换一个 LLGRAPH_HOME"
  fi
  log "克隆 ${REPO_URL} @ ${REPO_REF} → ${dest} …"
  mkdir -p "$(dirname "$dest")"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$dest"
}

install_python() {
  local root="$1"
  local py extras_csv extras_arg
  extras_csv="$EXTRAS"
  if [[ "$WITH_WEB" == "1" ]]; then
    if [[ -n "$extras_csv" ]]; then
      extras_csv="${extras_csv},web"
    else
      extras_csv="web"
    fi
  fi
  py="$(pick_python)" || die "需要 Python 3.12+（python3.12 / python3.13 / python3）"
  log "使用 $($py -V)"
  cd "$root"
  if [[ ! -d .venv ]]; then
    log "创建虚拟环境 .venv …"
    if command -v uv >/dev/null 2>&1; then
      uv venv --python "$py" .venv
    else
      "$py" -m venv .venv
    fi
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  extras_arg="."
  if [[ -n "$extras_csv" ]]; then
    extras_arg=".[${extras_csv}]"
  fi
  log "安装 ${extras_arg} （含索引依赖时首次会较久）…"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -p .venv/bin/python -e "$extras_arg"
  else
    python -m pip install -U pip
    python -m pip install -e "$extras_arg"
  fi
}

link_bin() {
  local root="$1"
  local src dest
  src="${root}/.venv/bin/llgraph"
  [[ -x "$src" ]] || die "未生成 ${src}"
  mkdir -p "$BIN_DIR"
  dest="${BIN_DIR}/llgraph"
  ln -sfn "$src" "$dest"
  log "命令: ${dest}"
}

seed_config() {
  local root="$1"
  local cfg_dir example dest
  cfg_dir="${HOME}/.config/llgraph"
  mkdir -p "$cfg_dir"
  example="${root}/examples/llgraph.env.example"
  dest="${cfg_dir}/llgraph.env"
  if [[ -f "$example" && ! -f "$dest" ]]; then
    cp "$example" "$dest"
    log "已写入 ${dest}（示例值，请自己改网关和密钥，不要提交）"
  elif [[ -f "$dest" ]]; then
    log "保留已有 ${dest}"
  fi
  if [[ -x "${root}/.venv/bin/llgraph" ]]; then
    "${root}/.venv/bin/llgraph" --init-user-config >/dev/null 2>&1 || true
  fi
}

maybe_web() {
  local root="$1"
  [[ "$WITH_WEB" == "1" ]] || return 0
  need_cmd npm
  log "构建 Web Console …"
  (cd "${root}/web-ui" && npm install && npm run build)
}

check_rg() {
  if command -v rg >/dev/null 2>&1; then
    return
  fi
  log "未找到 ripgrep（rg）。grep/glob 会变慢。macOS: brew install ripgrep"
}

check_path() {
  case ":${PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *)
      log "把 ${BIN_DIR} 加进 PATH，例如："
      log "  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
      ;;
  esac
}

main() {
  local root
  if [[ -n "${LLGRAPH_HOME:-}" ]]; then
    root="${LLGRAPH_HOME}"
    clone_or_update "$root"
  elif root="$(in_llgraph_checkout)"; then
    log "在已有克隆中安装: ${root}"
  else
    root="$DEFAULT_HOME"
    clone_or_update "$root"
  fi
  [[ -f "${root}/pyproject.toml" ]] || die "${root} 不是 llgraph 仓库"
  install_python "$root"
  maybe_web "$root"
  link_bin "$root"
  seed_config "$root"
  check_rg
  check_path
  log ""
  log "装好了。下一步："
  log "  1. 编辑 ~/.config/llgraph/llgraph.env"
  log "  2. llgraph --init-config -C /path/to/workspace"
  log "  3. llgraph -C /path/to/workspace"
}

main "$@"
