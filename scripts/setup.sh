#!/usr/bin/env bash
# llgraph 统一依赖安装（Python extras + Web UI npm）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUIET=false
SKIP_NPM=false
DO_BUILD=false
PROFILE="web"

usage() {
  cat <<EOF
用法: $(basename "$0") [profile] [options]

profile（默认 web）:
  web       Web Console：core + [web] + web-ui npm（推荐）
  dev       开发全量：web + terminal + index + watch + mcp + search + ast + dev
  minimal   仅核心 editable 安装
  check     检查 optional 依赖是否就绪（不安装）

options:
  -q, --quiet     减少输出
  --skip-npm      跳过 web-ui npm install
  --build         安装后执行 web-ui npm run build

示例:
  $(basename "$0")              # 装 Web 所需依赖
  $(basename "$0") dev          # 开发机全量 optional
  $(basename "$0") check
  $(basename "$0") web --build  # 装依赖并构建前端

环境变量:
  LLGRAPH_SETUP_EXTRAS   自定义 extra 列表（逗号分隔，覆盖 profile）
EOF
}

log() {
  if ! $QUIET; then
    echo "$@"
  fi
}

extras_for_profile() {
  case "$1" in
    web) echo "web" ;;
    minimal) echo "" ;;
    dev) echo "web,terminal,index,watch,mcp,search,ast,dev" ;;
    check) echo "" ;;
    *)
      echo "未知 profile: $1" >&2
      usage
      exit 1
      ;;
  esac
}

ensure_venv() {
  if [[ -d .venv ]]; then
    return
  fi
  log "创建 .venv …"
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv
  else
    python3 -m venv .venv
  fi
}

install_python_extras() {
  local extras_csv="$1"
  if [[ -z "$extras_csv" ]]; then
    log "安装 llgraph 核心 …"
    if command -v uv >/dev/null 2>&1; then
      uv sync
    else
      # shellcheck disable=SC1091
      source .venv/bin/activate
      pip install -q -U pip
      pip install -q -e .
    fi
    return
  fi

  log "安装 Python extras: $extras_csv"
  if command -v uv >/dev/null 2>&1; then
    local -a uv_args=(sync)
    local IFS=,
    for extra in $extras_csv; do
      extra="${extra// /}"
      [[ -n "$extra" ]] && uv_args+=(--extra "$extra")
    done
    uv "${uv_args[@]}"
  else
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -q -U pip
    pip install -q -e ".[$extras_csv]"
  fi
}

install_npm() {
  if $SKIP_NPM; then
    return
  fi
  if [[ ! -f web-ui/package.json ]]; then
    return
  fi
  if [[ ! -d web-ui/node_modules ]]; then
    log "安装 web-ui npm 依赖 …"
    (cd web-ui && npm install)
  else
    log "web-ui node_modules 已存在，跳过 npm install（删目录可强制重装）"
  fi
}

build_web_ui() {
  if ! $DO_BUILD; then
    return
  fi
  log "构建 web-ui …"
  (cd web-ui && npm run build)
}

run_check() {
  if command -v uv >/dev/null 2>&1; then
    uv run python - <<'PY'
from llgraph.terminal.install_extras import format_install_extras_report

print(format_install_extras_report())
PY
  else
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
    python - <<'PY'
from llgraph.terminal.install_extras import format_install_extras_report

print(format_install_extras_report())
PY
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -q|--quiet) QUIET=true; shift ;;
      --skip-npm) SKIP_NPM=true; shift ;;
      --build) DO_BUILD=true; shift ;;
      -h|--help|help) usage; exit 0 ;;
      web|dev|minimal|check) PROFILE="$1"; shift ;;
      *) echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
  done
}

main() {
  parse_args "$@"

  if [[ "$PROFILE" == "check" ]]; then
    run_check
    exit 0
  fi

  local extras_csv="${LLGRAPH_SETUP_EXTRAS:-$(extras_for_profile "$PROFILE")}"

  ensure_venv
  install_python_extras "$extras_csv"

  if [[ "$extras_csv" == *web* ]] || [[ "$PROFILE" == "dev" ]]; then
    install_npm
    build_web_ui
  fi

  log ""
  log "完成。Web 启动: llgraph web  或  ./scripts/web-dev.sh dev"
  if ! $QUIET; then
    run_check
  fi
}

main "$@"
