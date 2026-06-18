#!/usr/bin/env bash
# llgraph Web Console 开发脚本（可选 UI；集成请用 llgraph.console 库）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_DIR="$ROOT/.run"
API_PID_FILE="$RUN_DIR/web-api.pid"
VITE_PID_FILE="$RUN_DIR/web-vite.pid"
API_LOG="$RUN_DIR/web-api.log"
VITE_LOG="$RUN_DIR/web-vite.log"

API_HOST="${LLGRAPH_WEB_HOST:-127.0.0.1}"
API_PORT="${LLGRAPH_WEB_PORT:-8765}"
VITE_PORT="${LLGRAPH_WEB_VITE_PORT:-5173}"

usage() {
  cat <<EOF
用法: $(basename "$0") <command> [options]

命令:
  start [--dev] [--build]   后台启动 Web UI（单端口 8765 托管 web-ui/dist）
  stop                      停止后台服务
  status                    查看运行状态
  dev                       前台开发（llgraph web + Vite 热更新）

示例:
  $(basename "$0") dev
  $(basename "$0") start --build

环境变量:
  LLGRAPH_WEB_HOST / LLGRAPH_WEB_PORT / LLGRAPH_WEB_VITE_PORT / LLGRAPH_WEB_STATIC
EOF
}

ensure_env() {
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q -e ".[web]"
  if [[ ! -d web-ui/node_modules ]]; then
    (cd web-ui && npm install)
  fi
  mkdir -p "$RUN_DIR"
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    cat "$file"
  fi
}

pids_on_port() {
  lsof -t -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

kill_port() {
  local port="$1"
  local pids
  pids=$(pids_on_port "$port")
  if [[ -n "$pids" ]]; then
    echo "结束端口 $port 上的进程: $pids"
    kill $pids 2>/dev/null || true
    sleep 0.3
    pids=$(pids_on_port "$port")
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

stop_process() {
  local name="$1"
  local pid_file="$2"
  local port="$3"
  local pid
  pid=$(read_pid_file "$pid_file")
  if pid_alive "$pid"; then
    echo "停止 $name (PID $pid)…"
    kill "$pid" 2>/dev/null || true
    sleep 0.3
    if pid_alive "$pid"; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
  kill_port "$port"
}

cmd_stop() {
  stop_process "Web API" "$API_PID_FILE" "$API_PORT"
  stop_process "Vite" "$VITE_PID_FILE" "$VITE_PORT"
  echo "已停止 llgraph Web UI"
}

ensure_port_free() {
  local port="$1"
  if [[ -n $(pids_on_port "$port") ]]; then
    echo "端口 $port 已被占用，尝试释放…"
    kill_port "$port"
  fi
}

cmd_status() {
  local api_pid vite_pid
  api_pid=$(read_pid_file "$API_PID_FILE")
  vite_pid=$(read_pid_file "$VITE_PID_FILE")
  echo "llgraph Web UI"
  echo "  API  $API_HOST:$API_PORT"
  if pid_alive "$api_pid"; then
    echo "    运行中 PID=$api_pid"
  else
    echo "    未运行"
  fi
  echo "  Vite 127.0.0.1:$VITE_PORT"
  if pid_alive "$vite_pid"; then
    echo "    运行中 PID=$vite_pid"
  else
    echo "    未运行"
  fi
}

cmd_start() {
  local dev_mode=false
  local do_build=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dev) dev_mode=true; shift ;;
      --build) do_build=true; shift ;;
      *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
  done
  ensure_env
  cmd_stop 2>/dev/null || true

  if $dev_mode; then
    ensure_port_free "$API_PORT"
    ensure_port_free "$VITE_PORT"
    nohup llgraph web --host "$API_HOST" --port "$API_PORT" >>"$API_LOG" 2>&1 &
    echo $! >"$API_PID_FILE"
    nohup npm --prefix web-ui run dev -- --host 127.0.0.1 --port "$VITE_PORT" >>"$VITE_LOG" 2>&1 &
    echo $! >"$VITE_PID_FILE"
    echo "UI: http://127.0.0.1:$VITE_PORT  API: http://$API_HOST:$API_PORT"
    return
  fi

  if [[ ! -d web-ui/dist ]] || $do_build; then
    (cd web-ui && npm run build)
  fi
  ensure_port_free "$API_PORT"
  export LLGRAPH_WEB_STATIC="$ROOT/web-ui/dist"
  nohup env LLGRAPH_WEB_STATIC="$LLGRAPH_WEB_STATIC" \
    llgraph web --host "$API_HOST" --port "$API_PORT" >>"$API_LOG" 2>&1 &
  echo $! >"$API_PID_FILE"
  echo "打开: http://$API_HOST:$API_PORT"
}

cmd_dev() {
  ensure_env
  ensure_port_free "$API_PORT"
  llgraph web --host "$API_HOST" --port "$API_PORT" &
  local api_pid=$!
  echo $api_pid >"$API_PID_FILE"
  trap 'kill "$api_pid" 2>/dev/null || true; rm -f "$API_PID_FILE"' EXIT INT TERM
  echo "API: http://$API_HOST:$API_PORT"
  echo "UI:  http://127.0.0.1:$VITE_PORT"
  (cd web-ui && npm run dev -- --host 127.0.0.1 --port "$VITE_PORT")
}

main() {
  local cmd="${1:-dev}"
  shift || true
  case "$cmd" in
    start) cmd_start "$@" ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    dev) cmd_dev ;;
    -h|--help|help) usage ;;
    *) echo "未知命令: $cmd"; usage; exit 1 ;;
  esac
}

main "$@"
