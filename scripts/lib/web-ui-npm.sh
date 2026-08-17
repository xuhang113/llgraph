#!/usr/bin/env bash
# web-ui npm 依赖同步（setup / web-dev 共用）

web_ui_needs_npm_install() {
  local root="${1:?}"
  local pkg="$root/web-ui/package.json"
  local lock="$root/web-ui/package-lock.json"
  local modules="$root/web-ui/node_modules"
  local stamp="$modules/.package-lock.json"

  [[ -f "$pkg" ]] || return 1
  [[ ! -d "$modules" ]] && return 0
  [[ ! -f "$stamp" ]] && return 0
  [[ -f "$lock" && "$lock" -nt "$stamp" ]] && return 0
  [[ "$pkg" -nt "$stamp" ]] && return 0
  return 1
}

web_ui_ensure_npm() {
  local root="${1:?}"
  local quiet="${2:-false}"

  if ! web_ui_needs_npm_install "$root"; then
    return 0
  fi
  if [[ "$quiet" != "true" ]]; then
    echo "同步 web-ui npm 依赖 …"
  fi
  (cd "$root/web-ui" && npm install)
}
