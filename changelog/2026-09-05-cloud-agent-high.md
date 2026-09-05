# 2026-09-05 Cloud Agent 改为 Opus High

选题：省 token（Max 一轮一两千万）。

## 做了什么

- 定时 Cloud Agent 从 Opus 5 Max 改为 High（`effort=high`）
- 候选列表不再尝试 max / xhigh

## 改了哪些路径

- `.github/workflows/main.yml`

## 怎么验收

- `main` 上 workflow 硬编码候选只有 `effort=high`
- 下次定时日志 `using model` 含 high，不含 max / xhigh
- 不要为这次小改动手动 Run

## 未做 / 下一步不要做

- 定时仍是每天北京时间 10:00 一次
- 不要自行改回 Max
