# 2026-09-05 定时改回每天一次

选题：省 token（Cloud Agent 每天两次太烧 Other 池）。

## 做了什么

- Cloud Agent 定时从每天北京时间 10:00、22:00 改回只在 10:00 一次

## 改了哪些路径

- `.github/workflows/main.yml`
- `AGENTS.md`

## 怎么验收

- `main` 上 `main.yml` 只有一条 `cron: "0 2 * * *"`，没有 `0 14 * * *`
- 不要为这次小改动手动 Run workflow

## 未做 / 下一步不要做

- 没有改模型（仍是 Opus 5 Max）
- 不要再加回晚上那一次，除非人手要求
