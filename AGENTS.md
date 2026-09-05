# llgraph · Agent 开发规范

给 Cursor Cloud Agent / 本机 Agent 用。

开工先读：本文件 → `changelog/` 最近 3 条 → 下手文件。对照 `docs/cursor-agent.md`、`docs/项目结构.md`、`docs/模块说明.md`。不要全库通读。

不要把 API Key、网关令牌、`.env` 写进代码或文档。

## 两种开工

**人手一轮：** 只做用户点名的事，不顺手加系统。

**定时自动迭代**（`cursor/auto_upgrade`，每天北京时间 10:00、22:00）：

- 直接在当前分支提交，不要开新分支，不要 PR。
- 按影响力选一件做深：速度 / 性能 / 稳定性 / 商用体验。两件都很小才可顺带第二件。
- 先看 changelog 已有方向，接着做，不要推倒重来。
- 没有高价值切口：只写 changelog 说明下轮攻哪，不要硬改。

## 产品

终端侧 Cursor 式 Agent：LangGraph ReAct + OpenAI 兼容网关 + CLI / Web Console。对标 Cursor Agent、Claude Code、Codex CLI。提交后仍可 `pip install -e` 并启动现有 CLI。

## 改哪里

改完必须让相关测试通过；没有测试的核心路径补最小回归测试。不要为炫技无关 rename / 纯格式化。

## Changelog

- 每轮必写 `changelog/YYYY-MM-DD-英文slug.md`，从 `_template.md` 复制。
- 写清：选了哪一类、改了什么、怎么验收、下一步不要做。
- 禁止写密钥、会话原文、用户隐私。

## 提交

定时任务在 `cursor/auto_upgrade` 上直接 commit。人手任务等用户要求再提交。  
不要 `--no-verify`，不要 force push `main`。
