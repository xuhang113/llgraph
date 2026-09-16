# llgraph · Agent 开发规范

给 Cursor Cloud Agent / 本机 Agent 用。

开工先读：本文件 → `changelog/` 最近 3 条 → 下手文件。对照 `docs/cursor-agent.md`、`docs/项目结构.md`、`docs/模块说明.md`。不要全库通读。

不要把 API Key、网关令牌、`.env` 写进代码或文档。

## 两种开工

**人手一轮：** 只做用户点名的事，不顺手加系统。

**定时自动迭代**（`cursor/auto_upgrade`，每天北京时间 10:00）：

- 直接在当前分支提交，不要开新分支，不要 PR。
- 按下面「排队」做，不要自己另开主题，不要两件一起开。
- 先看 changelog 已有方向再接着做，不要推倒重来。
- 没有可下手切口：只写 changelog 说明下轮攻哪，不要硬改。

## 产品

终端 Agent：LangGraph ReAct + CLI / Web Console。模型目前走 OpenAI 兼容网关（`LLGRAPH_*`）。提交后仍可 `pip install -e` 并启动现有 CLI。陌生人安装走 `scripts/install.sh`。

## Cloud Agent 排队（做完一件再换）

1. **模型入口（当前优先）**：开箱接 Anthropic / OpenAI / Gemini / Ollama，**保留**现有 OpenAI 兼容网关。做到用官方 Key 或本地 Ollama 能跑一轮对话+工具。下手：`llgraph/core/llm.py`、`llgraph/config/config.py`、`examples/llgraph.env.example`。
2. **编辑器里干活**：VS Code 扩展或 ACP 插件；终端 TUI 后置。**1 未完成前不要开 2。**

速度 / 性能 / 稳定性：仅当上面两件本轮都没有可下手切口时才选。不要做 PyPI / brew（安装脚本已经有了）。

## 改哪里

改完必须让相关测试通过；没有测试的核心路径补最小回归测试。不要为炫技无关 rename / 纯格式化。

## Changelog

- 每轮必写 `changelog/YYYY-MM-DD-英文slug.md`，从 `_template.md` 复制。
- 写清：选了哪一类、改了什么、怎么验收、下一步不要做。
- 禁止写密钥、会话原文、用户隐私。

## 提交

定时任务在 `cursor/auto_upgrade` 上直接 commit。人手任务等用户要求再提交。  
不要 `--no-verify`，不要 force push `main`。
