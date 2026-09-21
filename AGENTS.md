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

终端 Agent：LangGraph ReAct + CLI / Web Console。模型入口：OpenAI 兼容网关（`LLGRAPH_*`，默认）或 Anthropic / OpenAI / Gemini / Ollama 官方入口（见 `llgraph/config/providers.py`）。提交后仍可 `pip install -e` 并启动现有 CLI。陌生人安装走 `scripts/install.sh`。

## Cloud Agent 排队（做完一件再换）

1. ~~**模型入口**：开箱接 Anthropic / OpenAI / Gemini / Ollama，保留现有网关。~~ 2026-09-17 完成（`changelog/2026-09-17-multi-provider-model-entry.md`）。别再重做一遍；剩下的小口子（Gemini 协议层测试、`/model list` 接官方列表、一个会话混用多家）都写在那篇的「不要做」里。
2. **编辑器里干活（当前优先）**：ACP 服务端与 `llgraph acp` 2026-09-18 落地（`changelog/2026-09-18-acp-editor-entry.md`）；写入授权弹窗（`session/request_permission`）2026-09-19 落地（`changelog/2026-09-19-acp-write-permission.md`）；`session/load` 续聊 2026-09-20 落地（`changelog/2026-09-20-acp-session-load.md`）；`fs/read_text_file` / `fs/write_text_file` 反向请求 2026-09-21 落地（`changelog/2026-09-21-acp-fs-reverse-requests.md`）。编辑器里已能提问、看过程、停、**真改代码**、重启后接回旧会话、读写**未保存的缓冲区**。**不要重写这四层**，接着做 `session/update` 的 `tool_call` 中间态（`pending` → `in_progress` → `completed`，让编辑器里看到工具正在跑而不是跑完才出现）。VS Code 扩展与终端 TUI 仍后置。

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
