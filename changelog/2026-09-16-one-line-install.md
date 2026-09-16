# 2026-09-16 一键安装；模型和编辑器交给 Cloud Agent 排队

选题：商用体验（分发）。

## 做了什么

- 加 `scripts/install.sh`：`curl | bash` 或在已有克隆里跑；venv + extras + `~/.local/bin/llgraph`
- README / 操作手册补上这一条命令
- Cloud Agent 排队改成：先模型入口（Anthropic / OpenAI / Gemini / Ollama），再 VS Code 或 ACP；1 没做完不要开 2

## 改了哪些路径

- `scripts/install.sh`
- `README.md`
- `docs/操作手册.md`
- `AGENTS.md`
- `.cursor/rules/cloud-agent.mdc`
- `.github/workflows/main.yml`

## 怎么验收

- `bash -n scripts/install.sh` 通过
- 管道安装说明出现在 README「安装」
- 定时 Agent 开工读到排队第 1 条是模型入口，而不是再去做速度/性能

## 未做 / 下一步不要做

- 没有发 PyPI / brew
- 没有改 `llm.py` 接多家厂商（交给定时轮第 1 件）
- 没有建 VS Code / ACP 扩展（第 1 件能跑一轮之前不要开）
- 安装脚本不要读取或写入真实 API Key
