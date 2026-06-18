# llgraph

基于 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 的终端 Agent，通过 **OpenAI 兼容 API 网关** 调用大模型。

在 monorepo 工作区中提供：**语义搜代码**、Rules/Skills、局部改代码、MCP、上下文压缩与动态落盘、会话记忆；可选 **Web Console**（浏览器 UI）与 **Python 库 API**（`llgraph.console`）。

| 文档 | 说明 |
|------|------|
| [docs/操作手册.md](docs/操作手册.md) | 安装、命令、配置、排障 |
| [docs/项目结构.md](docs/项目结构.md) | **开发文档**：目录与子包、调用链、改哪 |
| [docs/模块说明.md](docs/模块说明.md) | 源码模块职责与扩展 |
| [docs/code-index.md](docs/code-index.md) | 代码向量索引 |
| [docs/会话上下文与历史.md](docs/会话上下文与历史.md) | 会话格式、加载、压缩与 Token 节约 |

**集成方式**：优先使用 **`llgraph.console` Python API** 或终端 **`llgraph` CLI**；Web UI 为可选本地界面（`llgraph web`），不对外暴露独立 REST 产品。

---

## 环境

- Python **3.12+**
- **[ripgrep](https://github.com/BurntSushi/ripgrep)**（`rg` 命令，需在 PATH 中）— `glob_files` / `grep_files` 及 hybrid 检索的 grep 路依赖它；未安装时会降级为慢速 Python 遍历，且 `glob_files` 不可用

```bash
# macOS (Homebrew)
brew install ripgrep

# Debian / Ubuntu
sudo apt install ripgrep

# Fedora / RHEL
sudo dnf install ripgrep

# 验证
which rg && rg --version
```

## 配置 API 凭据

1. **推荐**：`~/.config/llgraph/llgraph.env`（见 `examples/llgraph.env.example`）
2. 可选：llgraph 项目根 `.env`（覆盖用）

```bash
mkdir -p ~/.config/llgraph
cp examples/llgraph.env.example ~/.config/llgraph/llgraph.env
# 编辑 LLGRAPH_API_BASE_URL、LLGRAPH_API_KEY、LLGRAPH_MODEL
```

| 变量 | 含义 |
|------|------|
| `LLGRAPH_API_BASE_URL` | OpenAI 兼容网关根地址 |
| `LLGRAPH_API_KEY` | API 令牌 |
| `LLGRAPH_MODEL` | 默认模型名 |

凭据使用 **`LLGRAPH_*`**，不复用 Claude CLI 的 `ANTHROPIC_*`。

## 安装

```bash
cd /path/to/llgraph
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip -e '.[index,watch,mcp]'
```

可选依赖：

| extra | 用途 |
|-------|------|
| `[ast]` | tree-sitter AST 切块 |
| `[search]` | Tavily `web_search` |
| `[web]` | 浏览器 Console（`llgraph web`） |
| `[terminal]` | Rich 终端增强 |

Web Console：

```bash
pip install -e '.[web]'
```

## 快速开始

```bash
llgraph --init-user-config                    # ~/.llgraph/agent.json
llgraph --init-config -C /path/to/workspace   # 工作区 .llgraph/
llgraph index -C /path/to/workspace           # 代码索引（推荐）

llgraph -C /path/to/workspace               # 交互（只读）
llgraph -w -C /path/to/workspace            # 允许改代码
```

---

## Web Console（可选）

浏览器版工作区 / Agent / Plan 控制台。前端在 `web-ui/`，后端与 Agent 执行同属 **llgraph 进程**（经 `llgraph.console` 调库，非独立 REST 服务）。

### 启动

**开发模式**（Vite 热更新，推荐日常改 UI）：

```bash
./scripts/web-dev.sh dev
# UI: http://127.0.0.1:5173   API: http://127.0.0.1:8765
# 关闭：终端 Ctrl+C
```

**单端口**（托管已构建的静态页）：

```bash
cd web-ui && npm install && npm run build   # 首次或前端变更后
llgraph web                                 # http://127.0.0.1:8765
# 关闭：Ctrl+C；或 ./scripts/web-dev.sh stop（若用 start 后台启动）
```

常用参数：`llgraph web --host 127.0.0.1 --port 8765`

### Python 库 API（集成推荐）

```python
from llgraph.console import Console

c = Console()
c.list_workspaces()
c.register_workspace("/path/to/workspace")
c.session_tree("WorkspaceV2-xxxxxxxx")
c.delete_session("WorkspaceV2-xxxxxxxx", "plan-xxxxxxxx")  # Plan 含 Worker 级联
```

控制面边界见 `llgraph.gateway`（`ControlGateway`）；远程部署预留 `LLGRAPH_CONTROL_GATEWAY=remote`。

---

## 数据放在哪（重要）

llgraph 把 **仓库内配置** 与 **用户目录下的会话/记忆** 分开，避免污染 git 工作区。

### 工作区内（随仓库提交 `.llgraph/`，索引默认忽略）

| 路径 | 用途 |
|------|------|
| `.llgraph/agent.json` | 项目配置（thought、context、review…） |
| `.llgraph/rules/`、`skills/`、`commands/` | 规则、技能、自定义命令 |
| `.llgraph/mcp.json` | MCP 服务配置 |
| `.llgraph/index/` | LanceDB 代码向量索引 |
| `.llgraph/context/tool-results/` | 超大工具结果落盘指针（P6） |

### 用户目录（不占仓库）

| 路径 | 用途 |
|------|------|
| `~/.config/llgraph/llgraph.env` | API 凭据 |
| `~/.llgraph/agent.json` | 用户级配置（模型列表、日志等） |
| `~/.llgraph/rules/`、`~/.llgraph/skills/` | 个人规则/技能（同名时优先于项目） |
| `~/.llgraph/context/<工作区名>/sessions/<thread_id>/` | **单会话全部落盘**（见下表） |

### 单会话目录内容（`~/.llgraph/context/<slug>/sessions/<thread_id>/`）

| 文件 | 说明 |
|------|------|
| `messages.jsonl` | 对话正文（可人工打开编辑） |
| `meta.json` | 标题、更新时间等 |
| `manifest.json` | Skill/Rule 锚点（压缩后仍保留指针） |
| `conversation_anchor.json` | 结构化会话摘要（Tier2 压缩） |
| `edits.jsonl` | 本会话文件改动账本（需 `-w`） |
| `snapshots/` | 首次编辑前快照（`/undo`） |

同级目录下还可能有 `<thread_id>.jsonl`：`/compress` 时的对话归档。

> **旧版路径**：`<工作区>/.llgraph/sessions/<id>/` 已在启动时**自动迁移**到用户目录；`agent.json` 中的 `edits.sessions_dir: ".llgraph/sessions"` 不再写入工作区（仅**绝对路径**可自定义落盘位置）。

---

## 会话管理

```bash
llgraph --list-sessions -C <工作区>              # 列出会话（标题 + thread_id）
llgraph -C <工作区> --thread-id cli-xxxxxxxx     # 恢复指定会话
llgraph -C <工作区> --delete-session cli-xxx     # 删除 Agent 会话
llgraph -C <工作区> --delete-session plan-xxx    # 删除 Plan（含 Worker/Planner 子节点）
llgraph -C <工作区> --purge-sessions --including-current   # 全量删除
```

会话内：

| 命令 | 说明 |
|------|------|
| `/sessions`、`/session` | 列表（仅展示**有实质内容**的会话；空壳另提示） |
| `/session use <id>` | 切换会话 |
| `/session new` | 新建会话 |
| `/session title <标题>` | 重命名（≤30 字；首条用户消息也会自动生成标题） |
| `/session delete <id>` | 删除指定会话（`cli-*` 或 `plan-*`，Plan 级联 Worker） |
| `/session delete empty` | 删除空壳会话（仅 manifest/meta、无对话） |
| `/session delete all` | 删除除当前外全部 |
| `/session delete all --including-current` | 全量删除并切到新会话 |

记忆实现：**MemorySaver + `messages.jsonl`**，无 SQLite。

---

## 配置分层

| 层级 | 路径 | 典型内容 |
|------|------|----------|
| API | `~/.config/llgraph/llgraph.env` | `LLGRAPH_*` |
| 用户 | `~/.llgraph/agent.json` | `llm.models`、`logging.level` |
| 工作区 | `<workspace>/.llgraph/agent.json` | `llm.model`、`thought`、`context`、`edits` |
| 会话 | `/model`、`/log` | 仅当前进程，不写盘 |

合并：先用户配置，再工作区**深度合并**覆盖。`/config` 查看实际路径。

`context` 常用项：

- **`compress_strategy`**：`auto`（默认，滚动 anchor + auto 出站）或 `legacy`（`keep_recent_*` + 固定出站轮数）
- **`dispatch_keep_user_turns`**：`0` = 按 `dispatch_window_token_ratio` 自动扩展 user 轮；`>0` = 固定 N 轮
- **`auto_compress_ratio`**、**`compress_trigger_max_tokens`**：自动压缩阈值
- **`compress_retrieval_enabled`**：Tier3 压缩前代码检索

模板与字段注释见 `examples/default-workspace/.llgraph/agent.json` → `context._docs`；会话内 `/context` 可查看生效值。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **文件工具** | `glob_files`、`grep_files`（ripgrep）、`read_file`、`search_workspace`；沙箱在工作区内 |
| **代码索引** | `llgraph index` + `search_code_hybrid`（RRF） |
| **Rules / Skills** | `.llgraph` + `~/.llgraph` 双源，个人优先；`/rule`、`/skill` |
| **局部改代码** | `-w`：`search_replace`、`write_file`；`/changes`、`/undo` |
| **上下文压缩** | `/compress` + 自动压缩（`compress_strategy: auto`）；auto 出站按 token 扩展 user 轮；Tier2 锚点 / Tier3 检索 |
| **动态落盘 (P6)** | 大工具结果 → `.llgraph/context/tool-results/` |
| **MCP** | `.llgraph/mcp.json`，工具名 `mcp__<server>__<tool>` |
| **监听索引** | 保存后 debounce 增量索引（可 `--no-watch-index`） |
| **Shell** | `run_shell_command`（只读模式禁写盘类命令） |
| **Web 搜索** | `/web on` + `TAVILY_API_KEY` |
| **模型** | `/model`、`--model` |
| **评审** | `/review` → `~/llgraph-review/` |
| **自定义命令** | `.llgraph/commands/*.md` |

## 常用 CLI

```bash
llgraph web --help                          # Web Console（需 [web]）
llgraph index --status -C .
llgraph index -C . --incremental
llgraph search "NotFoundException" -C .
llgraph plan -C . "实现某某功能"             # Plan 模式
llgraph --once -C . "现在 UTC 几点"
llgraph --no-watch-index -C .
llgraph --no-spill -C .
```

## 会话内常用命令

`/help` · `/config` · `/context` · `/trace` · `/compress` · `/tools` · `/paste` · `/changes` · `/review`

---

## Code Index

索引目录：`<workspace>/.llgraph/index/`。详见 [docs/code-index.md](docs/code-index.md)。

```bash
llgraph index -C . --rebuild
llgraph index -C . --path some-service
```

默认 Embedding：`BAAI/bge-small-zh-v1.5`（`embedding.json` 可改）。

---

## 项目结构

```
llgraph/
  llgraph/
    main.py                 CLI 入口（含 llgraph web / index / search / plan）
    cli/                    子命令（web_cli 等）
    console/                Web Console 库 API + runtime（Agent/Plan SSE）
    gateway/                控制面网关（local / remote 预留）
    web/server/             bundled UI 本地 HTTP 适配（非公开 API）
    core/                   ReAct Agent、工具
    session/                会话持久化、删除、注册
    plan/                   Plan 多 Agent 工作流
    context/                压缩、落盘
    code_index/             向量索引
  web-ui/                   React 前端（Vite；build → dist/）
  scripts/web-dev.sh        Web 开发/启停脚本
docs/
examples/
  user-llgraph/             用户级 init 模板
  default-workspace/.llgraph/
```

会话落盘核心模块：`session_file_store.py`、`session_registry.py`、`session_meta.py`、`session_delete.py` 等，见 [docs/项目结构.md](docs/项目结构.md)。

---

## 开发说明

本项目使用 [Cursor](https://cursor.com) 助手辅助开发。
