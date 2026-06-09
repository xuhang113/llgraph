"""llgraph 交互与会话内命令帮助文案。"""

from __future__ import annotations

from pathlib import Path

from llgraph.context.context_session import ContextSession
from llgraph.display.trace_display import TRACE_MODE_LABELS, TraceSession
from llgraph.loaders.rules_loader import discover_rules
from llgraph.loaders.skills_loader import discover_skills
from llgraph.terminal.keys import HELP_SHORTCUT_LINES

def print_interactive_help(
    *,
    allow_write: bool,
    web_search_enabled: bool = False,
    trace_session: TraceSession,
    context_session: ContextSession | None = None,
    workspace: Path | None = None,
    mcp_summary: str = "",
) -> None:
    """
    打印交互模式下的 /help 说明。

    @param allow_write 当前会话是否允许写文件
    @param trace_session 过程展示配置
    """
    write_state = "已开启" if allow_write else "未开启（只读）"
    web_state = "已启用" if web_search_enabled else "未启用（/web on）"
    trace_state = TRACE_MODE_LABELS[trace_session.mode]
    ctx = context_session or ContextSession()
    skills_on = ", ".join(ctx.active_skills) if ctx.active_skills else "（无，可用 /skill <name>）"
    rule_count = len(discover_rules(workspace)) if workspace else 0
    skill_count = len(discover_skills(workspace)) if workspace else 0

    mcp_line = mcp_summary if mcp_summary else "MCP: 未加载"
    shortcut_lines = "\n".join(HELP_SHORTCUT_LINES)

    text = f"""
llgraph 交互帮助
================

【会话内命令】（不会发给模型）

  /help, help, ?     显示本帮助
  /paste, /p         多行粘贴模式（--- 或连续两次回车结束；Ctrl+C 取消）
  /trace             查看过程展示模式说明
  /context           查看会话上下文分项占用（类似 Cursor Context 面板）
  /tools             列举当前会话内置工具与 MCP 工具（含简要说明）
  /trace stats       查看 token 估算与工具结果落盘统计（P6）
  /trace token       开关每步 token/cache 摘要（on|off|stats）
  /trace step        列出本轮折叠步骤
  /trace step <#>    展开指定步骤（规划/工具输出详情）
  /trace step last   展开最近一步
  /trace context     同 /context
  /trace all         完整过程（规划+工具参数+输出，同截图）
  /trace steps       折叠步骤摘要（默认，左侧步骤列表可点击展开）
  /trace reply       仅流式最终回复
  /trace none        都不展示
  /log               向量检索日志 + 执行日志路径与保留策略
  /log debug|info|warning  动态调整级别（Agent 调 search_code_* 时生效）
  /log file on|off   开关落盘 .llgraph/index/logs/search.log
  /log tail          最近执行日志（token/压缩/工具/索引缓存摘要）
  /log purge         按 retention_days 清理过期日志文件
  /config            查看 agent.json 用户/工作区配置路径与合并规则
  /survey            followup 开关；/survey off|on|status；--no-survey 禁用
  /sessions          列出会话（标题 + thread_id，类似 Cursor 历史）
  /session           同 /sessions；use/new/title 子命令见 /session
  /session current   显示当前 thread_id 与恢复命令（/session id、/sessionid 同）
  /session title <标题>  重命名当前会话；/session title <id> <标题> 改指定会话
  /session delete <id>   删除会话；delete all 删其它；加 --including-current 全删
  /model             列出 AI 网关支持模型（用户 ~/.llgraph + 工作区覆盖）
  /model <名>        切换模型（保留会话历史）
  /model reset       恢复 env / agent.json 默认模型
  /model refresh     重新校验网关 /v1/models 与目录对齐
  /write             查看当前只读/可写模式（-w 时写文件前弹出 Yes/No 菜单，同 Claude）
  /write on          切换为可写（等价 -w，保留会话历史）
  /write off         切换为只读
  /watch             查看索引监听状态
  /watch on          启动文件监听 + debounce 增量索引
  /watch off         停止监听（不退出会话）
  /watch status      同 /watch
  /web               查看 Web 搜索（Tavily）状态
  /web on            启用 web_search 内置工具
  /web off           禁用 web_search
  /web status        同 /web
  /rule              列出规则（项目 .llgraph/rules + 个人 ~/.llgraph/rules）
  /rule on|off <id>  强制启用/禁用某条 glob 规则
  /skill             列出技能（项目 + 个人 ~/.llgraph/skills/，同名个人优先）
  /skill <name>      启用技能（下一条消息起生效）
  /{{name}} <任务>     启用技能并执行（如 /tracking 梳理埋点链路）
  输入 /             斜杠补全（Skills / Commands / 内置）
  /skill clear       清空已启用技能
  Thought 规范       .llgraph/thought/*.md + .llgraph/agent.json（检索无结果扩词等）
  /index             查看索引状态；/index full|incremental|rebuild 在会话内构建
  /compress          压缩上下文（token 切分+tool 掩码+结构化锚点+可选代码检索）
  /review [主题]     评审本会话变更文件，落盘 ~/llgraph-review/
  /changes           本会话 Agent 改过的文件（需 -w）
  /changes diff <path>  对比首次编辑前快照与当前
  /undo              查看可还原文件列表
  /undo all          还原本会话全部改动（快照写回，新建文件删除）
  /undo <path>       还原单个文件
  /changes clear     清空内存列表（落盘保留）
  /changes reset     清空内存并删除落盘记录
  /diff <path>       同 /changes diff
{shortcut_lines}

【当前会话】

  文件写入: {write_state}
  Web 搜索: {web_state}
  过程展示: {trace_state}
  已启用技能: {skills_on}
  工作区规则: {rule_count} 条 | 技能定义: {skill_count} 个
  {mcp_line}

【启动参数】

  llgraph              交互（/trace steps）
  llgraph --trace all  启动即为完整过程展示
  llgraph -C <目录>    工作区根目录（文件工具沙箱）
  llgraph -w           允许写入（search_replace / write_file）；会话内可用 /write off
  llgraph --no-watch-index  不随 Agent 启动自动增量索引（会话内可用 /watch on）
  llgraph --no-spill   禁用大工具结果落盘（调试）
  llgraph --no-survey  禁用交互式 survey（长期非交互 Agent）
  llgraph --log-level debug  向量检索 debug（默认仅 search.log；加 --log-console 才打终端）
  llgraph --model <名>       启动时指定网关模型
  llgraph --trace none 启动时即为「都不展示」
  llgraph --trace reply 启动时为「仅回复」
  llgraph --once "…"   单轮后退出
  llgraph index -C .   构建代码向量索引（需 pip install -e '.[index]'）
                       Embedding 默认本地；配置 .llgraph/embedding.json
                       重建: --rebuild  日志: .llgraph/index/logs/latest.log
  llgraph search "…" -C .  Hybrid 检索调试
                       加 --log-level info 可看到 [vector] 是否走向量路

【向量检索日志】

  级别: --log-level debug|info|warning  或  LLGRAPH_LOG_LEVEL  或  /log debug
  落盘: .llgraph/index/logs/search.log（INFO/DEBUG 默认开启，可用 /log file off 关闭）
  关键字: 日志行以 [vector] 开头；hybrid 另有 vector_hits= 汇总

【示例】

  cd /path/to/your/workspace
  llgraph -C .
  > /trace reply
  > 帮我梳理一下项目结构
  > /skill demo
  > /rule list
  > exit

【Rule / Skill 目录】（仅 llgraph，不读取 .cursor；同名个人优先）

  项目: .llgraph/rules/*.mdc 、.llgraph/skills/<name>/SKILL.md
  个人: ~/.llgraph/rules/ 、~/.llgraph/skills/（写作风格等，勿提交仓库）
  工作区: llgraph --init-config -C .
  用户级: llgraph --init-user-config  → ~/.llgraph/（agent.json、rules、skills）
  agent.json: 工作区覆盖用户（web_search 仅用户级）；/config 查看路径

【会话恢复】

  对话记忆: ~/.llgraph/context/<工作区名>/sessions/<thread_id>/messages.jsonl
  会话锚点: manifest.json + conversation_anchor.json + cli-*.jsonl 归档
  llgraph --list-sessions -C <工作区>     列出 thread_id
  llgraph -C <工作区> --thread-id cli-xxx  恢复该会话
  会话内: /session use <id>  |  /session new
  省略 --thread-id 则每次新建 cli-xxxxxxxx

【检索建议】

  search_workspace 的 keywords 须由模型一次给出多个相关词（无内置业务词典）。
  例如整理直播：keywords=live,livestream,broadcast,streaming,room,acme-live,直播
"""
    from llgraph.terminal.output import emit_report

    emit_report(text.strip())
