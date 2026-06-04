"""
LangGraph terminal agent with LLGRAPH_* API credentials.

子包布局（按业务域）::

    core      Agent 编排、Gateway LLM、工具
    context   上下文、消息规范、压缩
    session   会话持久化与生命周期
    config    环境变量与 settings
    loaders   Rules / Skills / Commands
    commands  斜杠命令
    survey    交互确认
    display   trace 与终端样式
    cli       index / search 子命令
    code_index / terminal / ui  — 已有子包
"""
