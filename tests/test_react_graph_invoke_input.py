"""ReAct sync agent 节点：LLM 入参格式。"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from llgraph.core.react_invoke import invoke_agent_runnable_cancellable


def test_chat_model_rejects_messages_dict_wrapper() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    with pytest.raises(ValueError, match="Invalid input type"):
        model.invoke({"messages": [HumanMessage(content="hi")]})


def test_invoke_agent_runnable_accepts_message_list() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    out = invoke_agent_runnable_cancellable(
        model,
        [HumanMessage(content="hi")],
        RunnableConfig(),
    )
    assert out.content == "ok"
