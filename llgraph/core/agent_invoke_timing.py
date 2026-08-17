"""Agent 单次 LLM 调用分段计时（prepare / HTTP）。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentInvokeTiming:
    """单次 agent 节点 LLM 调用的分段耗时。"""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prepare_sec: float = 0.0
    http_sec: float = 0.0
    normalize_sec: float = 0.0

    @property
    def total_sec(self) -> float:
        return self.prepare_sec + self.http_sec + self.normalize_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prepare_sec": round(self.prepare_sec, 4),
            "http_sec": round(self.http_sec, 4),
            "normalize_sec": round(self.normalize_sec, 4),
            "total_sec": round(self.total_sec, 4),
        }


def attach_invoke_timing(msg: Any, timing: AgentInvokeTiming) -> Any:
    """
    将分段计时写入 AIMessage.additional_kwargs.llgraph。

    @param msg LangChain 消息
    @param timing 分段计时
    @return 更新后的消息（新实例）
    """
    from langchain_core.messages import AIMessage

    if not isinstance(msg, AIMessage):
        return msg
    extra = dict(getattr(msg, "additional_kwargs", None) or {})
    meta = dict(extra.get("llgraph") or {})
    meta["invoke_timing"] = timing.to_dict()
    extra["llgraph"] = meta
    return msg.model_copy(update={"additional_kwargs": extra})


def read_invoke_timing(msg: Any) -> AgentInvokeTiming | None:
    """
    从 AIMessage 读取 invoke_timing。

    @param msg LangChain 消息
    @return 分段计时或 None
    """
    extra = getattr(msg, "additional_kwargs", None) or {}
    if not isinstance(extra, dict):
        return None
    meta = extra.get("llgraph")
    if not isinstance(meta, dict):
        return None
    raw = meta.get("invoke_timing")
    if not isinstance(raw, dict):
        return None
    request_id = str(raw.get("request_id") or "").strip() or uuid.uuid4().hex[:12]
    return AgentInvokeTiming(
        request_id=request_id,
        prepare_sec=float(raw.get("prepare_sec") or 0.0),
        http_sec=float(raw.get("http_sec") or 0.0),
        normalize_sec=float(raw.get("normalize_sec") or 0.0),
    )
