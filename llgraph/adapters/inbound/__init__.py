"""入站 response 归一化（与 MessageDispatchProfile 对称）。"""

from llgraph.adapters.inbound.kimi_native import (
    content_has_kimi_tool_tokens,
    normalize_plain_functions_tool_calls,
    parse_kimi_native_tool_calls,
    strip_kimi_tool_call_markup,
    strip_plain_functions_tool_calls,
)
from llgraph.adapters.inbound.normalize import classify_tool_call_response, normalize_ai_response
from llgraph.adapters.inbound.profile import InboundAdapterProfile, resolve_inbound_profile
from llgraph.adapters.inbound.structured import (
    UnstructuredToolCallError,
    ai_message_has_structured_tool_calls,
    validate_structured_tool_response,
)
from llgraph.adapters.inbound.xml_tool_call import (
    content_has_xml_tool_calls,
    parse_xml_tool_calls,
    strip_inbound_tool_call_markup,
    strip_xml_tool_call_markup,
)

__all__ = [
    "InboundAdapterProfile",
    "UnstructuredToolCallError",
    "ai_message_has_structured_tool_calls",
    "classify_tool_call_response",
    "content_has_kimi_tool_tokens",
    "content_has_xml_tool_calls",
    "normalize_ai_response",
    "normalize_plain_functions_tool_calls",
    "parse_kimi_native_tool_calls",
    "parse_xml_tool_calls",
    "resolve_inbound_profile",
    "strip_inbound_tool_call_markup",
    "strip_kimi_tool_call_markup",
    "strip_plain_functions_tool_calls",
    "strip_xml_tool_call_markup",
    "validate_structured_tool_response",
]
