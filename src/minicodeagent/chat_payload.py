"""Build Chat Completions SDK arguments without importing the SDK."""

from .protocol import ChatMessage, ChatRequest, parse_tool_arguments, wire_call_id


def encode_chat_payload(
    request: ChatRequest, *, model: str | None = None, deepseek_thinking: str | None = None,
) -> dict:
    params = {"model": model} if model is not None else {}
    params["messages"] = [_message(message) for message in request.messages]
    if request.tools:
        params["tools"] = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        } for tool in request.tools]
    if deepseek_thinking == "disabled":
        params["extra_body"] = {"thinking": {"type": "disabled"}}
    return params


def _message(message: ChatMessage) -> dict:
    wire = {"role": message.role, "content": message.content}
    if message.tool_calls:
        calls = []
        for item in message.tool_calls:
            try:
                parse_tool_arguments(item.arguments_raw)
                arguments = item.arguments_raw
            except (ValueError, RecursionError, OverflowError):
                arguments = "{}"
            calls.append({
                "id": item.wire_id,
                "type": "function",
                "function": {"name": item.name, "arguments": arguments},
            })
        wire["tool_calls"] = calls
    if message.call_id is not None:
        wire["tool_call_id"] = wire_call_id(message.call_id)
    return wire
