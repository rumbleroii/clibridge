"""Anthropic Messages API wire format."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .core import ChatItem, ChatResult, Thinking, new_id

Event = Tuple[str, Dict[str, Any]]


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif block.get("type") == "image":
                parts.append("[image]")
    return "\n".join(parts)


def _image_url(source: Dict[str, Any]) -> Optional[str]:
    if source.get("type") == "base64":
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    if source.get("type") == "url":
        return source.get("url")
    return None


def to_chat_request(body: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _blocks_text(system)})

    for message in body.get("messages") or []:
        role, content = message.get("role"), message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if role == "assistant":
            texts, calls = [], []
            for block in content or []:
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "tool_use":
                    calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})},
                    })
            out: Dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
            continue
        parts: List[Dict[str, Any]] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "tool_result":
                text = _blocks_text(block.get("content"))
                if block.get("is_error"):
                    text = f"[error] {text}"
                messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": text})
            elif kind == "text":
                parts.append({"type": "text", "text": block.get("text") or ""})
            elif kind == "image":
                url = _image_url(block.get("source") or {})
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "document":
                source = block.get("source") or {}
                text = source.get("data") if source.get("type") == "text" else "[document]"
                parts.append({"type": "text", "text": text or ""})
        if parts:
            messages.append({"role": "user", "content": parts})

    request: Dict[str, Any] = {"model": body.get("model"), "messages": messages, "stream": bool(body.get("stream"))}
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for tool in body.get("tools") or []
        if isinstance(tool, dict) and tool.get("name") and (tool.get("input_schema") or tool.get("type") in (None, "custom"))
    ]
    if tools:
        request["tools"] = tools
    choice = body.get("tool_choice") or {}
    kind = choice.get("type")
    if kind == "any":
        request["tool_choice"] = "required"
    elif kind == "tool":
        request["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
    elif kind == "none":
        request["tool_choice"] = "none"
    if choice.get("disable_parallel_tool_use"):
        request["parallel_tool_calls"] = False
    effort = (body.get("output_config") or {}).get("effort") or thinking_effort(body.get("thinking"))
    if effort:
        request["reasoning_effort"] = effort
    for key in ("agent", "cwd", "max_turns"):
        if key in body:
            request[key] = body[key]
    return request


def thinking_effort(thinking: Any) -> Optional[str]:
    """Map the ``thinking`` parameter to a thinking level (the CLIs take levels, not token budgets)."""
    if not isinstance(thinking, dict):
        return None
    if thinking.get("type") == "disabled":
        return "none"
    budget = thinking.get("budget_tokens")
    if thinking.get("type") != "enabled" or not isinstance(budget, int):
        return None  # "adaptive": let the model decide
    for limit, effort in ((4096, "low"), (16384, "medium"), (32768, "high")):
        if budget < limit:
            return effort
    return "xhigh"


def wants_thinking(body: Dict[str, Any]) -> bool:
    """Thinking blocks are only returned when the request turned thinking on, like the real API."""
    thinking = body.get("thinking")
    return (isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
            and thinking.get("display") != "omitted")


def _input(arguments: str) -> Dict[str, Any]:
    try:
        value = json.loads(arguments)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _stop_reason(result: ChatResult) -> str:
    return {"tool_calls": "tool_use", "length": "max_tokens"}.get(result.finish_reason, "end_turn")


def _usage(result: ChatResult) -> Dict[str, Any]:
    return {
        "input_tokens": max(0, result.prompt_tokens - result.cached_tokens),
        "output_tokens": result.completion_tokens,
        "cache_read_input_tokens": result.cached_tokens,
        "cache_creation_input_tokens": 0,
    }


def message(result: ChatResult, body: Dict[str, Any]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if result.reasoning and wants_thinking(body):
        content.append({"type": "thinking", "thinking": result.reasoning, "signature": ""})
    if result.content:
        content.append({"type": "text", "text": result.content})
    for call in result.tool_calls:
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": _input(call.arguments)})
    return {
        "id": new_id("msg_"),
        "type": "message",
        "role": "assistant",
        "model": body.get("model") or "claude-code",
        "content": content,
        "stop_reason": _stop_reason(result),
        "stop_sequence": None,
        "usage": _usage(result),
        "clibridge": result.meta,
    }


def count_tokens(body: Dict[str, Any]) -> Dict[str, Any]:
    """Rough estimate (~4 characters per token); the CLIs don't expose a tokenizer."""
    size = len(json.dumps([body.get("system"), body.get("messages"), body.get("tools")], ensure_ascii=False))
    return {"input_tokens": max(1, size // 4)}


def error_body(message: str, status: int) -> Dict[str, Any]:
    kind = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
            429: "rate_limit_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


class MessageEncoder:
    """Turns iter_chat() items into Messages API streaming events."""

    def __init__(self, body: Dict[str, Any]):
        self.body = body
        self.thinking = wants_thinking(body)
        self.index = -1
        self.open: Optional[str] = None  # kind of the content block being streamed

    def start(self) -> List[Event]:
        return [("message_start", {
            "type": "message_start",
            "message": {
                "id": new_id("msg_"),
                "type": "message",
                "role": "assistant",
                "model": self.body.get("model") or "claude-code",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })]

    def _close(self) -> List[Event]:
        if self.open is None:
            return []
        self.open = None
        return [("content_block_stop", {"type": "content_block_stop", "index": self.index})]

    def _delta(self, kind: str, text: str) -> List[Event]:
        events = []
        if self.open != kind:
            events += self._close()
            self.index += 1
            self.open = kind
            block = {"type": "text", "text": ""} if kind == "text" else {"type": "thinking", "thinking": "", "signature": ""}
            events.append(("content_block_start", {"type": "content_block_start", "index": self.index,
                                                   "content_block": block}))
        delta = {"type": "text_delta", "text": text} if kind == "text" else {"type": "thinking_delta", "thinking": text}
        events.append(("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": delta}))
        return events

    def encode(self, item: ChatItem) -> List[Event]:
        if isinstance(item, str):
            return self._delta("text", item)
        if isinstance(item, Thinking):
            return self._delta("thinking", item.text) if self.thinking else []
        events = []
        if self.thinking and item.reasoning and not item.reasoning_streamed:
            events += self._delta("thinking", item.reasoning)
        if item.content and not item.streamed:
            events += self._delta("text", item.content)
        events += self._close()
        for call in item.tool_calls:
            self.index += 1
            events.append(("content_block_start", {
                "type": "content_block_start",
                "index": self.index,
                "content_block": {"type": "tool_use", "id": call.id, "name": call.name, "input": {}},
            }))
            events.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": self.index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(_input(call.arguments))},
            }))
            events.append(("content_block_stop", {"type": "content_block_stop", "index": self.index}))
        events.append(("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": _stop_reason(item), "stop_sequence": None},
            "usage": _usage(item),
        }))
        events.append(("message_stop", {"type": "message_stop"}))
        return events
