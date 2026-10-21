"""OpenAI Chat Completions wire format."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from .core import ChatItem, ChatResult, Thinking, new_id

MODELS = [
    "claude-code",
    "claude-code/fable",
    "claude-code/opus",
    "claude-code/sonnet",
    "claude-code/haiku",
    "claude-code-agent",
    "codex",
    "codex-agent",
]


def models_list() -> Dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": now, "owned_by": "clibridge"} for m in MODELS],
    }


def usage(result: ChatResult) -> Dict[str, Any]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": result.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": result.reasoning_tokens},
    }


def _tool_calls(result: ChatResult) -> List[Dict[str, Any]]:
    return [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in result.tool_calls
    ]


def completion(result: ChatResult, request: Dict[str, Any]) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": result.content, "refusal": None}
    if result.reasoning:
        # Same field DeepSeek, vLLM and LiteLLM use for thinking text.
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = _tool_calls(result)
    return {
        "id": new_id("chatcmpl-"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model") or "claude-code",
        "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason, "logprobs": None}],
        "usage": usage(result),
        "system_fingerprint": None,
        "clibridge": result.meta,
    }


class ChunkEncoder:
    """Turns iter_chat() items into ``chat.completion.chunk`` objects."""

    def __init__(self, request: Dict[str, Any]):
        self.request = request
        self.base = {
            "id": new_id("chatcmpl-"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.get("model") or "claude-code",
            "system_fingerprint": None,
        }

    def _chunk(self, delta: Dict[str, Any], finish: Any = None) -> Dict[str, Any]:
        return {**self.base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}

    def start(self) -> List[Dict[str, Any]]:
        return [self._chunk({"role": "assistant", "content": ""})]

    def encode(self, item: ChatItem) -> List[Dict[str, Any]]:
        if isinstance(item, str):
            return [self._chunk({"content": item})]
        if isinstance(item, Thinking):
            return [self._chunk({"reasoning_content": item.text})]
        out = []
        if item.reasoning and not item.reasoning_streamed:
            out.append(self._chunk({"reasoning_content": item.reasoning}))
        if item.content and not item.streamed:
            out.append(self._chunk({"content": item.content}))
        if item.tool_calls:
            calls = [{"index": i, **call} for i, call in enumerate(_tool_calls(item))]
            out.append(self._chunk({"tool_calls": calls}))
        out.append(self._chunk({}, item.finish_reason))
        if (self.request.get("stream_options") or {}).get("include_usage"):
            out.append({**self.base, "choices": [], "usage": usage(item)})
        return out
