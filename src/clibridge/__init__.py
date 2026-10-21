"""clibridge: use your Claude Code and Codex subscriptions from any LLM SDK.

Requests in the OpenAI Chat Completions, OpenAI Responses or Anthropic Messages
format are answered by running ``claude -p`` or ``codex exec`` locally.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

from .app import App
from .config import BridgeConfig, Route, resolve_model
from .core import ChatResult, acomplete_chat, aiter_chat, complete_chat, iter_chat
from .errors import BridgeError

__version__ = "0.1.0"

# Loaded on first use so the SDKs stay optional.
_LAZY = {
    "OpenAI": "_openai",
    "AsyncOpenAI": "_openai",
    "Anthropic": "_anthropic",
    "AsyncAnthropic": "_anthropic",
    "BridgeTransport": "transport",
    "AsyncBridgeTransport": "transport",
    "serve": "server",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'clibridge' has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def ask(prompt: str, model: str = "claude-code", system: Optional[str] = None,
        config: Optional[BridgeConfig] = None) -> str:
    """One-shot helper: ``clibridge.ask("Explain monads in one line")``."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return complete_chat({"model": model, "messages": messages}, config).content or ""


__all__ = [
    "App", "BridgeConfig", "BridgeError", "ChatResult", "Route", "acomplete_chat", "aiter_chat", "ask",
    "complete_chat", "iter_chat", "resolve_model", *_LAZY,
]
