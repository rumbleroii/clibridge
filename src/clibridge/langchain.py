"""LangChain chat models backed by the local CLIs.

``ChatClaudeCode`` and ``ChatCodex`` are ``ChatOpenAI`` subclasses wired to the
in-process transport, so ``invoke``, ``stream``, ``bind_tools``,
``with_structured_output``, async and LangGraph agents all work unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import openai
from langchain_openai import ChatOpenAI

from ._transport_clients import BASE_URL
from .config import BridgeConfig
from .transport import http_client, sdk_httpx

_BACKENDS = ("claude", "codex")


class ChatCLI(ChatOpenAI):
    """``ChatOpenAI`` whose requests run on ``claude -p`` / ``codex exec``.

    ``model`` takes clibridge model names: "claude-code", "claude-code/opus",
    "codex", "claude-code-agent", ...
    """

    def __init__(self, model: str = "claude-code", *, config: Optional[BridgeConfig] = None, **kwargs: Any):
        kwargs.setdefault("api_key", "clibridge")
        kwargs.setdefault("base_url", BASE_URL + "/v1")
        kwargs.setdefault("max_retries", 0)
        # ChatOpenAI switches model names containing "codex" to the Responses API; keep Chat Completions.
        kwargs.setdefault("use_responses_api", False)
        if "http_socket_options" in ChatOpenAI.model_fields:
            kwargs.setdefault("http_socket_options", ())  # no sockets involved; skips a proxy notice
        flavour = sdk_httpx(openai)
        kwargs.setdefault("http_client", http_client(flavour, async_=False, config=config))
        kwargs.setdefault("http_async_client", http_client(flavour, async_=True, config=config))
        super().__init__(model=kwargs.pop("model_name", model), **kwargs)

    @property
    def _llm_type(self) -> str:
        return "clibridge"


def _qualify(backend: str, model: Optional[str], agent: bool) -> str:
    if model and model.lower().startswith(_BACKENDS):
        return model  # already a clibridge name
    prefix = backend + ("-agent" if agent else "")
    return f"{prefix}/{model}" if model else prefix


class ChatClaudeCode(ChatCLI):
    """Claude via Claude Code. ``ChatClaudeCode(model="opus")``; ``agent=True`` for agent mode."""

    def __init__(self, model: Optional[str] = None, *, agent: bool = False, **kwargs: Any):
        super().__init__(_qualify("claude-code", kwargs.pop("model_name", model), agent), **kwargs)


class ChatCodex(ChatCLI):
    """OpenAI models via the Codex CLI. ``ChatCodex(model="gpt-5-codex")``; ``agent=True`` for agent mode."""

    def __init__(self, model: Optional[str] = None, *, agent: bool = False, **kwargs: Any):
        super().__init__(_qualify("codex", kwargs.pop("model_name", model), agent), **kwargs)
