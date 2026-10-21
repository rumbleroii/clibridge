"""``openai.OpenAI`` / ``openai.AsyncOpenAI`` answered by the local CLIs."""

from __future__ import annotations

from typing import Any, Optional

import openai

from ._transport_clients import BASE_URL, client_kwargs
from .config import BridgeConfig


class OpenAI(openai.OpenAI):
    """``openai.OpenAI`` whose requests run on your Claude Code / Codex CLI. Use model="claude-code" or "codex"."""

    def __init__(self, *, config: Optional[BridgeConfig] = None, **kwargs: Any) -> None:
        super().__init__(**client_kwargs(kwargs, openai, BASE_URL + "/v1", config, async_=False))


class AsyncOpenAI(openai.AsyncOpenAI):
    """Async version of :class:`OpenAI`."""

    def __init__(self, *, config: Optional[BridgeConfig] = None, **kwargs: Any) -> None:
        super().__init__(**client_kwargs(kwargs, openai, BASE_URL + "/v1", config, async_=True))
