"""``anthropic.Anthropic`` / ``anthropic.AsyncAnthropic`` answered by the local CLIs."""

from __future__ import annotations

from typing import Any, Optional

import anthropic

from ._transport_clients import BASE_URL, client_kwargs
from .config import BridgeConfig

# A fixed timeout skips the SDK's "use streaming for long requests" check.
_TIMEOUT = 900.0


class Anthropic(anthropic.Anthropic):
    """``anthropic.Anthropic`` whose requests run on your Claude Code / Codex CLI."""

    def __init__(self, *, config: Optional[BridgeConfig] = None, **kwargs: Any) -> None:
        kwargs.setdefault("timeout", _TIMEOUT)
        super().__init__(**client_kwargs(kwargs, anthropic, BASE_URL, config, async_=False))


class AsyncAnthropic(anthropic.AsyncAnthropic):
    """Async version of :class:`Anthropic`."""

    def __init__(self, *, config: Optional[BridgeConfig] = None, **kwargs: Any) -> None:
        kwargs.setdefault("timeout", _TIMEOUT)
        super().__init__(**client_kwargs(kwargs, anthropic, BASE_URL, config, async_=True))
