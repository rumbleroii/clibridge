"""Shared helpers for the drop-in SDK clients."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Dict, Optional

from .config import BridgeConfig
from .transport import http_client, sdk_httpx

BASE_URL = "http://clibridge.local"


def client_kwargs(kwargs: Dict[str, Any], sdk: ModuleType, base_url: str, config: Optional[BridgeConfig],
                  async_: bool) -> Dict[str, Any]:
    kwargs.setdefault("api_key", "clibridge")
    kwargs.setdefault("base_url", base_url)
    kwargs.setdefault("max_retries", 0)
    if kwargs.get("http_client") is None:
        kwargs["http_client"] = http_client(sdk_httpx(sdk), async_, config)
    return kwargs
