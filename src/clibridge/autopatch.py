"""Make the plain openai / anthropic / litellm / langchain_openai packages serve clibridge models.

Applied automatically at import time by the startup hook (see ``_clibridge_autoload``);
you can also call :func:`patch` yourself. A request is taken over only when its
``model`` is a clibridge name (:func:`clibridge.config.is_bridge_model`), so the same
client keeps talking to the real API for every other model.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
from typing import Any, Dict, Optional, Tuple

from .app import App
from .config import BridgeConfig, is_bridge_model
from .transport import sdk_httpx, transports_for

PLACEHOLDER_KEY = "clibridge-no-key"
_PATHS = ("/chat/completions", "/responses", "/messages", "/messages/count_tokens")

_app: Optional[App] = None
_clients: Dict[Tuple[str, bool], Any] = {}


def _bridge_client(flavour: str, async_: bool) -> Any:
    """One in-process client per httpx flavour, sharing one App (so previous_response_id works)."""
    global _app
    key = (flavour, async_)
    if key not in _clients:
        if _app is None:
            _app = App(BridgeConfig())
        sync_cls, async_cls = transports_for(flavour)
        httpx = importlib.import_module(flavour)
        transport = async_cls(app=_app) if async_ else sync_cls(app=_app)
        client_cls = httpx.AsyncClient if async_ else httpx.Client
        _clients[key] = client_cls(transport=transport, timeout=None)
    return _clients[key]


def _requested_model(request: Any) -> Optional[str]:
    if request.method != "POST" or not request.url.path.endswith(_PATHS):
        return None
    try:
        body = json.loads(request.content or b"{}")
    except (ValueError, UnicodeDecodeError, RuntimeError):  # RuntimeError: streaming body
        return None
    return body.get("model") if isinstance(body, dict) else None


def route_http_client(client: Any, flavour: str) -> None:
    """Send requests for clibridge models made through this httpx client to the local CLIs."""
    if client is None or getattr(client, "_clibridge_routed", False):
        return
    original = client.send
    if inspect.iscoroutinefunction(original):

        async def send(request, *args, **kwargs):
            if is_bridge_model(_requested_model(request)):
                return await _bridge_client(flavour, True).send(request, *args, **kwargs)
            return await original(request, *args, **kwargs)
    else:

        def send(request, *args, **kwargs):
            if is_bridge_model(_requested_model(request)):
                return _bridge_client(flavour, False).send(request, *args, **kwargs)
            return original(request, *args, **kwargs)

    client.send = send
    client._clibridge_routed = True


def _wrap_init(cls: type, after) -> None:
    if getattr(cls.__init__, "_clibridge", False):
        return
    original = cls.__init__

    @functools.wraps(original)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        after(self, original, args, kwargs)

    __init__._clibridge = True  # type: ignore[attr-defined]
    cls.__init__ = __init__


def patch_openai() -> None:
    import openai

    flavour = sdk_httpx(openai)

    def after(self, original, args, kwargs):
        try:
            original(self, *args, **kwargs)
        except openai.OpenAIError as exc:
            # No credentials at all: allow it, since clibridge models don't need any.
            message = str(exc).lower()
            if kwargs.get("api_key") is not None or not ("credential" in message or "api_key" in message):
                raise
            original(self, *args, **{**kwargs, "api_key": PLACEHOLDER_KEY})
        route_http_client(getattr(self, "_client", None), flavour)

    for cls in (openai.OpenAI, openai.AsyncOpenAI):
        _wrap_init(cls, after)


def _anthropic_auto_credentials() -> bool:
    try:
        from anthropic.lib.credentials._constants import _has_auto_discoverable_credentials
    except Exception:
        return False
    try:
        return bool(_has_auto_discoverable_credentials())
    except Exception:
        return True  # unsure, so don't touch auth


def patch_anthropic() -> None:
    import anthropic

    flavour = sdk_httpx(anthropic)

    def after(self, original, args, kwargs):
        original(self, *args, **kwargs)
        no_auth = all(getattr(self, a, None) is None for a in ("api_key", "auth_token", "credentials"))
        if no_auth and not _anthropic_auto_credentials():
            self.api_key = PLACEHOLDER_KEY  # clibridge models don't need credentials
        route_http_client(getattr(self, "_client", None), flavour)

    for cls in (anthropic.Anthropic, anthropic.AsyncAnthropic):
        _wrap_init(cls, after)


def patch_litellm() -> None:
    import litellm

    from .litellm import PROVIDERS, register

    register()
    # Let a bare "claude-code" / "codex" work without a "provider/model" suffix.
    for name in PROVIDERS:
        litellm.model_alias_map.setdefault(name, f"{name}/default")


def patch_langchain_openai() -> None:
    from langchain_openai import ChatOpenAI

    def after(self, original, args, kwargs):
        # ChatOpenAI refuses to build a client without an API key; clibridge models don't need one.
        model = kwargs.get("model", kwargs.get("model_name"))
        if is_bridge_model(model) and kwargs.get("api_key") is None and not os.environ.get("OPENAI_API_KEY"):
            kwargs = {**kwargs, "api_key": PLACEHOLDER_KEY}
        original(self, *args, **kwargs)

    _wrap_init(ChatOpenAI, after)


_PATCHES = {
    "openai": patch_openai,
    "anthropic": patch_anthropic,
    "litellm": patch_litellm,
    "langchain_openai": patch_langchain_openai,
}


def patch(name: Optional[str] = None) -> None:
    """Patch one SDK by module name, or every installed one when ``name`` is None."""
    for target, apply in _PATCHES.items():
        if name is None or name == target:
            try:
                importlib.import_module(target)
            except ImportError:
                continue
            apply()
