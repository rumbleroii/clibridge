"""httpx transports that answer requests in-process, so no server is needed.

Any SDK built on httpx (openai, anthropic, and everything built on those) can use them::

    httpx.Client(transport=BridgeTransport())

Newer SDKs may use the ``httpx2`` package instead; :func:`transports_for` builds
the same transports for either.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType
from typing import Any, Optional, Tuple

from .app import App
from .config import BridgeConfig


def sdk_httpx(sdk: ModuleType) -> str:
    """Name of the httpx package an SDK (``openai``, ``anthropic``) is built on."""
    try:
        base = importlib.import_module(sdk.__name__ + "._base_client")
    except ImportError:
        return "httpx"
    return "httpx2" if hasattr(base, "httpx2") else "httpx"


@functools.lru_cache(maxsize=None)
def transports_for(module_name: str = "httpx") -> Tuple[type, type]:
    """Return ``(BridgeTransport, AsyncBridgeTransport)`` classes for ``httpx`` or ``httpx2``."""
    httpx = importlib.import_module(module_name)

    class _SyncBody(httpx.SyncByteStream):
        def __init__(self, reply):
            self.reply = reply

        def __iter__(self):
            yield from self.reply.body

        def close(self) -> None:
            self.reply.body.close()
            if self.reply.close is not None:
                self.reply.close()

    class _AsyncBody(httpx.AsyncByteStream):
        def __init__(self, reply):
            self.reply = reply

        async def __aiter__(self):
            async for part in self.reply.body:
                yield part

        async def aclose(self) -> None:
            await self.reply.body.aclose()
            if self.reply.close is not None:
                await self.reply.close()

    class BridgeTransport(httpx.BaseTransport):
        def __init__(self, config: Optional[BridgeConfig] = None, app: Optional[App] = None):
            self.app = app or App(config)

        def handle_request(self, request):
            reply = self.app.dispatch(request.method, request.url.raw_path.decode("ascii"), request.read())
            if isinstance(reply.body, bytes):
                return httpx.Response(reply.status, headers=reply.headers, content=reply.body, request=request)
            return httpx.Response(reply.status, headers=reply.headers, stream=_SyncBody(reply), request=request)

    class AsyncBridgeTransport(httpx.AsyncBaseTransport):
        def __init__(self, config: Optional[BridgeConfig] = None, app: Optional[App] = None):
            self.app = app or App(config)

        async def handle_async_request(self, request):
            body = await request.aread()
            reply = await self.app.adispatch(request.method, request.url.raw_path.decode("ascii"), body)
            if isinstance(reply.body, bytes):
                return httpx.Response(reply.status, headers=reply.headers, content=reply.body, request=request)
            return httpx.Response(reply.status, headers=reply.headers, stream=_AsyncBody(reply), request=request)

    return BridgeTransport, AsyncBridgeTransport


def http_client(module_name: str, async_: bool, config: Optional[BridgeConfig] = None) -> Any:
    """An ``httpx(2).Client`` / ``AsyncClient`` whose requests are answered in-process."""
    httpx = importlib.import_module(module_name)
    sync_cls, async_cls = transports_for(module_name)
    if async_:
        return httpx.AsyncClient(transport=async_cls(config), timeout=None)
    return httpx.Client(transport=sync_cls(config), timeout=None)


def __getattr__(name: str) -> Any:
    if name == "BridgeTransport":
        return transports_for("httpx")[0]
    if name == "AsyncBridgeTransport":
        return transports_for("httpx")[1]
    raise AttributeError(name)
