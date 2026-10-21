"""Route HTTP requests to the API handlers. Shared by the server and the in-process transports."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Union

from . import anthropic_api, openai_api, responses_api
from .config import BridgeConfig
from .core import ChatResult, acomplete_chat, aiter_chat, complete_chat, iter_chat
from .errors import BridgeError

log = logging.getLogger("clibridge")

JSON_HEADERS = {"content-type": "application/json"}
SSE_HEADERS = {"content-type": "text/event-stream", "cache-control": "no-cache"}


@dataclass
class Reply:
    status: int
    headers: Dict[str, str]
    body: Union[bytes, Iterator[bytes], AsyncIterator[bytes]]
    close: Optional[Callable[[], Any]] = None  # sync callable, or coroutine function for async bodies


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _json(status: int, obj: Any) -> Reply:
    return Reply(status, dict(JSON_HEADERS), _dumps(obj))


def _route(method: str, path: str) -> Optional[str]:
    path = path.split("?", 1)[0].rstrip("/")
    if method == "GET":
        if path in ("", "/health", "/healthz"):
            return "health"
        if path.endswith("/models"):
            return "models"
        if "/models/" in path:
            return "model"
    elif method == "POST":
        for suffix, route in (("/chat/completions", "chat"), ("/responses", "responses"),
                              ("/messages/count_tokens", "count_tokens"), ("/messages", "messages")):
            if path.endswith(suffix):
                return route
    return None


def _sse(event: Any) -> bytes:
    if isinstance(event, tuple):
        name, data = event
        return b"event: " + name.encode() + b"\ndata: " + _dumps(data) + b"\n\n"
    return b"data: " + _dumps(event) + b"\n\n"


def _openai_error(message: str, status: int) -> Dict[str, Any]:
    return {"error": {"message": message, "type": "clibridge_error", "code": status}}


class _ChatApi:
    def __init__(self, body: Dict[str, Any], app: "App"):
        self.body = body

    def prepare(self) -> Dict[str, Any]:
        return self.body

    def payload(self, result: ChatResult) -> Dict[str, Any]:
        return openai_api.completion(result, self.body)

    def encoder(self):
        return openai_api.ChunkEncoder(self.body)

    def finished(self, result: ChatResult) -> None:
        pass

    trailer = b"data: [DONE]\n\n"

    def error_frame(self, message: str) -> bytes:
        return _sse(_openai_error(message, 500))

    def error_body(self, message: str, status: int) -> Dict[str, Any]:
        return _openai_error(message, status)


class _ResponsesApi(_ChatApi):
    def __init__(self, body: Dict[str, Any], app: "App"):
        self.body = body
        self.store = app.store
        self.encoder_ = responses_api.ResponseEncoder(body)
        self.conversation: list = []

    def prepare(self) -> Dict[str, Any]:
        request, self.conversation = responses_api.to_chat_request(self.body, self.store)
        return request

    def payload(self, result: ChatResult) -> Dict[str, Any]:
        return self.encoder_.response(result)

    def encoder(self):
        return self.encoder_

    def finished(self, result: ChatResult) -> None:
        self.store.put(self.encoder_.id, self.conversation + [responses_api.assistant_message(result)])

    trailer = b""

    def error_frame(self, message: str) -> bytes:
        return _sse(self.encoder_.error(message))


class _MessagesApi(_ChatApi):
    def prepare(self) -> Dict[str, Any]:
        return anthropic_api.to_chat_request(self.body)

    def payload(self, result: ChatResult) -> Dict[str, Any]:
        return anthropic_api.message(result, self.body)

    def encoder(self):
        return anthropic_api.MessageEncoder(self.body)

    trailer = b""

    def error_frame(self, message: str) -> bytes:
        return _sse(("error", anthropic_api.error_body(message, 500)))

    def error_body(self, message: str, status: int) -> Dict[str, Any]:
        return anthropic_api.error_body(message, status)


_APIS = {"chat": _ChatApi, "responses": _ResponsesApi, "messages": _MessagesApi}


class App:
    """Handles OpenAI Chat Completions, OpenAI Responses and Anthropic Messages requests."""

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.store = responses_api.ResponseStore()

    def _simple(self, route: Optional[str], method: str, path: str, body: bytes):
        """Handle everything that doesn't run a CLI. Returns (reply, api, parsed body)."""
        if route is None:
            return _json(404, _openai_error(f"Unknown endpoint {method} {path}", 404)), None
        if route == "health":
            return _json(200, {"status": "ok"}), None
        if route == "models":
            return _json(200, openai_api.models_list()), None
        if route == "model":
            name = path.split("?", 1)[0].split("/models/", 1)[1]
            return _json(200, {"id": name, "object": "model", "created": int(time.time()), "owned_by": "clibridge"}), None
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            data = None
        if not isinstance(data, dict):
            return _json(400, _openai_error("Request body must be a JSON object", 400)), None
        if route == "count_tokens":
            return _json(200, anthropic_api.count_tokens(data)), None
        return None, _APIS[route](data, self)

    def dispatch(self, method: str, path: str, body: bytes) -> Reply:
        reply, api = self._simple(_route(method, path), method, path, body)
        if reply is not None:
            return reply
        try:
            request = api.prepare()
            if request.get("stream"):
                items = iter_chat(request, self.config)
                first = next(items)  # surface startup errors as a proper HTTP error
                return Reply(200, dict(SSE_HEADERS), self._stream(api, first, items), items.close)
            result = complete_chat(request, self.config)
            api.finished(result)
            return _json(200, api.payload(result))
        except BridgeError as exc:
            log.warning("%s %s failed: %s", method, path, exc)
            return _json(exc.status, api.error_body(str(exc), exc.status))
        except Exception as exc:
            log.exception("%s %s crashed", method, path)
            return _json(500, api.error_body(f"clibridge internal error: {exc!r}", 500))

    def _stream(self, api, first, items) -> Iterator[bytes]:
        encoder = api.encoder()
        try:
            for event in encoder.start():
                yield _sse(event)
            for item in _prepend(first, items):
                if isinstance(item, ChatResult):
                    api.finished(item)
                for event in encoder.encode(item):
                    yield _sse(event)
            if api.trailer:
                yield api.trailer
        except BridgeError as exc:
            log.warning("stream failed: %s", exc)
            yield api.error_frame(str(exc))
        except Exception as exc:
            log.exception("stream crashed")
            yield api.error_frame(f"clibridge internal error: {exc!r}")
        finally:
            items.close()

    async def adispatch(self, method: str, path: str, body: bytes) -> Reply:
        reply, api = self._simple(_route(method, path), method, path, body)
        if reply is not None:
            return reply
        try:
            request = api.prepare()
            if request.get("stream"):
                items = aiter_chat(request, self.config)
                first = await items.__anext__()
                return Reply(200, dict(SSE_HEADERS), self._astream(api, first, items), items.aclose)
            result = await acomplete_chat(request, self.config)
            api.finished(result)
            return _json(200, api.payload(result))
        except BridgeError as exc:
            log.warning("%s %s failed: %s", method, path, exc)
            return _json(exc.status, api.error_body(str(exc), exc.status))
        except Exception as exc:
            log.exception("%s %s crashed", method, path)
            return _json(500, api.error_body(f"clibridge internal error: {exc!r}", 500))

    async def _astream(self, api, first, items) -> AsyncIterator[bytes]:
        encoder = api.encoder()
        try:
            for event in encoder.start():
                yield _sse(event)
            item = first
            while True:
                if isinstance(item, ChatResult):
                    api.finished(item)
                for event in encoder.encode(item):
                    yield _sse(event)
                try:
                    item = await items.__anext__()
                except StopAsyncIteration:
                    break
            if api.trailer:
                yield api.trailer
        except BridgeError as exc:
            log.warning("stream failed: %s", exc)
            yield api.error_frame(str(exc))
        except Exception as exc:
            log.exception("stream crashed")
            yield api.error_frame(f"clibridge internal error: {exc!r}")
        finally:
            await items.aclose()


def _prepend(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from rest
