"""A small OpenAI- and Anthropic-compatible HTTP server (standard library only)."""

from __future__ import annotations

import hmac
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .app import App, _json, _openai_error
from .config import BridgeConfig

log = logging.getLogger("clibridge")

CORS_HEADERS = {
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "*",
    "access-control-allow-methods": "GET, POST, OPTIONS",
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "clibridge"
    server: "BridgeServer"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def _authorized(self) -> bool:
        expected = self.server.app.config.api_key
        if not expected:
            return True
        auth = self.headers.get("authorization") or ""
        given = auth[7:] if auth.lower().startswith("bearer ") else self.headers.get("x-api-key") or ""
        return hmac.compare_digest(given.encode(), expected.encode())

    def _handle(self) -> None:
        started = time.monotonic()
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        if not self._authorized():
            reply = _json(401, _openai_error("Invalid API key for this clibridge server", 401))
        else:
            reply = self.server.app.dispatch(self.command, self.path, body)
        try:
            self.send_response(reply.status)
            for key, value in {**reply.headers, **CORS_HEADERS}.items():
                self.send_header(key, value)
            if isinstance(reply.body, bytes):
                self.send_header("content-length", str(len(reply.body)))
                self.end_headers()
                self.wfile.write(reply.body)
            else:
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                for part in reply.body:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            log.info("client disconnected")
            self.close_connection = True
        finally:
            if not isinstance(reply.body, bytes):
                reply.body.close()
            if reply.close is not None:
                reply.close()
        log.info("%s %s -> %s (%.1fs)", self.command, self.path, reply.status, time.monotonic() - started)

    def log_message(self, format: str, *args) -> None:  # silence the default stderr access log
        log.debug(format, *args)


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, config: Optional[BridgeConfig] = None):
        super().__init__((host, port), _Handler)
        self.app = App(config)

    def handle_error(self, request, client_address) -> None:
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            log.debug("client %s disconnected", client_address)
            return
        super().handle_error(request, client_address)


def serve(host: str = "127.0.0.1", port: int = 8787, config: Optional[BridgeConfig] = None) -> None:
    server = BridgeServer(host, port, config)
    url = f"http://{host}:{server.server_port}"
    print(f"clibridge listening on {url}")
    print(f"  OpenAI:    OPENAI_BASE_URL={url}/v1  OPENAI_API_KEY=anything  model=claude-code | codex")
    print(f"  Anthropic: ANTHROPIC_BASE_URL={url}  ANTHROPIC_API_KEY=anything")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
