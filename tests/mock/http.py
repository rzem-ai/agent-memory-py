"""A minimal HTTP harness mirroring the real server's src/http.ts bridge:
uvicorn in a background thread in front of the mock MCPServer's streamable
HTTP app (stateless), static bearer-token auth answered with the RFC 9728-style
401 challenge, and the optional /memory path prefix stripped."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import uvicorn
from starlette.types import ASGIApp, Receive, Scope, Send

from .server import create_mock_server

_PREFIX_RE = re.compile(r"^/memory(?=/|$)")


async def _send_json(send: Send, status: int, body: dict[str, Any], extra_headers: list[tuple[bytes, bytes]]) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), *extra_headers],
        }
    )
    await send({"type": "http.response.body", "body": payload})


class _Bridge:
    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        path = _PREFIX_RE.sub("", scope["path"]) or "/"
        if path == "/health":
            await _send_json(send, 200, {"status": "ok"}, [])
            return
        if path not in ("/mcp", "/"):
            await _send_json(send, 404, {"error": "not_found"}, [])
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        if headers.get("authorization") != f"Bearer {self._token}":
            challenge = b'Bearer resource_metadata="http://127.0.0.1/.well-known/oauth-protected-resource"'
            await _send_json(send, 401, {"error": "unauthorized"}, [(b"www-authenticate", challenge)])
            return
        await self._app({**scope, "path": "/mcp", "raw_path": b"/mcp"}, receive, send)


@dataclass(frozen=True)
class HttpHarness:
    origin: str
    """Base origin, e.g. http://127.0.0.1:49152 — append /mcp or /memory/mcp."""
    token: str
    _server: uvicorn.Server
    _thread: threading.Thread

    @property
    def mcp_url(self) -> str:
        return f"{self.origin}/mcp"

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_http_harness(token: str) -> HttpHarness:
    mcp_app = create_mock_server().streamable_http_app(stateless_http=True)
    port = _free_port()
    config = uvicorn.Config(_Bridge(mcp_app, token), host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), name="mock-agent-memory-http", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("mock HTTP harness failed to start")
        time.sleep(0.01)
    return HttpHarness(origin=f"http://127.0.0.1:{port}", token=token, _server=server, _thread=thread)
