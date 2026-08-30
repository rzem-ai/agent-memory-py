"""Transport construction for ``AgentMemory.connect``. HTTP is the primary path
(streamable HTTP + bearer token); stdio spawns the server's ``dist/stdio.js``;
"custom" accepts any pre-built Transport or in-process ``MCPServer`` (the test
seam, and the escape hatch for exotic setups)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import Implementation

DEFAULT_URL = "http://127.0.0.1:3010/mcp"

DEFAULT_TIMEOUT = 120.0
"""Generous default (seconds): a cold Ollama embed on the server can take tens
of seconds, and the server's own embed timeout is 30s per leg."""

CLIENT_INFO = Implementation(name="agent-memory-py", version="0.1.0")


class StatusRecorder:
    """Remembers the last HTTP status the transport saw. The MCP SDK folds a
    non-2xx response into a generic INTERNAL_ERROR, losing the status — this
    hook is how a 401 becomes AuthError instead of TransportError."""

    __slots__ = ("last_status",)

    def __init__(self) -> None:
        self.last_status: int | None = None

    async def __call__(self, response: httpx2.Response) -> None:
        self.last_status = response.status_code


@dataclass(frozen=True, slots=True)
class BuiltClient:
    client: Client
    timeout: float
    status: StatusRecorder | None


@dataclass(frozen=True, slots=True, kw_only=True)
class HttpConnectOptions:
    url: str | None = None
    """MCP endpoint URL. Default: AGENT_MEMORY_URL env, else http://127.0.0.1:3010/mcp."""
    token: str | None = None
    """Bearer token. Default: AGENT_MEMORY_TOKEN env. Omitted = unauthenticated."""
    headers: dict[str, str] = field(default_factory=dict)
    """Extra headers merged into every request."""
    timeout: float | None = None
    """Per-request timeout in seconds. Default: AGENT_MEMORY_TIMEOUT_MS env / 1000, else 120."""
    era: Literal["legacy", "auto"] = "legacy"
    """Protocol era negotiation: "legacy" (default, no probe round-trip) or "auto"."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StdioConnectOptions:
    server_path: str
    """Path to the server's dist/stdio.js."""
    config_path: str | None = None
    """TOML config path, passed as --config."""
    command: str = "node"
    """Executable to run server_path with."""
    env: dict[str, str] | None = None
    timeout: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CustomTransportOptions:
    instance: Any
    """Anything ``mcp.client.Client`` accepts as a server: a ``Transport``, an
    in-process ``MCPServer``/``Server``, or ``StdioServerParameters``."""
    timeout: float | None = None


ConnectOptions = HttpConnectOptions | StdioConnectOptions | CustomTransportOptions


def resolve_timeout(options: ConnectOptions) -> float:
    if options.timeout is not None:
        return options.timeout
    raw = os.environ.get("AGENT_MEMORY_TIMEOUT_MS")
    try:
        ms = float(raw) if raw else 0.0
    except ValueError:
        ms = 0.0
    return ms / 1000 if ms > 0 else DEFAULT_TIMEOUT


def resolve_url(options: HttpConnectOptions) -> str:
    return options.url or os.environ.get("AGENT_MEMORY_URL") or DEFAULT_URL


def build_client(options: ConnectOptions) -> BuiltClient:
    timeout = resolve_timeout(options)
    server: Any
    status: StatusRecorder | None = None
    mode: Literal["legacy", "auto"] = "legacy"
    if isinstance(options, CustomTransportOptions):
        server = options.instance
    elif isinstance(options, StdioConnectOptions):
        args = [options.server_path]
        if options.config_path is not None:
            args += ["--config", options.config_path]
        server = StdioServerParameters(command=options.command, args=args, env=options.env)
    else:
        token = options.token if options.token is not None else os.environ.get("AGENT_MEMORY_TOKEN")
        headers = dict(options.headers)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        http_client = create_mcp_http_client(
            headers=headers or None,
            timeout=httpx2.Timeout(timeout, read=max(timeout, 300.0)),
        )
        status = StatusRecorder()
        http_client.event_hooks = {"response": [status]}
        server = streamable_http_client(resolve_url(options), http_client=http_client)
        mode = options.era
    client = Client(server, read_timeout_seconds=timeout, client_info=CLIENT_INFO, mode=mode, cache=None)
    return BuiltClient(client=client, timeout=timeout, status=status)
