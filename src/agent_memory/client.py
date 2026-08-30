"""AgentMemory — the typed client over the agent-memory MCP server's nine
tools. Trusts structuredContent where the server sends real structure
(thoughts search, capture, forget, kv) and parses the frozen text where it
does not (merged search, tree, document read). Misses are values, never
raises; ``isError`` tool results become ToolError; transport failures become
TransportError/AuthError."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Self, cast

from mcp.client.client import Client
from mcp.types import CallToolResult, TextContent

from .connect import ConnectOptions, HttpConnectOptions, StatusRecorder, build_client
from .errors import AuthError, ToolError, TransportError
from .parse import parse_document, parse_merged_results, parse_tree_list, parse_tree_node, parse_tree_search
from .types import (
    Corpus,
    MergedHit,
    ParsedDocument,
    ParsedTreeList,
    ParsedTreeNode,
    ParsedTreeSearch,
    RankedThought,
    RelevanceMode,
)


@dataclass(frozen=True, slots=True)
class ThoughtsSearchResponse:
    mode: RelevanceMode
    results: list[RankedThought]
    text: str
    corpus: Literal["thoughts"] = "thoughts"


@dataclass(frozen=True, slots=True)
class MergedSearchResponse:
    corpus: Literal["documents", "all"]
    hits: list[MergedHit]
    degraded: bool
    text: str


SearchResponse = ThoughtsSearchResponse | MergedSearchResponse


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureOutcome:
    captured: bool
    id: str | None = None
    """The new thought's id when captured."""
    superseded: int = 0
    """How many stale thoughts the capture retired."""
    duplicate_of: str | None = None
    """The existing thought's id when the server skipped a near-duplicate."""


@dataclass(frozen=True, slots=True)
class ToolReply:
    text: str
    structured: dict[str, Any] = field(default_factory=dict)


def _to_transport_error(err: BaseException, status: StatusRecorder | None) -> TransportError:
    if isinstance(err, TransportError):
        return err
    wrapped: TransportError
    if status is not None and status.last_status == 401:
        wrapped = AuthError(f"Authentication failed: HTTP 401 ({err})")
    else:
        wrapped = TransportError(_describe(err))
    wrapped.__cause__ = err
    return wrapped


def _describe(err: BaseException) -> str:
    """A one-line message; unwrap single-member exception groups so the real
    failure (connection refused, timeout) is what the caller reads."""
    while isinstance(err, BaseExceptionGroup) and len(err.exceptions) == 1:
        err = err.exceptions[0]
    return str(err) or type(err).__name__


class TreeApi:
    __slots__ = ("_mem",)

    def __init__(self, mem: AgentMemory) -> None:
        self._mem = mem

    async def list(self, path: str | None = None) -> ParsedTreeList:
        args: dict[str, Any] = {"op": "list"}
        if path is not None:
            args["path"] = path
        return parse_tree_list((await self._mem.raw("memory_tree", args)).text)

    async def read(self, path: str) -> ParsedTreeNode | None:
        reply = await self._mem.raw("memory_tree", {"op": "read", "path": path})
        return None if reply.structured.get("found") is False else parse_tree_node(reply.text)

    async def search(self, query: str, *, limit: int | None = None) -> ParsedTreeSearch:
        args: dict[str, Any] = {"op": "search", "query": query}
        if limit is not None:
            args["limit"] = limit
        return parse_tree_search((await self._mem.raw("memory_tree", args)).text)


class KvApi:
    __slots__ = ("_mem",)

    def __init__(self, mem: AgentMemory) -> None:
        self._mem = mem

    async def get(self, key: str) -> Any:
        """The stored value, or None on a miss (a stored null is a miss too — server quirk)."""
        reply = await self._mem.raw("memory_kv_get", {"key": key})
        return None if reply.structured.get("found") is False else reply.structured.get("value")

    async def set(self, key: str, value: Any) -> None:
        await self._mem.raw("memory_kv_set", {"key": key, "value": value})

    async def delete(self, key: str) -> bool:
        return (await self._mem.raw("memory_kv_delete", {"key": key})).structured.get("deleted") is True

    async def list(self) -> dict[str, Any]:
        entries = (await self._mem.raw("memory_kv_list", {})).structured.get("entries")
        return cast(dict[str, Any], entries) if isinstance(entries, dict) else {}


class AgentMemory:
    """Async client. Use ``async with await AgentMemory.connect(...) as mem:``
    or call ``close()`` yourself."""

    __slots__ = ("_client", "_stack", "_status", "_timeout", "kv", "tree")

    tree: TreeApi
    kv: KvApi

    def __init__(
        self, client: Client, stack: AsyncExitStack, timeout: float, status: StatusRecorder | None = None
    ) -> None:
        self._client = client
        self._stack: AsyncExitStack | None = stack
        self._timeout = timeout
        self._status = status
        self.tree = TreeApi(self)
        self.kv = KvApi(self)

    @classmethod
    async def connect(
        cls,
        options: ConnectOptions | None = None,
        *,
        url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ) -> Self:
        """Connect and initialise. Pass an options object, or HTTP keyword shortcuts."""
        if options is None:
            options = HttpConnectOptions(url=url, token=token, timeout=timeout)
        built = build_client(options)
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(built.client)
        except Exception as err:
            await stack.aclose()
            raise _to_transport_error(err, built.status) from err
        return cls(built.client, stack, built.timeout, built.status)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    async def close(self) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    async def list_tools(self) -> list[str]:
        """The nine tool names as the server lists them — a connectivity check."""
        try:
            result = await self._client.list_tools()
        except Exception as err:
            raise _to_transport_error(err, self._status) from err
        return [tool.name for tool in result.tools]

    async def search(
        self,
        query: str,
        *,
        corpus: Corpus = "all",
        limit: int | None = None,
        relevance_mode: RelevanceMode | None = None,
        relevance_value: float | None = None,
    ) -> SearchResponse:
        args: dict[str, Any] = {"query": query, "corpus": corpus}
        if limit is not None:
            args["limit"] = limit
        if relevance_mode is not None:
            args["relevance_mode"] = relevance_mode
        if relevance_value is not None:
            args["relevance_value"] = relevance_value
        reply = await self.raw("memory_search", args)
        if corpus == "thoughts":
            results = reply.structured.get("results") or []
            return ThoughtsSearchResponse(
                mode=cast(RelevanceMode, reply.structured.get("mode")),
                results=cast(list[RankedThought], results),
                text=reply.text,
            )
        parsed = parse_merged_results(reply.text)
        return MergedSearchResponse(corpus=corpus, hits=parsed.hits, degraded=parsed.degraded, text=reply.text)

    async def capture(self, content: str, tags: list[str] | None = None) -> CaptureOutcome:
        reply = await self.raw("memory_capture", {"content": content, "tags": list(tags or [])})
        if reply.structured.get("skipped") is True:
            return CaptureOutcome(captured=False, duplicate_of=cast(str, reply.structured.get("duplicate_of")))
        superseded = reply.structured.get("superseded")
        return CaptureOutcome(
            captured=True,
            id=cast(str, reply.structured.get("id")),
            superseded=superseded if isinstance(superseded, int) else 0,
        )

    async def forget(self, thought_id: str) -> bool:
        """True when the thought existed and is now gone; False on a miss."""
        return (await self.raw("memory_forget", {"thought_id": thought_id})).structured.get("forgotten") is True

    async def read_document(self, document_id: str, *, max_chars: int | None = None) -> ParsedDocument | None:
        args: dict[str, Any] = {"document_id": document_id}
        if max_chars is not None:
            args["max_chars"] = max_chars
        reply = await self.raw("memory_read_document", args)
        return None if reply.structured.get("found") is False else parse_document(reply.text)

    async def raw(self, name: str, args: dict[str, Any] | None = None) -> ToolReply:
        """Low-level escape hatch: one tools/call, returning the frozen text and
        structuredContent as the server sent them."""
        try:
            result = await self._client.call_tool(name, args or {}, read_timeout_seconds=self._timeout)
        except Exception as err:
            raise _to_transport_error(err, self._status) from err
        if not isinstance(result, CallToolResult):
            raise TransportError(f"{name}: unexpected result type {type(result).__name__}")
        text = next((block.text for block in result.content if isinstance(block, TextContent)), "")
        if result.is_error:
            raise ToolError(name, text)
        structured = result.structured_content
        return ToolReply(text=text, structured=dict(structured) if isinstance(structured, dict) else {})
