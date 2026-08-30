"""A mock agent-memory server for round-trip tests: the same nine tools, the
same input schemas, the same frozen text + structuredContent shapes as the
real server (mirrored from its src/tools/*), with canned recall results and
a live in-memory KV store. Input sentinels steer the error paths:

- search query "none" -> empty sentinel; "degraded" -> degraded note suffix;
  corpus "documents" + query "degraded" -> note-only isError result.
- capture content "dup" -> dedup skip; "denied" -> scope-denial error.
- forget/read/tree misses via the KNOWN_* ids below.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from agent_memory import DOCUMENTS_UNAVAILABLE_NOTE

from ..parse.helpers import fixture

KNOWN_THOUGHT_ID = "3f2a8c1e-5b7d-4e9a-8c3d-1a2b3c4d5e6f"
KNOWN_DOCUMENT_ID = "mail-2026-07-15-0042"
KNOWN_TREE_PATH = "mail/2026/07"
CAPTURED_ID = "0b0b0b0b-1111-2222-3333-444444444444"
DUPLICATE_OF_ID = "9e9e9e9e-8888-7777-6666-555555555555"

RANKED_THOUGHT: dict[str, Any] = {
    "id": KNOWN_THOUGHT_ID,
    "agent_id": "angus",
    "content": "Alex prefers pnpm for the angus2 workspace",
    "tags": ["claude-code", "project:angus2"],
    "similarity": 0.903,
    "score": 0.812,
    "created_at": "2026-08-01T09:00:00.000Z",
}

TOOL_NAMES = [
    "memory_search",
    "memory_capture",
    "memory_forget",
    "memory_read_document",
    "memory_tree",
    "memory_kv_get",
    "memory_kv_set",
    "memory_kv_delete",
    "memory_kv_list",
]


def text_result(text: str, structured: dict[str, Any] | None = None) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=structured if structured is not None else {"text": text},
    )


def error_result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)], structured_content={"error": text}, is_error=True
    )


def create_mock_server() -> MCPServer:
    server = MCPServer(name="agent-memory", version="0.0.0-mock")
    kv: dict[str, Any] = {}

    @server.tool(name="memory_search", description="mock memory_search")
    async def memory_search(
        query: str,
        corpus: Literal["thoughts", "documents", "all"] = "all",
        limit: int | None = None,
        relevance_mode: Literal["recency_weighted", "similarity", "recent", "since"] | None = None,
        relevance_value: float | None = None,
    ) -> CallToolResult:
        if query == "slow":
            await anyio.sleep(0.5)
        if query == "denied":
            return error_result(
                "Insufficient scope: 'memory_search' requires 'memory:read' (caller 'nobody' has: none)."
            )
        if corpus == "thoughts":
            results = [] if query == "none" else [RANKED_THOUGHT]
            text = (
                "No matching memories found."
                if not results
                else (
                    f"mode: recency_weighted\n\n[1] (id: {KNOWN_THOUGHT_ID}, score: 0.812, sim: 0.903, 2026-08-01)"
                    f" [angus] {RANKED_THOUGHT['content']} | tags: claude-code, project:angus2"
                )
            )
            return text_result(text, {"mode": "recency_weighted", "count": len(results), "results": results})
        if query == "degraded":
            if corpus == "documents":
                return error_result(DOCUMENTS_UNAVAILABLE_NOTE)
            return text_result(fixture("merged-all-degraded.txt"))
        if query == "none":
            return text_result("No matching memories found.")
        return text_result(fixture("merged-documents.txt" if corpus == "documents" else "merged-all.txt"))

    @server.tool(name="memory_capture", description="mock memory_capture")
    def memory_capture(content: str, tags: list[str]) -> CallToolResult:
        if content == "denied":
            return error_result(
                "Insufficient scope: 'memory_capture' requires 'memory:write' (caller 'reader' has: memory:read)."
            )
        if content == "dup":
            return text_result(
                "Memory skipped - near-duplicate of an existing thought"
                f" (id: {DUPLICATE_OF_ID}, cosine >= 0.85 within 48h).",
                {"id": None, "skipped": True, "duplicate_of": DUPLICATE_OF_ID},
            )
        if content == "supersede":
            return text_result(
                f"Memory captured successfully (id: {CAPTURED_ID}, superseded 2 stale thoughts)",
                {"id": CAPTURED_ID, "skipped": False, "superseded": 2},
            )
        return text_result(
            f"Memory captured successfully (id: {CAPTURED_ID})",
            {"id": CAPTURED_ID, "skipped": False, "superseded": 0},
        )

    @server.tool(name="memory_forget", description="mock memory_forget")
    def memory_forget(thought_id: str) -> CallToolResult:
        forgotten = thought_id == KNOWN_THOUGHT_ID
        return text_result(
            f"Forgot thought {thought_id}" if forgotten else f"Thought {thought_id} not found or already deleted",
            {"thought_id": thought_id, "forgotten": forgotten},
        )

    @server.tool(name="memory_read_document", description="mock memory_read_document")
    def memory_read_document(document_id: str, max_chars: int | None = None) -> CallToolResult:
        if document_id == KNOWN_DOCUMENT_ID:
            return text_result(
                fixture("document-full.txt"),
                {
                    "document_id": document_id,
                    "found": True,
                    "taint": "external",
                    "body_source": "vault",
                    "truncated": False,
                },
            )
        return text_result(f"No document with id '{document_id}'.", {"document_id": document_id, "found": False})

    @server.tool(name="memory_tree", description="mock memory_tree")
    def memory_tree(
        op: Literal["list", "read", "search"],
        path: str | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> CallToolResult:
        if op == "list":
            return text_result(fixture("tree-list-roots.txt"), {"op": "list", "count": 2})
        if op == "read":
            if not path or not path.strip():
                return error_result("Error: 'path' is required for op 'read'.")
            if path == KNOWN_TREE_PATH:
                return text_result(
                    fixture("tree-node-summarised.txt"), {"op": "read", "path": path, "state": "summarised"}
                )
            return text_result(f"No tree node at path '{path}'.", {"op": "read", "path": path, "found": False})
        return text_result(fixture("tree-search.txt"), {"op": "search", "count": 2})

    @server.tool(name="memory_kv_get", description="mock memory_kv_get")
    def memory_kv_get(key: str) -> CallToolResult:
        value = kv.get(key)
        # Mirrors the real server: a stored null is indistinguishable from a miss.
        if value is None:
            return text_result(f"No value found for key '{key}'", {"key": key, "found": False})
        return text_result(json.dumps(value, indent=2), {"key": key, "found": True, "value": value})

    @server.tool(name="memory_kv_set", description="mock memory_kv_set")
    def memory_kv_set(key: str, value: Any = None) -> CallToolResult:
        kv[key] = value
        return text_result(f"Set '{key}'", {"key": key, "set": True})

    @server.tool(name="memory_kv_delete", description="mock memory_kv_delete")
    def memory_kv_delete(key: str) -> CallToolResult:
        deleted = kv.pop(key, None) is not None or False
        return text_result(
            f"Deleted '{key}'" if deleted else f"Key '{key}' not found", {"key": key, "deleted": deleted}
        )

    @server.tool(name="memory_kv_list", description="mock memory_kv_list")
    def memory_kv_list() -> CallToolResult:
        if not kv:
            return text_result("No KV entries", {"count": 0, "entries": {}})
        lines = "\n".join(f"{k}: {json.dumps(v)}" for k, v in kv.items())
        return text_result(lines, {"count": len(kv), "entries": dict(kv)})

    return server
