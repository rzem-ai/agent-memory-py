"""SyncAgentMemory: the blocking wrapper for scripts and notebooks. Drives the
async client on a private event loop thread, so it works from plain
synchronous code and from inside a running event loop alike."""

import asyncio
from collections.abc import Iterator

import pytest

from agent_memory import (
    AuthError,
    CaptureOutcome,
    CustomTransportOptions,
    MergedSearchResponse,
    SyncAgentMemory,
    ToolError,
)

from .conftest import TOKEN
from .mock.http import HttpHarness
from .mock.server import CAPTURED_ID, KNOWN_THOUGHT_ID, KNOWN_TREE_PATH, TOOL_NAMES, create_mock_server


@pytest.fixture
def mem() -> Iterator[SyncAgentMemory]:
    with SyncAgentMemory.connect(CustomTransportOptions(instance=create_mock_server())) as mem:
        yield mem


def test_list_tools(mem: SyncAgentMemory) -> None:
    assert mem.list_tools() == TOOL_NAMES


def test_search_capture_forget(mem: SyncAgentMemory) -> None:
    res = mem.search("pnpm")
    assert isinstance(res, MergedSearchResponse)
    assert len(res.hits) == 2
    assert mem.capture("a fresh thought", ["t"]) == CaptureOutcome(captured=True, id=CAPTURED_ID, superseded=0)
    assert mem.forget(KNOWN_THOUGHT_ID) is True


def test_tree_and_kv_namespaces(mem: SyncAgentMemory) -> None:
    node = mem.tree.read(KNOWN_TREE_PATH)
    assert node is not None and node.state == "summarised"
    assert mem.tree.list().scope == "roots"
    assert len(mem.tree.search("kickoff").results) == 2
    mem.kv.set("k", [1, 2])
    assert mem.kv.get("k") == [1, 2]
    assert mem.kv.list() == {"k": [1, 2]}
    assert mem.kv.delete("k") is True


def test_read_document_and_raw(mem: SyncAgentMemory) -> None:
    assert mem.read_document("nope") is None
    assert mem.raw("memory_kv_get", {"key": "absent"}).structured == {"key": "absent", "found": False}


def test_tool_errors_propagate(mem: SyncAgentMemory) -> None:
    with pytest.raises(ToolError) as info:
        mem.capture("denied")
    assert info.value.kind == "scope_denied"


def test_close_is_idempotent_and_usable_after_context() -> None:
    mem = SyncAgentMemory.connect(CustomTransportOptions(instance=create_mock_server()))
    assert len(mem.list_tools()) == 9
    mem.close()
    mem.close()


def test_http_connect_and_auth_error(harness: HttpHarness) -> None:
    with SyncAgentMemory.connect(url=harness.mcp_url, token=TOKEN) as mem:
        assert len(mem.list_tools()) == 9
    with pytest.raises(AuthError):
        SyncAgentMemory.connect(url=harness.mcp_url, token="nope")


def test_works_from_inside_a_running_event_loop(harness: HttpHarness) -> None:
    async def inside() -> int:
        with SyncAgentMemory.connect(url=harness.mcp_url, token=TOKEN) as mem:
            return len(mem.list_tools())

    assert asyncio.run(inside()) == 9
