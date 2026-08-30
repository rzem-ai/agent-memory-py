from collections.abc import AsyncIterator

import pytest

from agent_memory import (
    AgentMemory,
    CaptureOutcome,
    CustomTransportOptions,
    DocumentHit,
    MergedSearchResponse,
    ThoughtHit,
    ThoughtsSearchResponse,
    ToolError,
)

from .mock.server import (
    CAPTURED_ID,
    DUPLICATE_OF_ID,
    KNOWN_DOCUMENT_ID,
    KNOWN_THOUGHT_ID,
    KNOWN_TREE_PATH,
    TOOL_NAMES,
    create_mock_server,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
async def mem() -> AsyncIterator[AgentMemory]:
    async with await AgentMemory.connect(CustomTransportOptions(instance=create_mock_server())) as mem:
        yield mem


async def test_list_tools_reports_the_nine_names_in_registration_order(mem: AgentMemory) -> None:
    assert await mem.list_tools() == TOOL_NAMES


async def test_search_over_thoughts_returns_structured_rows_verbatim(mem: AgentMemory) -> None:
    res = await mem.search("pnpm", corpus="thoughts")
    assert isinstance(res, ThoughtsSearchResponse)
    assert res.corpus == "thoughts"
    assert res.mode == "recency_weighted"
    assert len(res.results) == 1
    assert res.results[0]["id"] == KNOWN_THOUGHT_ID
    assert res.results[0]["agent_id"] == "angus"
    assert res.results[0]["score"] == 0.812
    assert "mode: recency_weighted" in res.text


async def test_search_over_all_parses_frozen_text_into_typed_hits(mem: AgentMemory) -> None:
    res = await mem.search("pnpm")
    assert isinstance(res, MergedSearchResponse)
    assert res.corpus == "all"
    assert res.degraded is False
    assert len(res.hits) == 2
    assert isinstance(res.hits[0], ThoughtHit) and res.hits[0].id == KNOWN_THOUGHT_ID
    assert isinstance(res.hits[1], DocumentHit) and res.hits[1].document_id == KNOWN_DOCUMENT_ID
    assert res.text.startswith("corpus: all")


async def test_search_reports_degradation_without_raising_for_corpus_all(mem: AgentMemory) -> None:
    res = await mem.search("degraded")
    assert isinstance(res, MergedSearchResponse)
    assert res.degraded is True
    assert len(res.hits) == 1


async def test_degraded_empty_documents_search_raises_tool_error_of_kind_degraded(mem: AgentMemory) -> None:
    with pytest.raises(ToolError) as info:
        await mem.search("degraded", corpus="documents")
    assert info.value.kind == "degraded"


async def test_scope_denial_surfaces_as_tool_error_scope_denied(mem: AgentMemory) -> None:
    with pytest.raises(ToolError) as info:
        await mem.capture("denied")
    assert info.value.kind == "scope_denied"
    assert info.value.tool == "memory_capture"


async def test_capture_returns_the_structured_outcome(mem: AgentMemory) -> None:
    assert await mem.capture("a fresh thought", ["tag"]) == CaptureOutcome(captured=True, id=CAPTURED_ID, superseded=0)
    assert await mem.capture("supersede") == CaptureOutcome(captured=True, id=CAPTURED_ID, superseded=2)
    assert await mem.capture("dup") == CaptureOutcome(captured=False, duplicate_of=DUPLICATE_OF_ID)


async def test_forget_returns_the_flag_for_hit_and_miss(mem: AgentMemory) -> None:
    assert await mem.forget(KNOWN_THOUGHT_ID) is True
    assert await mem.forget("00000000-0000-0000-0000-000000000000") is False


async def test_read_document_parses_frozen_text_and_a_miss_is_none(mem: AgentMemory) -> None:
    doc = await mem.read_document(KNOWN_DOCUMENT_ID)
    assert doc is not None
    assert doc.id == KNOWN_DOCUMENT_ID
    assert doc.title == "Project kickoff notes"
    assert doc.taint == "external"
    assert doc.body_source == "vault"
    assert await mem.read_document("nope") is None


async def test_tree_list_read_search_parse_frozen_formats_and_read_miss_is_none(mem: AgentMemory) -> None:
    listing = await mem.tree.list()
    assert listing.scope == "roots"
    assert len(listing.entries) == 2

    node = await mem.tree.read(KNOWN_TREE_PATH)
    assert node is not None
    assert node.path == KNOWN_TREE_PATH
    assert node.state == "summarised"
    assert await mem.tree.read("missing/2031") is None

    search = await mem.tree.search("kickoff planning")
    assert search.query == "kickoff planning"
    assert len(search.results) == 2


async def test_kv_round_trip_and_a_miss_is_none(mem: AgentMemory) -> None:
    assert await mem.kv.get("absent") is None
    await mem.kv.set("config", {"theme": "dark", "n": 2})
    assert await mem.kv.get("config") == {"theme": "dark", "n": 2}
    assert await mem.kv.list() == {"config": {"theme": "dark", "n": 2}}
    assert await mem.kv.delete("config") is True
    assert await mem.kv.delete("config") is False
    assert await mem.kv.list() == {}


async def test_kv_stored_null_is_indistinguishable_from_a_miss(mem: AgentMemory) -> None:
    await mem.kv.set("nullish", None)
    assert await mem.kv.get("nullish") is None


async def test_raw_returns_text_and_structured_as_sent(mem: AgentMemory) -> None:
    reply = await mem.raw("memory_kv_get", {"key": "absent"})
    assert reply.text == "No value found for key 'absent'"
    assert reply.structured == {"key": "absent", "found": False}


async def test_close_is_idempotent() -> None:
    mem = await AgentMemory.connect(CustomTransportOptions(instance=create_mock_server()))
    await mem.close()
    await mem.close()
