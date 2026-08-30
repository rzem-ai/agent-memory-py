"""Live integration suite against a real agent-memory server. Opt-in:

    AGENT_MEMORY_LIVE=1 AGENT_MEMORY_URL=... AGENT_MEMORY_TOKEN=... uv run pytest tests/test_live.py

Non-destructive by design: KV uses a namespaced test key, capture forgets
its own row. Generous timeouts — a cold Ollama embed can take tens of
seconds on the first call."""

import os
from collections.abc import AsyncIterator

import pytest

from agent_memory import AgentMemory, MergedSearchResponse, ThoughtsSearchResponse

LIVE = os.environ.get("AGENT_MEMORY_LIVE") == "1"
TEST_KEY = "agent-memory-py:live-test"

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not LIVE, reason="AGENT_MEMORY_LIVE=1 not set")]


@pytest.fixture
async def mem() -> AsyncIterator[AgentMemory]:
    async with await AgentMemory.connect(timeout=180.0) as mem:
        yield mem


async def test_lists_the_nine_tools(mem: AgentMemory) -> None:
    assert len(await mem.list_tools()) == 9


async def test_kv_round_trip_with_a_namespaced_key(mem: AgentMemory) -> None:
    await mem.kv.set(TEST_KEY, {"stamp": "live-test", "n": 1})
    assert await mem.kv.get(TEST_KEY) == {"stamp": "live-test", "n": 1}
    assert await mem.kv.delete(TEST_KEY) is True
    assert await mem.kv.get(TEST_KEY) is None


async def test_search_over_thoughts_returns_structured_rows(mem: AgentMemory) -> None:
    res = await mem.search("memory client", corpus="thoughts", limit=3)
    assert isinstance(res, ThoughtsSearchResponse)
    for row in res.results:
        assert isinstance(row["id"], str)
        assert isinstance(row["similarity"], float | int)


async def test_search_over_all_parses_the_live_frozen_text(mem: AgentMemory) -> None:
    res = await mem.search("memory client", corpus="all", limit=3)
    # The parse itself is the assertion: drift in the frozen text raises.
    assert isinstance(res, MergedSearchResponse)
    assert isinstance(res.degraded, bool)


async def test_capture_then_forget_our_own_row(mem: AgentMemory) -> None:
    outcome = await mem.capture(
        f"agent-memory-py live-test marker {TEST_KEY} - safe to forget", ["agent-memory-py", "live-test"]
    )
    # A dedup skip (from an earlier run inside 48h) is a valid success too.
    thought_id = outcome.id if outcome.captured else outcome.duplicate_of
    assert isinstance(thought_id, str)
    assert isinstance(await mem.forget(thought_id), bool)
