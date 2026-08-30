import pytest

from agent_memory import AgentMemory, AuthError, MergedSearchResponse, TransportError

from .conftest import TOKEN
from .mock.http import HttpHarness

pytestmark = pytest.mark.anyio


async def test_connects_with_bearer_token_and_lists_the_nine_tools(harness: HttpHarness) -> None:
    async with await AgentMemory.connect(url=harness.mcp_url, token=TOKEN) as mem:
        tools = await mem.list_tools()
    assert len(tools) == 9
    assert tools[0] == "memory_search"


async def test_tolerates_the_memory_path_prefix(harness: HttpHarness) -> None:
    async with await AgentMemory.connect(url=f"{harness.origin}/memory/mcp", token=TOKEN) as mem:
        assert len(await mem.list_tools()) == 9


async def test_real_call_round_trips_search_over_all(harness: HttpHarness) -> None:
    async with await AgentMemory.connect(url=harness.mcp_url, token=TOKEN) as mem:
        res = await mem.search("pnpm")
    assert isinstance(res, MergedSearchResponse)
    assert len(res.hits) == 2


async def test_env_vars_supply_url_and_token(harness: HttpHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MEMORY_URL", harness.mcp_url)
    monkeypatch.setenv("AGENT_MEMORY_TOKEN", TOKEN)
    async with await AgentMemory.connect() as mem:
        assert len(await mem.list_tools()) == 9


async def test_wrong_token_surfaces_as_auth_error(harness: HttpHarness) -> None:
    with pytest.raises(AuthError):
        await AgentMemory.connect(url=harness.mcp_url, token="nope")


async def test_missing_token_surfaces_as_auth_error(harness: HttpHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_MEMORY_TOKEN", raising=False)
    with pytest.raises(AuthError):
        await AgentMemory.connect(url=harness.mcp_url)


async def test_unreachable_server_surfaces_as_transport_error() -> None:
    with pytest.raises(TransportError) as info:
        await AgentMemory.connect(url="http://127.0.0.1:9/mcp", token=TOKEN, timeout=2.0)
    assert not isinstance(info.value, AuthError)


async def test_slow_tool_call_beyond_timeout_surfaces_as_transport_error(harness: HttpHarness) -> None:
    async with await AgentMemory.connect(url=harness.mcp_url, token=TOKEN, timeout=0.1) as mem:
        with pytest.raises(TransportError):
            await mem.search("slow")
