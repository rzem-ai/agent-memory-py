from collections.abc import Iterator

import pytest

from .mock.http import HttpHarness, start_http_harness

TOKEN = "test-token"


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def harness() -> Iterator[HttpHarness]:
    harness = start_http_harness(TOKEN)
    try:
        yield harness
    finally:
        harness.close()
