"""SyncAgentMemory — a blocking facade over the async client for scripts and
notebooks. All work runs on a private event-loop thread inside ONE long-lived
task: the mcp SDK's anyio cancel scopes must be entered and exited by the same
task, so connect, every call, and close are queued to that single driver
rather than spawned as separate tasks. Usable from plain synchronous code and
from inside a running event loop alike."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, Self

from .client import AgentMemory, CaptureOutcome, SearchResponse, ToolReply
from .connect import ConnectOptions, HttpConnectOptions
from .types import Corpus, ParsedDocument, ParsedTreeList, ParsedTreeNode, ParsedTreeSearch, RelevanceMode

_Job = tuple[Callable[[], Awaitable[Any]], "concurrent.futures.Future[Any]"]


class _Driver:
    """One thread, one loop, one task that awaits queued coroutines in order."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[_Job | None] | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="agent-memory-sync", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        self._queue = asyncio.Queue()
        self._ready.set()
        while (job := await self._queue.get()) is not None:
            fn, future = job
            try:
                future.set_result(await fn())
            except BaseException as err:
                future.set_exception(err)

    def submit[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        assert self._queue is not None
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (fn, future))
        return future.result()

    def stop(self) -> None:
        if self._queue is not None and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            self._thread.join()


class SyncTreeApi:
    __slots__ = ("_driver", "_mem")

    def __init__(self, driver: _Driver, mem: AgentMemory) -> None:
        self._driver = driver
        self._mem = mem

    def list(self, path: str | None = None) -> ParsedTreeList:
        return self._driver.submit(lambda: self._mem.tree.list(path))

    def read(self, path: str) -> ParsedTreeNode | None:
        return self._driver.submit(lambda: self._mem.tree.read(path))

    def search(self, query: str, *, limit: int | None = None) -> ParsedTreeSearch:
        return self._driver.submit(lambda: self._mem.tree.search(query, limit=limit))


class SyncKvApi:
    __slots__ = ("_driver", "_mem")

    def __init__(self, driver: _Driver, mem: AgentMemory) -> None:
        self._driver = driver
        self._mem = mem

    def get(self, key: str) -> Any:
        return self._driver.submit(lambda: self._mem.kv.get(key))

    def set(self, key: str, value: Any) -> None:
        self._driver.submit(lambda: self._mem.kv.set(key, value))

    def delete(self, key: str) -> bool:
        return self._driver.submit(lambda: self._mem.kv.delete(key))

    def list(self) -> dict[str, Any]:
        return self._driver.submit(self._mem.kv.list)


class SyncAgentMemory:
    """Blocking client. Use ``with SyncAgentMemory.connect(...) as mem:`` or
    call ``close()`` yourself."""

    __slots__ = ("_driver", "_mem", "kv", "tree")

    tree: SyncTreeApi
    kv: SyncKvApi

    def __init__(self, driver: _Driver, mem: AgentMemory) -> None:
        self._driver: _Driver | None = driver
        self._mem = mem
        self.tree = SyncTreeApi(driver, mem)
        self.kv = SyncKvApi(driver, mem)

    @classmethod
    def connect(
        cls,
        options: ConnectOptions | None = None,
        *,
        url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ) -> Self:
        resolved = options if options is not None else HttpConnectOptions(url=url, token=token, timeout=timeout)
        driver = _Driver()
        try:
            mem = driver.submit(lambda: AgentMemory.connect(resolved))
        except BaseException:
            driver.stop()
            raise
        return cls(driver, mem)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        driver, self._driver = self._driver, None
        if driver is None:
            return
        try:
            driver.submit(self._mem.close)
        finally:
            driver.stop()

    def _submit[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        if self._driver is None:
            raise RuntimeError("SyncAgentMemory is closed")
        return self._driver.submit(fn)

    def list_tools(self) -> list[str]:
        return self._submit(self._mem.list_tools)

    def search(
        self,
        query: str,
        *,
        corpus: Corpus = "all",
        limit: int | None = None,
        relevance_mode: RelevanceMode | None = None,
        relevance_value: float | None = None,
    ) -> SearchResponse:
        return self._submit(
            lambda: self._mem.search(
                query, corpus=corpus, limit=limit, relevance_mode=relevance_mode, relevance_value=relevance_value
            )
        )

    def capture(self, content: str, tags: list[str] | None = None) -> CaptureOutcome:
        return self._submit(lambda: self._mem.capture(content, tags))

    def forget(self, thought_id: str) -> bool:
        return self._submit(lambda: self._mem.forget(thought_id))

    def read_document(self, document_id: str, *, max_chars: int | None = None) -> ParsedDocument | None:
        return self._submit(lambda: self._mem.read_document(document_id, max_chars=max_chars))

    def raw(self, name: str, args: dict[str, Any] | None = None) -> ToolReply:
        return self._submit(lambda: self._mem.raw(name, args))
