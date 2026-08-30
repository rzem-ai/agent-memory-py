# agent-memory-py

Typed Python client, CLI, and Claude Code plugin for the
[agent-memory](https://github.com/rzem-ai/agent-memory) MCP server — the
Python twin of [`@rzem-ai/agent-memory-js`](https://github.com/rzem-ai/agent-memory.js).

The server exposes nine tools over MCP (semantic recall across a thoughts
corpus and a synced document vault, a tree of LLM-summarised digests, and a
KV store). This package wraps them in typed methods, parses the server's
frozen text formats back into structured objects, and ships the hooks that
give Claude Code sessions automatic long-term memory.

- **No `agent_id` anywhere** — the credential carries the namespace.
- **Fail-open by design** — a down memory server is "no memory this turn",
  never a broken session. Hooks always exit 0.
- **Frozen-text aware** — `memory_search` (documents/all), `memory_tree`, and
  `memory_read_document` return their payloads only as text in a frozen
  format; the parsers here mirror the server's formatters exactly.

Requires Python 3.12+. Depends only on the official `mcp` SDK (2.x).

## Library

```python
from agent_memory import AgentMemory

async with await AgentMemory.connect(
    url="http://127.0.0.1:3010/mcp",   # default; env AGENT_MEMORY_URL
    token=os.environ["AGENT_MEMORY_TOKEN"],  # env AGENT_MEMORY_TOKEN
    timeout=120.0,                     # seconds; generous: cold Ollama embeds
) as mem:
    res = await mem.search("badgers", corpus="all", limit=5)
    # res.hits: ThoughtHit | DocumentHit dataclasses; res.degraded; res.text (raw)

    await mem.capture("Alex prefers pnpm for the angus2 workspace", ["claude-code"])
    await mem.forget(thought_id)               # bool
    await mem.read_document(doc_id)            # ParsedDocument | None
    await mem.tree.list("mail/2026")           # ParsedTreeList(scope, entries)
    await mem.kv.set("key", {"any": "json"})
    await mem.kv.get("key")                    # None on miss
```

Blocking code (scripts, notebooks) gets the same surface without `await`:

```python
from agent_memory import SyncAgentMemory

with SyncAgentMemory.connect(token=...) as mem:
    print(mem.search("badgers").hits)
```

`SyncAgentMemory` drives the async client on a private thread, so it also
works from inside an already-running event loop.

Stdio (no auth; identity from the server's TOML):

```python
from agent_memory import AgentMemory, StdioConnectOptions

mem = await AgentMemory.connect(StdioConnectOptions(
    server_path="/path/to/agent-memory/dist/stdio.js",
    config_path="/path/to/mcp.toml",
))
```

`CustomTransportOptions(instance=...)` accepts anything `mcp.client.Client`
does — a `Transport`, an in-process `MCPServer`, or `StdioServerParameters`.

Errors: `AuthError` (HTTP 401) and `TransportError` (network/timeout) at the
transport layer; `ToolError` with `.kind` (`scope_denied | validation |
no_namespace | degraded | failed`) for `isError` tool results; `ParseError`
when the frozen text drifts. Misses are values, not raises: `forget` →
`False`, `kv.get` → `None`, `read_document`/`tree.read` → `None`. Note the
server-side quirk: a stored KV `null` is indistinguishable from a missing key.

Field names are snake_case (`document_id`, `vault_path`, `last_appended_at`);
the one keyword collision is `TreeWindow.from_`. `--json` CLI output and
`agent_memory.cli.output.to_jsonable` render it as `from`, matching the JS
envelope byte-for-byte.

## CLI

```
agent-memory-py search "badgers" [--corpus all|thoughts|documents] [--limit 5] [--json]
agent-memory-py capture "content" --tag claude-code --tag project:x
agent-memory-py forget <uuid>
agent-memory-py read-document <doc-id> [--max-chars 20000]
agent-memory-py tree list [path] | tree read <path> | tree search "query"
agent-memory-py kv get|set|delete|list ...
agent-memory-py tools | health
```

Config via `AGENT_MEMORY_URL` / `AGENT_MEMORY_TOKEN` / `AGENT_MEMORY_TIMEOUT_MS`
or `--url` / `--token` / `--timeout-ms` (flags go after the subcommand).
Default output is the server's frozen text verbatim; `--json` emits one
machine envelope (`{"ok":true,"tool","data","text"}` or
`{"ok":false,"error":{kind,message}}`). Exit codes: `0` ok, `1` tool/parse,
`2` usage, `3` transport, `4` auth. `--fail-open` forces exit 0 whatever
happened; `--quiet` silences output.

Run it without installing: `uvx agent-memory-py search "badgers"`.

## Claude Code plugin

The `plugin/` directory is a Claude Code plugin providing:

- **`.mcp.json`** registering the server over HTTP — sessions get the
  `mcp__agent-memory__*` tools directly.
- **Auto-recall**: a UserPromptSubmit hook searches memory with each prompt
  and injects hits as "Relevant long-term memory" (8s deadline, silent on
  miss or cold server).
- **Auto-capture**: a Stop hook distils durable facts from the finished turn
  via a one-shot `claude -p` (haiku) and captures them, tagged
  `auto-captured`. Server-side dedup absorbs repeats.
- **`using-memory` skill** teaching when to capture/search and the
  taint-external rule.

Install from this repo:

```
claude plugin marketplace add /path/to/agent-memory-py   # or the git URL
claude plugin install agent-memory@agent-memory
```

Set `AGENT_MEMORY_URL` and `AGENT_MEMORY_TOKEN` in the environment Claude
Code runs in. The hooks resolve the CLI via `$AGENT_MEMORY_CLI`, then `PATH`,
then `uvx agent-memory-py` (slow first run).

This plugin and the JS package's plugin share the name `agent-memory` — install
one or the other per machine, not both. Already registered agent-memory at user
scope? Remove one of the two entries — the plugin's `.mcp.json` is meant to be
the single source.

## Development

```
uv sync
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
uv run pytest
uv build

AGENT_MEMORY_LIVE=1 AGENT_MEMORY_URL=... AGENT_MEMORY_TOKEN=... uv run pytest tests/test_live.py
```

The parsers in `src/agent_memory/parse/` mirror the server's
`src/domain/recall.ts` formatters — a frozen cross-consumer contract. Change
them only in lockstep with the server. The fixtures in
`tests/parse/fixtures/` are byte-identical to the JS package's. First calls
against a cold server can take tens of seconds (Ollama model load); keep
client timeouts generous and consumers fail-open.
