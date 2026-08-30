# agent-memory-py — maintainer invariants

Typed Python client + CLI + Claude Code plugin for the agent-memory MCP server
(sibling repos: `../agent-memory` (server), `../agent-memory-js` (the JS twin
this package mirrors; keep the two in lockstep).

- **The frozen-text contract.** The parsers in `src/agent_memory/parse/` are
  exact inverses of the server's formatters in
  `../agent-memory/src/domain/recall.ts` (`formatMergedResults`,
  `formatTreeList`, `formatTreeNode`, `formatTreeSearch`, `formatDocument`).
  The server repo declares those text shapes frozen for external consumers.
  Never "fix" a parser to accept a new shape without a matching, deliberate
  server change — and update the fixtures in `tests/parse/fixtures/`
  byte-for-byte from the formatter code (they are identical to the JS repo's),
  not from memory.
- **structuredContent asymmetry.** Trust structuredContent only where the
  server sends real structure: search-over-thoughts, capture, forget, kv_*.
  Tree and read_document carry metadata-only structure (used for miss
  detection); merged search carries none. Payloads for those come from text.
- **Hooks always exit 0.** Exit 2 from a UserPromptSubmit hook blocks and
  erases the user's prompt. `src/agent_memory/cli/hook.py` swallows every
  failure and the bash wrappers in `plugin/scripts/` end in `exit 0`. Bare
  stdout on UserPromptSubmit is injected as context — never print anything
  there except the intended `hookSpecificOutput` JSON.
- **Fail-open posture.** A down/cold memory server is "no memory this turn",
  never an error surfaced to the session. Default client timeout is 120s
  (cold Ollama embeds); the recall hook uses a deliberate 8s deadline.
- **No agent_id.** The credential carries the namespace. If a tool call
  seems to need an agent parameter, the tool name is from the dead old
  server.
- **mcp 2.x quirks worth knowing.** The streamable-HTTP transport folds every
  non-2xx response into a generic INTERNAL_ERROR MCPError, so `connect.py`
  installs an httpx response hook (`StatusRecorder`) to turn a 401 into
  `AuthError`. anyio cancel scopes must exit in the task that entered them:
  tests use anyio's pytest plugin (not pytest-asyncio), and `SyncAgentMemory`
  funnels every call through one long-lived driver task.
- **Python naming.** Public fields are snake_case; `TreeWindow.from_` is the
  keyword-collision exception and `to_jsonable` renders it as `from` so the
  CLI envelope matches the JS one.
- **Gate:** `uv run ruff check src tests && uv run ruff format --check src tests
  && uv run mypy && uv run pytest && uv build` before claiming any change
  done. `tests/test_plugin.py` guards the plugin files; `claude plugin
  validate ./plugin` and `claude plugin validate .` must pass when plugin
  files change.
