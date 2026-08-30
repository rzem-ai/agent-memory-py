#!/bin/bash
# Stop hook: distil durable facts from the finished turn and capture them into
# agent-memory. All real logic lives in the tested Python CLI
# (`agent-memory-py hook stop`); this wrapper resolves the CLI and stays
# silent on any failure. Always exits 0 — memory must never block the session.

resolve_cli() {
  if [ -n "${AGENT_MEMORY_CLI:-}" ]; then
    echo "$AGENT_MEMORY_CLI"
    return 0
  fi
  if command -v agent-memory-py >/dev/null 2>&1; then
    echo "agent-memory-py"
    return 0
  fi
  echo "uvx agent-memory-py"
  return 0
}

CLI="$(resolve_cli)" || exit 0
# shellcheck disable=SC2086 — the CLI value may legitimately carry arguments.
$CLI hook stop || true
exit 0
