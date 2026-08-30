#!/bin/bash
# UserPromptSubmit hook: recall relevant long-term memory for the new prompt.
# All real logic lives in the tested Python CLI (`agent-memory-py hook
# user-prompt`); this wrapper only resolves the CLI and stays silent on any
# failure. NEVER exit non-zero here — exit 2 would block and erase the prompt.

resolve_cli() {
  if [ -n "${AGENT_MEMORY_CLI:-}" ]; then
    echo "$AGENT_MEMORY_CLI"
    return 0
  fi
  if command -v agent-memory-py >/dev/null 2>&1; then
    echo "agent-memory-py"
    return 0
  fi
  # Last resort: slow on first run while uvx populates its cache.
  echo "uvx agent-memory-py"
  return 0
}

CLI="$(resolve_cli)" || exit 0
# shellcheck disable=SC2086 — the CLI value may legitimately carry arguments.
$CLI hook user-prompt 2>/dev/null || true
exit 0
