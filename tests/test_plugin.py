"""Structural checks for the shipped Claude Code plugin, so a plugin regression
fails the test suite without needing the `claude` binary."""

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent

KNOWN_HOOK_EVENTS = {
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "SessionStart",
    "SessionEnd",
    "PreCompact",
    "Notification",
}


def read_json(relative: str) -> dict[str, Any]:
    data = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_plugin_json_names_the_plugin_agent_memory() -> None:
    manifest = read_json("plugin/.claude-plugin/plugin.json")
    assert manifest["name"] == "agent-memory"
    assert isinstance(manifest["version"], str)
    assert isinstance(manifest["description"], str)


def test_mcp_json_registers_exactly_one_http_server_named_agent_memory() -> None:
    servers = read_json("plugin/.mcp.json")["mcpServers"]
    assert list(servers) == ["agent-memory"]
    assert servers["agent-memory"]["type"] == "http"
    assert "${AGENT_MEMORY_URL" in servers["agent-memory"]["url"]
    assert "${AGENT_MEMORY_TOKEN}" in json.dumps(servers["agent-memory"]["headers"])


def test_hooks_json_uses_plugin_wrapper_format_with_known_events_and_portable_paths() -> None:
    hooks = read_json("plugin/hooks/hooks.json")["hooks"]
    assert sorted(hooks) == ["Stop", "UserPromptSubmit"]
    for event, matchers in hooks.items():
        assert event in KNOWN_HOOK_EVENTS
        for matcher in matchers:
            for hook in matcher["hooks"]:
                assert hook["type"] == "command"
                assert "${CLAUDE_PLUGIN_ROOT}" in hook["command"]


def test_hook_scripts_resolve_the_python_cli_and_never_exit_non_zero() -> None:
    for name in ("user-prompt-submit.sh", "stop.sh"):
        script = (ROOT / "plugin" / "scripts" / name).read_text(encoding="utf-8")
        assert script
        assert "exit 0" in script
        assert re.search(r"^\s*exit [1-9]", script, re.MULTILINE) is None
        assert "agent-memory-py" in script
        assert "uvx" in script
        assert "agent-memory-js" not in script
        assert "npx" not in script


def test_using_memory_skill_has_frontmatter_with_name_and_description() -> None:
    skill = (ROOT / "plugin/skills/using-memory/SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\n")
    assert "name: using-memory" in skill
    assert "description: Use when" in skill


def test_marketplace_json_points_at_plugin() -> None:
    plugins = read_json(".claude-plugin/marketplace.json")["plugins"]
    assert len(plugins) == 1
    assert plugins[0]["name"] == "agent-memory"
    assert plugins[0]["source"] == "./plugin"


def test_plugin_is_shipped_in_the_sdist() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"plugin"' in pyproject
