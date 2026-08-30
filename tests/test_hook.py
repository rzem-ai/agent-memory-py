import json
import sys
from pathlib import Path

import pytest

from .cli_helpers import FAKE_DISTILL, CliRun, CliRunner, run_cli
from .conftest import TOKEN
from .mock.http import HttpHarness

TRANSCRIPT_LINES = [
    json.dumps({"type": "user", "message": {"role": "user", "content": "How should I package the memory client?"}}),
    json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Ship it as agent-memory-py with a bundled Claude Code plugin."}],
            },
        }
    ),
]


@pytest.fixture(scope="module")
def transcript_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("agent-memory-hook") / "transcript.jsonl"
    path.write_text("\n".join(TRANSCRIPT_LINES) + "\n")
    return path


@pytest.fixture
def hook(harness: HttpHarness) -> CliRunner:
    def run(event: str, stdin: str, env: dict[str, str] | None = None) -> CliRun:
        return run_cli(
            ["hook", event],
            {
                "AGENT_MEMORY_URL": harness.mcp_url,
                "AGENT_MEMORY_TOKEN": TOKEN,
                "AGENT_MEMORY_DISTILL_CMD": f"{sys.executable} {FAKE_DISTILL}",
                **(env or {}),
            },
            stdin=stdin,
        )

    return run


def test_user_prompt_injects_additional_context_from_search_hits(hook: CliRunner) -> None:
    run = hook("user-prompt", json.dumps({"prompt": "what package manager does alex prefer?"}))
    assert run.code == 0
    body = json.loads(run.stdout)
    out = body["hookSpecificOutput"]
    assert out["hookEventName"] == "UserPromptSubmit"
    assert "Relevant long-term memory" in out["additionalContext"]
    assert "pnpm" in out["additionalContext"]


def test_user_prompt_stays_silent_on_a_short_prompt(hook: CliRunner) -> None:
    run = hook("user-prompt", json.dumps({"prompt": "hi"}))
    assert run.code == 0
    assert run.stdout == ""


def test_user_prompt_stays_silent_on_a_slash_command(hook: CliRunner) -> None:
    run = hook("user-prompt", json.dumps({"prompt": "/compact please and thank you"}))
    assert run.code == 0
    assert run.stdout == ""


def test_user_prompt_stays_silent_when_server_is_down(hook: CliRunner) -> None:
    run = hook(
        "user-prompt",
        json.dumps({"prompt": "what package manager does alex prefer?"}),
        {"AGENT_MEMORY_URL": "http://127.0.0.1:9/mcp"},
    )
    assert run.code == 0
    assert run.stdout == ""


def test_user_prompt_stays_silent_on_unparseable_stdin(hook: CliRunner) -> None:
    run = hook("user-prompt", "not json at all")
    assert run.code == 0
    assert run.stdout == ""


def test_unknown_hook_event_exits_0(hook: CliRunner) -> None:
    run = hook("bogus", "{}")
    assert run.code == 0
    assert run.stdout == ""


def test_stop_distils_transcript_tail_and_captures_facts(hook: CliRunner, transcript_path: Path) -> None:
    run = hook(
        "stop", json.dumps({"transcript_path": str(transcript_path), "cwd": "/Users/alex/Dev/Work/mcp/agent-memory-py"})
    )
    assert run.code == 0
    assert run.stdout == ""
    assert "captured 2 fact" in run.stderr


def test_stop_bails_out_when_stop_hook_active(hook: CliRunner, transcript_path: Path) -> None:
    run = hook("stop", json.dumps({"transcript_path": str(transcript_path), "stop_hook_active": True}))
    assert run.code == 0
    assert run.stderr == ""


def test_stop_captures_nothing_when_distiller_says_none(hook: CliRunner, transcript_path: Path) -> None:
    run = hook("stop", json.dumps({"transcript_path": str(transcript_path)}), {"FAKE_DISTILL_OUTPUT": "NONE"})
    assert run.code == 0
    assert "captured" not in run.stderr


def test_stop_stays_silent_when_transcript_path_is_missing(hook: CliRunner) -> None:
    run = hook("stop", json.dumps({}))
    assert run.code == 0
    assert run.stderr == ""


def test_stop_exits_0_even_when_server_is_down(hook: CliRunner, transcript_path: Path) -> None:
    run = hook(
        "stop", json.dumps({"transcript_path": str(transcript_path)}), {"AGENT_MEMORY_URL": "http://127.0.0.1:9/mcp"}
    )
    assert run.code == 0
