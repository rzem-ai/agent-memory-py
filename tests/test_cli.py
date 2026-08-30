import json
from typing import Any

import pytest

from .cli_helpers import CliRun, CliRunner, run_cli
from .conftest import TOKEN
from .mock.http import HttpHarness


@pytest.fixture
def cli(harness: HttpHarness) -> CliRunner:
    def run(args: list[str], env: dict[str, str] | None = None) -> CliRun:
        return run_cli(args, {"AGENT_MEMORY_URL": harness.mcp_url, "AGENT_MEMORY_TOKEN": TOKEN, **(env or {})})

    return run


def envelope(run: CliRun) -> dict[str, Any]:
    body = json.loads(run.stdout)
    assert isinstance(body, dict)
    return body


def test_search_json_returns_ok_envelope_with_typed_hits(cli: CliRunner) -> None:
    run = cli(["search", "pnpm", "--json"])
    assert run.code == 0, run.stderr
    body = envelope(run)
    assert body["ok"] is True
    assert body["tool"] == "memory_search"
    assert len(body["data"]["hits"]) == 2
    assert body["data"]["hits"][1]["document_id"] == "mail-2026-07-15-0042"
    assert "corpus: all" in body["text"]


def test_search_default_output_is_frozen_text_verbatim(cli: CliRunner) -> None:
    run = cli(["search", "pnpm"])
    assert run.code == 0, run.stderr
    assert run.stdout.startswith("corpus: all (thoughts + documents)")


def test_search_thoughts_json_returns_structured_rows(cli: CliRunner) -> None:
    run = cli(["search", "pnpm", "--corpus", "thoughts", "--json"])
    assert run.code == 0, run.stderr
    body = envelope(run)
    assert body["data"]["mode"] == "recency_weighted"
    assert body["data"]["results"][0]["agent_id"] == "angus"


def test_capture_json_returns_structured_outcome(cli: CliRunner) -> None:
    run = cli(["capture", "a fresh thought", "--tag", "cli", "--json"])
    assert run.code == 0, run.stderr
    assert envelope(run)["data"]["captured"] is True


def test_read_document_json_returns_parsed_document(cli: CliRunner) -> None:
    run = cli(["read-document", "mail-2026-07-15-0042", "--json"])
    assert run.code == 0, run.stderr
    assert envelope(run)["data"]["title"] == "Project kickoff notes"


def test_tree_list_default_output_is_frozen_text(cli: CliRunner) -> None:
    run = cli(["tree", "list"])
    assert run.code == 0, run.stderr
    assert run.stdout.startswith("tree under roots")


def test_kv_get_miss_reports_found_false(cli: CliRunner) -> None:
    run = cli(["kv", "get", "absent", "--json"])
    assert run.code == 0, run.stderr
    assert envelope(run)["data"] == {"key": "absent", "found": False}


def test_kv_set_parses_json_values(cli: CliRunner) -> None:
    run = cli(["kv", "set", "cli-key", '{"n": 1}', "--json"])
    assert run.code == 0, run.stderr
    assert envelope(run)["data"] == {"key": "cli-key", "set": True}
    run = cli(["kv", "get", "cli-key", "--json"])
    assert envelope(run)["data"]["value"] == {"n": 1}


def test_tools_lists_the_nine_names(cli: CliRunner) -> None:
    run = cli(["tools"])
    assert run.code == 0, run.stderr
    assert len(run.stdout.strip().split("\n")) == 9


def test_health_hits_the_public_route(cli: CliRunner) -> None:
    run = cli(["health", "--json"])
    assert run.code == 0, run.stderr
    assert envelope(run)["data"] == {"status": "ok"}


def test_wrong_token_exits_4_with_auth_error_envelope(cli: CliRunner) -> None:
    run = cli(["search", "pnpm", "--json"], {"AGENT_MEMORY_TOKEN": "nope"})
    assert run.code == 4
    body = envelope(run)
    assert body["ok"] is False
    assert body["error"]["kind"] == "auth"


def test_scope_denial_exits_1_with_tool_error_envelope(cli: CliRunner) -> None:
    run = cli(["capture", "denied", "--json"])
    assert run.code == 1
    body = envelope(run)
    assert body["error"]["kind"] == "tool"
    assert body["error"]["toolKind"] == "scope_denied"


def test_dead_server_exits_3_with_transport_error_on_stderr(cli: CliRunner) -> None:
    run = cli(["search", "pnpm", "--timeout-ms", "2000"], {"AGENT_MEMORY_URL": "http://127.0.0.1:9/mcp"})
    assert run.code == 3
    assert run.stdout == ""
    assert run.stderr.startswith("agent-memory-py: transport:")


def test_fail_open_forces_exit_0_on_dead_server_and_keeps_envelope(cli: CliRunner) -> None:
    run = cli(
        ["search", "pnpm", "--json", "--fail-open", "--timeout-ms", "2000"],
        {"AGENT_MEMORY_URL": "http://127.0.0.1:9/mcp"},
    )
    assert run.code == 0
    assert envelope(run)["ok"] is False


def test_fail_open_quiet_prints_nothing_on_failure(cli: CliRunner) -> None:
    run = cli(
        ["search", "pnpm", "--fail-open", "--quiet", "--timeout-ms", "2000"],
        {"AGENT_MEMORY_URL": "http://127.0.0.1:9/mcp"},
    )
    assert run.code == 0
    assert run.stdout == ""


def test_unknown_command_exits_2(cli: CliRunner) -> None:
    assert cli(["frobnicate"]).code == 2


def test_missing_argument_exits_2(cli: CliRunner) -> None:
    assert cli(["forget"]).code == 2


def test_help_exits_0_and_prints_usage(cli: CliRunner) -> None:
    run = cli(["--help"])
    assert run.code == 0
    assert "Usage: agent-memory-py" in run.stdout
