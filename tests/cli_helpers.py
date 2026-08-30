from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

FAKE_DISTILL = Path(__file__).parent / "mock" / "fake_distill.py"


@dataclass(frozen=True)
class CliRun:
    stdout: str
    stderr: str
    code: int


def run_cli(args: list[str], env: dict[str, str], stdin: str = "") -> CliRun:
    proc = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=60,
    )
    return CliRun(stdout=proc.stdout, stderr=proc.stderr, code=proc.returncode)


CliRunner = Callable[..., CliRun]
"""A pre-configured runner fixture: ``run(args, env=None)`` for the CLI,
``run(event, stdin, env=None)`` for hooks."""
