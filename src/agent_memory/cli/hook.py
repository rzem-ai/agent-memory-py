"""Claude Code hook subcommands. The contract is strict: NEVER exit non-zero
(exit 2 on UserPromptSubmit blocks and erases the user's prompt) and never
print anything except the intended payload — bare stdout on UserPromptSubmit
is injected as context verbatim. Every failure path is silence.

- ``hook user-prompt``: read the hook stdin JSON, search memory with the
  prompt, print {hookSpecificOutput:{additionalContext}} on hits.
- ``hook stop``: read the transcript tail, distil capture-worthy facts via a
  one-shot ``claude -p`` (overridable with AGENT_MEMORY_DISTILL_CMD), store
  them via memory_capture. Progress notes go to stderr only."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio

from ..client import AgentMemory, SearchResponse, ThoughtsSearchResponse
from ..connect import ConnectOptions
from ..types import Corpus

USER_PROMPT_DEADLINE = 8.0
STOP_DEADLINE = 30.0
DEFAULT_LIMIT = 4
DEFAULT_MIN_CHARS = 12
DEFAULT_MAX_FACTS = 3
DEFAULT_DISTILL_MODEL = "claude-haiku-4-5-20251001"
LINE_MAX_CHARS = 300
TRANSCRIPT_SIDE_MAX_CHARS = 4_000

PROG = "agent-memory-py"


@dataclass(frozen=True, slots=True, kw_only=True)
class HookFlags:
    limit: int | None = None
    corpus: Corpus | None = None
    timeout: float | None = None
    min_chars: int | None = None
    max_facts: int | None = None
    model: str | None = None


async def run_hook(event: str | None, connect: ConnectOptions, flags: HookFlags) -> None:
    try:
        if event == "user-prompt":
            await _user_prompt_hook(connect, flags)
        elif event == "stop":
            await _stop_hook(connect, flags)
        else:
            sys.stderr.write(f"{PROG} hook: unknown event '{event or ''}'\n")
    except Exception:
        pass


def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None


async def _with_deadline[T](connect: ConnectOptions, deadline: float, fn: Callable[[AgentMemory], Awaitable[T]]) -> T:
    """Run fn against a connected client, aborting the whole attempt (connect
    included) at the deadline."""
    with anyio.fail_after(deadline):
        async with await AgentMemory.connect(dataclasses.replace(connect, timeout=deadline)) as mem:
            return await fn(mem)


def _truncate(line: str, max_chars: int = LINE_MAX_CHARS) -> str:
    collapsed = re.sub(r"\s+", " ", line).strip()
    return f"{collapsed[: max_chars - 1]}…" if len(collapsed) > max_chars else collapsed


def _context_lines(res: SearchResponse) -> list[str]:
    if isinstance(res, ThoughtsSearchResponse):
        return [
            _truncate(
                f"- (thought {row['created_at'][:10]}) {row['content']}"
                + (f" [tags: {', '.join(row['tags'])}]" if row["tags"] else "")
            )
            for row in res.results
        ]
    lines: list[str] = []
    for hit in res.hits:
        if hit.corpus == "thoughts":
            tags = f" [tags: {', '.join(hit.tags)}]" if hit.tags else ""
            lines.append(_truncate(f"- (thought {hit.date}) {hit.content}{tags}"))
        else:
            lines.append(_truncate(f"- ({hit.source_kind} doc {hit.date}, id: {hit.document_id}) {hit.snippet}"))
    return lines


async def _user_prompt_hook(connect: ConnectOptions, flags: HookFlags) -> None:
    payload = _parse_json(sys.stdin.read())
    prompt = None if payload is None else payload.get("prompt", payload.get("user_prompt"))
    if not isinstance(prompt, str):
        return
    trimmed = prompt.strip()
    if len(trimmed) < (flags.min_chars or DEFAULT_MIN_CHARS) or trimmed.startswith("/"):
        return
    corpus: Corpus = flags.corpus or "all"
    limit = flags.limit or DEFAULT_LIMIT
    res = await _with_deadline(
        connect,
        flags.timeout or USER_PROMPT_DEADLINE,
        lambda mem: mem.search(trimmed, corpus=corpus, limit=limit),
    )
    lines = _context_lines(res)
    if not lines:
        return
    context = "Relevant long-term memory (agent-memory):\n" + "\n".join(lines)
    output = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}
    sys.stdout.write(json.dumps(output, ensure_ascii=False) + "\n")


def _entry_text(entry: dict[str, Any]) -> str:
    message = entry.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return ""


def _transcript_tail(path: str) -> str | None:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    entries = [e for e in (_parse_json(line) for line in raw.split("\n") if line) if e is not None]

    def last_of_type(kind: str) -> str:
        texts = [t for t in (_entry_text(e) for e in entries if e.get("type") == kind) if t]
        return texts[-1] if texts else ""

    user = last_of_type("user")[-TRANSCRIPT_SIDE_MAX_CHARS:]
    assistant = last_of_type("assistant")[-TRANSCRIPT_SIDE_MAX_CHARS:]
    if not user and not assistant:
        return None
    return f"User:\n{user}\n\nAssistant:\n{assistant}"


def _distill(excerpt: str, max_facts: int, model: str, timeout: float) -> list[str]:
    override = os.environ.get("AGENT_MEMORY_DISTILL_CMD")
    instruction = (
        f"Extract up to {max_facts} durable facts worth remembering long-term from this conversation excerpt: "
        "stable preferences, decisions, and project facts - never transient task detail. "
        "Output ONLY a JSON array of short self-contained strings, or NONE if nothing is worth keeping."
    )
    cmd = shlex.split(override) if override else ["claude", "-p", instruction, "--model", model]
    try:
        proc = subprocess.run(cmd, input=excerpt, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    stdout = proc.stdout
    start, end = stdout.find("["), stdout.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(stdout[start : end + 1])
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    return [fact for fact in parsed if isinstance(fact, str) and fact.strip()]


async def _stop_hook(connect: ConnectOptions, flags: HookFlags) -> None:
    payload = _parse_json(sys.stdin.read())
    if payload is None or payload.get("stop_hook_active") is True:
        return
    path = payload.get("transcript_path")
    if not isinstance(path, str):
        return
    excerpt = _transcript_tail(path)
    if not excerpt:
        return
    deadline = flags.timeout or STOP_DEADLINE
    max_facts = flags.max_facts or DEFAULT_MAX_FACTS
    facts = _distill(excerpt, max_facts, flags.model or DEFAULT_DISTILL_MODEL, deadline)
    if not facts:
        return
    cwd = payload.get("cwd")
    project = Path(cwd if isinstance(cwd, str) else os.getcwd()).name
    tags = ["claude-code", f"project:{project}", "auto-captured"]

    async def capture_all(mem: AgentMemory) -> int:
        count = 0
        for fact in facts[:max_facts]:
            if (await mem.capture(fact, tags)).captured:
                count += 1
        return count

    captured = await _with_deadline(connect, deadline, capture_all)
    sys.stderr.write(f"{PROG} hook stop: captured {captured} fact(s)\n")
