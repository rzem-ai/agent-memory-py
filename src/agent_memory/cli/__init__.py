"""agent-memory-py — CLI over the typed client. One subcommand per tool
(tree/kv grouped), plus ``tools``, ``health``, and the plugin-facing ``hook``
subcommands. Default output is the server's frozen text verbatim; --json
emits one machine envelope on stdout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import anyio
import httpx2

from ..client import AgentMemory, MergedSearchResponse
from ..connect import DEFAULT_URL, ConnectOptions, HttpConnectOptions, StdioConnectOptions, resolve_url
from ..errors import TransportError
from ..types import Corpus, RelevanceMode
from .hook import HookFlags, run_hook
from .output import PROG, OutputFlags, UsageError, emit_error, emit_ok, to_jsonable

USAGE = f"""Usage: {PROG} <command> [options]

Commands:
  search <query>            [--corpus thoughts|documents|all] [--limit N] [--mode M] [--value V]
  capture <content>         [--tag t]... | [--tags a,b]
  forget <thought-uuid>
  read-document <doc-id>    [--max-chars N]
  tree list [path] | tree read <path> | tree search <query> [--limit N]
  kv get <key> | kv set <key> <value> [--raw-string] | kv delete <key> | kv list
  tools                     list the server's tool names
  health                    check the server's public /health route
  hook user-prompt          Claude Code UserPromptSubmit hook (reads stdin JSON)
  hook stop                 Claude Code Stop hook (reads stdin JSON)

Connection: --url <mcp-url> --token <bearer> | --stdio --server <stdio.js> [--config <mcp.toml>]
            env: AGENT_MEMORY_URL, AGENT_MEMORY_TOKEN, AGENT_MEMORY_TIMEOUT_MS
Output:     --json --quiet --fail-open --timeout-ms N"""


class _Parser(argparse.ArgumentParser):
    """argparse that raises UsageError instead of exiting, so the envelope and
    exit-code policy in output.py stay the single source of truth."""

    def error(self, message: str) -> NoReturn:
        raise UsageError(f"{message}. Run with --help for usage.")


def _positive(name: str) -> Callable[[str], float]:
    def convert(raw: str) -> float:
        try:
            n = float(raw)
        except ValueError:
            n = float("nan")
        if not n > 0:
            raise UsageError(f"--{name} must be a positive number, got '{raw}'")
        return n

    return convert


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url")
    p.add_argument("--token")
    p.add_argument("--stdio", action="store_true")
    p.add_argument("--server")
    p.add_argument("--config")
    p.add_argument("--timeout-ms", type=_positive("timeout-ms"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-open", action="store_true")
    p.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog=PROG, add_help=False, usage=USAGE)
    commands = parser.add_subparsers(dest="command")

    def sub(name: str, parent: Any = commands) -> argparse.ArgumentParser:
        p = cast(argparse.ArgumentParser, parent.add_parser(name, add_help=False, usage=USAGE))
        _add_common(p)
        return p

    p = sub("search")
    p.add_argument("query")
    p.add_argument("--corpus", choices=["thoughts", "documents", "all"])
    p.add_argument("--limit", type=_positive("limit"))
    p.add_argument("--mode", choices=["recency_weighted", "similarity", "recent", "since"])
    p.add_argument("--value", type=_positive("value"))

    p = sub("capture")
    p.add_argument("content")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--tags")

    p = sub("forget")
    p.add_argument("thought_id")

    p = sub("read-document")
    p.add_argument("document_id")
    p.add_argument("--max-chars", type=_positive("max-chars"))

    tree = commands.add_parser("tree", add_help=False, usage=USAGE)
    tree_ops = tree.add_subparsers(dest="op", required=True)
    p = sub("list", tree_ops)
    p.add_argument("path", nargs="?")
    p = sub("read", tree_ops)
    p.add_argument("path")
    p = sub("search", tree_ops)
    p.add_argument("query")
    p.add_argument("--limit", type=_positive("limit"))

    kv = commands.add_parser("kv", add_help=False, usage=USAGE)
    kv_ops = kv.add_subparsers(dest="op", required=True)
    p = sub("get", kv_ops)
    p.add_argument("key")
    p = sub("set", kv_ops)
    p.add_argument("key")
    p.add_argument("value")
    p.add_argument("--raw-string", action="store_true")
    p = sub("delete", kv_ops)
    p.add_argument("key")
    sub("list", kv_ops)

    sub("tools")
    sub("health")

    p = sub("hook")
    p.add_argument("event", nargs="?")
    p.add_argument("--limit", type=_positive("limit"))
    p.add_argument("--corpus", choices=["thoughts", "documents", "all"])
    p.add_argument("--min-chars", type=_positive("min-chars"))
    p.add_argument("--max-facts", type=_positive("max-facts"))
    p.add_argument("--model")
    return parser


def _connect_options(ns: argparse.Namespace) -> ConnectOptions:
    timeout = ns.timeout_ms / 1000 if ns.timeout_ms is not None else None
    if ns.stdio:
        if not ns.server:
            raise UsageError("Missing --server <path to stdio.js> for --stdio. Run with --help for usage.")
        return StdioConnectOptions(server_path=ns.server, config_path=ns.config, timeout=timeout)
    return HttpConnectOptions(url=ns.url, token=ns.token, timeout=timeout)


@dataclass(frozen=True, slots=True)
class CommandResult:
    tool: str
    data: Any
    text: str | None = None


async def _with_client(connect: ConnectOptions, fn: Callable[[AgentMemory], Awaitable[CommandResult]]) -> CommandResult:
    async with await AgentMemory.connect(connect) as mem:
        return await fn(mem)


def _capture_tags(ns: argparse.Namespace) -> list[str]:
    tags = list(ns.tag)
    if ns.tags:
        tags.extend(t.strip() for t in ns.tags.split(",") if t.strip())
    return tags


def _kv_value(ns: argparse.Namespace) -> Any:
    if ns.raw_string:
        return ns.value
    try:
        return json.loads(ns.value)
    except ValueError:
        return ns.value


async def _run_typed(
    connect: ConnectOptions,
    out: OutputFlags,
    tool: str,
    args: dict[str, Any],
    typed: Callable[[AgentMemory], Awaitable[Any]],
) -> CommandResult:
    """Commands whose default output is the frozen text: fetch it via raw() when
    not in --json mode, the typed method otherwise."""

    async def run(mem: AgentMemory) -> CommandResult:
        if out.json:
            return CommandResult(tool, to_jsonable(await typed(mem)))
        reply = await mem.raw(tool, args)
        return CommandResult(tool, reply.structured, reply.text)

    return await _with_client(connect, run)


async def _run_search(connect: ConnectOptions, ns: argparse.Namespace) -> CommandResult:
    async def run(mem: AgentMemory) -> CommandResult:
        res = await mem.search(
            ns.query,
            corpus=cast(Corpus, ns.corpus or "all"),
            limit=int(ns.limit) if ns.limit is not None else None,
            relevance_mode=cast(RelevanceMode | None, ns.mode),
            relevance_value=ns.value,
        )
        data = (
            {"corpus": res.corpus, "degraded": res.degraded, "hits": to_jsonable(res.hits)}
            if isinstance(res, MergedSearchResponse)
            else {"corpus": res.corpus, "mode": res.mode, "results": res.results}
        )
        return CommandResult("memory_search", data, res.text)

    return await _with_client(connect, run)


async def _run_health(ns: argparse.Namespace) -> CommandResult:
    base = resolve_url(HttpConnectOptions(url=ns.url)) if ns.url else resolve_url(HttpConnectOptions())
    health_url = re.sub(r"/mcp/?$", "/health", base or DEFAULT_URL)
    try:
        async with httpx2.AsyncClient(timeout=5.0) as http:
            response = await http.get(health_url)
            body = response.json()
    except Exception as err:
        raise TransportError(f"health check failed: {err}") from err
    return CommandResult("health", body, json.dumps(body))


async def _dispatch(ns: argparse.Namespace, out: OutputFlags) -> CommandResult:
    connect = _connect_options(ns)
    args: dict[str, Any]
    match ns.command:
        case "search":
            return await _run_search(connect, ns)
        case "capture":
            tags = _capture_tags(ns)
            return await _run_typed(
                connect,
                out,
                "memory_capture",
                {"content": ns.content, "tags": tags},
                lambda m: m.capture(ns.content, tags),
            )
        case "forget":
            return await _run_typed(
                connect,
                out,
                "memory_forget",
                {"thought_id": ns.thought_id},
                lambda m: _forgotten(m, ns.thought_id),
            )
        case "read-document":
            max_chars = int(ns.max_chars) if ns.max_chars is not None else None
            args = {"document_id": ns.document_id}
            if max_chars is not None:
                args["max_chars"] = max_chars
            return await _run_typed(
                connect,
                out,
                "memory_read_document",
                args,
                lambda m: m.read_document(ns.document_id, max_chars=max_chars),
            )
        case "tree":
            return await _dispatch_tree(connect, out, ns)
        case "kv":
            return await _dispatch_kv(connect, out, ns)
        case "tools":

            async def tools(mem: AgentMemory) -> CommandResult:
                names = await mem.list_tools()
                return CommandResult("tools", names, "\n".join(names))

            return await _with_client(connect, tools)
        case "health":
            return await _run_health(ns)
        case _:
            raise UsageError(USAGE)


async def _forgotten(mem: AgentMemory, thought_id: str) -> dict[str, Any]:
    return {"thought_id": thought_id, "forgotten": await mem.forget(thought_id)}


async def _dispatch_tree(connect: ConnectOptions, out: OutputFlags, ns: argparse.Namespace) -> CommandResult:
    match ns.op:
        case "list":
            args: dict[str, Any] = {"op": "list"}
            if ns.path is not None:
                args["path"] = ns.path
            return await _run_typed(connect, out, "memory_tree", args, lambda m: m.tree.list(ns.path))
        case "read":
            return await _run_typed(
                connect, out, "memory_tree", {"op": "read", "path": ns.path}, lambda m: m.tree.read(ns.path)
            )
        case _:
            limit = int(ns.limit) if ns.limit is not None else None
            args = {"op": "search", "query": ns.query}
            if limit is not None:
                args["limit"] = limit
            return await _run_typed(connect, out, "memory_tree", args, lambda m: m.tree.search(ns.query, limit=limit))


async def _dispatch_kv(connect: ConnectOptions, out: OutputFlags, ns: argparse.Namespace) -> CommandResult:
    match ns.op:
        case "get":

            async def get(mem: AgentMemory) -> CommandResult:
                reply = await mem.raw("memory_kv_get", {"key": ns.key})
                return CommandResult("memory_kv_get", reply.structured, reply.text)

            return await _with_client(connect, get)
        case "set":
            value = _kv_value(ns)

            async def set_(mem: AgentMemory) -> dict[str, Any]:
                await mem.kv.set(ns.key, value)
                return {"key": ns.key, "set": True}

            return await _run_typed(connect, out, "memory_kv_set", {"key": ns.key, "value": value}, set_)
        case "delete":

            async def delete(mem: AgentMemory) -> dict[str, Any]:
                return {"key": ns.key, "deleted": await mem.kv.delete(ns.key)}

            return await _run_typed(connect, out, "memory_kv_delete", {"key": ns.key}, delete)
        case _:
            return await _run_typed(connect, out, "memory_kv_list", {}, lambda m: m.kv.list())


def run(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        sys.stdout.write(USAGE + "\n")
        return 0
    if not args:
        sys.stderr.write(USAGE + "\n")
        return 2
    try:
        ns = build_parser().parse_args(args)
    except UsageError as err:
        return emit_error(
            OutputFlags(json="--json" in args, quiet="--quiet" in args, fail_open="--fail-open" in args), err
        )

    if ns.command == "hook":
        # Hooks own their stdout/stderr and never fail the caller — the envelope
        # machinery below must not print anything around them.
        try:
            connect = _connect_options(ns)
            flags = HookFlags(
                limit=int(ns.limit) if ns.limit is not None else None,
                corpus=cast(Corpus | None, ns.corpus),
                timeout=ns.timeout_ms / 1000 if ns.timeout_ms is not None else None,
                min_chars=int(ns.min_chars) if ns.min_chars is not None else None,
                max_facts=int(ns.max_facts) if ns.max_facts is not None else None,
                model=ns.model,
            )
            anyio.run(run_hook, ns.event, connect, flags)
        except Exception:
            pass
        return 0

    out = OutputFlags(json=ns.json, fail_open=ns.fail_open, quiet=ns.quiet)
    try:
        result = anyio.run(_dispatch, ns, out)
    except Exception as err:
        return emit_error(out, err)
    emit_ok(out, result.tool, result.data, result.text)
    return 0


def main() -> None:
    sys.exit(run())
