"""CLI output + exit-code policy. Machine consumers get one JSON envelope on
stdout; humans get the server's frozen text verbatim. --fail-open forces
exit 0 whatever happened (the hook contract: a down memory server must
never break the caller), while still printing the error envelope unless
--quiet."""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass
from typing import Any, Literal

from ..errors import AuthError, ParseError, ToolError, TransportError

ErrorKind = Literal["auth", "transport", "tool", "parse", "usage"]

EXIT_CODES: dict[ErrorKind, int] = {"tool": 1, "parse": 1, "usage": 2, "transport": 3, "auth": 4}

PROG = "agent-memory-py"


class UsageError(Exception):
    """A bad invocation (unknown command, missing argument)."""


@dataclass(frozen=True, slots=True)
class OutputFlags:
    json: bool = False
    fail_open: bool = False
    quiet: bool = False


def to_jsonable(value: Any) -> Any:
    """Dataclasses become dicts (``from_`` -> ``from`` to match the JS envelope)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            ("from" if f.name == "from_" else f.name): to_jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    return value


def _dump(payload: Any) -> str:
    return json.dumps(to_jsonable(payload), ensure_ascii=False)


def emit_ok(flags: OutputFlags, tool: str, data: Any, text: str | None = None) -> None:
    if flags.quiet:
        return
    if flags.json:
        envelope: dict[str, Any] = {"ok": True, "tool": tool, "data": data}
        if text is not None:
            envelope["text"] = text
        sys.stdout.write(_dump(envelope) + "\n")
        return
    if text is not None:
        sys.stdout.write(text + "\n")
        return
    sys.stdout.write(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False) + "\n")


@dataclass(frozen=True, slots=True)
class ClassifiedError:
    kind: ErrorKind
    message: str
    detail: dict[str, Any]


def classify_error(err: BaseException) -> ClassifiedError:
    if isinstance(err, AuthError):
        return ClassifiedError("auth", str(err), {})
    if isinstance(err, TransportError):
        return ClassifiedError("transport", str(err), {})
    if isinstance(err, ToolError):
        return ClassifiedError("tool", err.text, {"tool": err.tool, "toolKind": err.kind})
    if isinstance(err, ParseError):
        return ClassifiedError("parse", str(err), {"text": err.text})
    if isinstance(err, UsageError):
        return ClassifiedError("usage", str(err), {})
    return ClassifiedError("transport", str(err) or type(err).__name__, {})


def emit_error(flags: OutputFlags, err: BaseException) -> int:
    """Print the failure and return the process exit code to use."""
    classified = classify_error(err)
    if not flags.quiet:
        if flags.json:
            error = {"kind": classified.kind, "message": classified.message, **classified.detail}
            sys.stdout.write(_dump({"ok": False, "error": error}) + "\n")
        else:
            sys.stderr.write(f"{PROG}: {classified.kind}: {classified.message}\n")
    return 0 if flags.fail_open else EXIT_CODES[classified.kind]
