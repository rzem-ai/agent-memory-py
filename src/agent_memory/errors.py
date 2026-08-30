"""Error taxonomy, matching how failures actually arrive from the server:
transport-level (network, timeout, HTTP 401) vs tool results flagged
``isError: true`` (classified by their frozen text prefixes) vs frozen-text
drift the parsers could not follow."""

from __future__ import annotations

from typing import Literal

from .types import DOCUMENTS_UNAVAILABLE_NOTE


class AgentMemoryError(Exception):
    """Base class for every error this package raises."""


class TransportError(AgentMemoryError):
    """Network / spawn / timeout / protocol failure below the tool layer."""


class AuthError(TransportError):
    """HTTP 401 — missing or rejected credential."""


ToolErrorKind = Literal["scope_denied", "validation", "no_namespace", "degraded", "failed"]


def classify_tool_error_text(text: str) -> ToolErrorKind:
    """Classify an ``isError: true`` tool-result text by its frozen prefix."""
    if text.startswith("Insufficient scope:"):
        return "scope_denied"
    if text.startswith("Error:"):
        return "validation"
    if text.startswith("This credential has no concrete namespace"):
        return "no_namespace"
    if text == DOCUMENTS_UNAVAILABLE_NOTE:
        return "degraded"
    return "failed"


class ToolError(AgentMemoryError):
    """A tool result the server flagged ``isError: true``."""

    tool: str
    text: str
    kind: ToolErrorKind

    def __init__(self, tool: str, text: str) -> None:
        super().__init__(f"{tool}: {text}")
        self.tool = tool
        self.text = text
        self.kind = classify_tool_error_text(text)


class ParseError(AgentMemoryError):
    """The server's frozen text no longer matches what this client can parse."""

    text: str
    """The offending text, in full, for diagnosis."""

    def __init__(self, message: str, text: str) -> None:
        super().__init__(message)
        self.text = text
