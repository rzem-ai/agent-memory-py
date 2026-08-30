"""Inverse of the server's ``formatDocument`` (src/domain/recall.ts) — the
frozen text contract for memory_read_document: eleven fixed-order header lines,
an optional truncation note, a blank line, then the verbatim Markdown body. The
body may itself contain blank lines and ``key: value``-looking lines, so the
split happens after the known header keys — never at the first blank line."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, cast

from ..errors import ParseError
from ..types import DOCUMENT_TAINTS, ParsedDocument
from .shared import parse_number

_HEADER_KEYS = (
    "id",
    "title",
    "source",
    "external_id",
    "taint",
    "score",
    "event_at",
    "ingested_at",
    "vault_path",
    "provenance",
    "body_source",
)
_BODY_SOURCES: tuple[Literal["vault", "chunks"], ...] = ("vault", "chunks")
_TRUNCATION_NOTE_RE = re.compile(r"^note: body truncated to \d+ characters$")


def _parse_provenance(raw: str, text: str) -> dict[str, Any]:
    if raw == "(none)":
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise ParseError(f"Document provenance is neither '(none)' nor JSON: '{raw}'", text) from None
    if not isinstance(parsed, dict):
        raise ParseError(f"Document provenance is not a JSON object: '{raw}'", text)
    return cast(dict[str, Any], parsed)


def parse_document(text: str) -> ParsedDocument:
    lines = text.split("\n")
    fields: dict[str, str] = {}
    for index, key in enumerate(_HEADER_KEYS):
        prefix = f"{key}: "
        if index >= len(lines) or not lines[index].startswith(prefix):
            raise ParseError(f"Document header line {index + 1} is not '{key}: ...'", text)
        fields[key] = lines[index][len(prefix) :]

    nxt = len(_HEADER_KEYS)
    truncated = nxt < len(lines) and _TRUNCATION_NOTE_RE.match(lines[nxt]) is not None
    if truncated:
        nxt += 1
    if nxt >= len(lines) or lines[nxt] != "":
        raise ParseError("Document header is not followed by a blank line", text)

    taint = fields["taint"]
    if taint not in DOCUMENT_TAINTS:
        raise ParseError(f"Unknown document taint '{taint}'", text)
    body_source = fields["body_source"]
    if body_source not in _BODY_SOURCES:
        raise ParseError(f"Unknown body_source '{body_source}'", text)

    return ParsedDocument(
        id=fields["id"],
        title=fields["title"],
        source_kind=fields["source"],
        external_id=fields["external_id"],
        taint=taint,
        score=parse_number(fields["score"], text),
        event_at=fields["event_at"],
        ingested_at=fields["ingested_at"],
        vault_path=fields["vault_path"],
        provenance=_parse_provenance(fields["provenance"], text),
        body_source=body_source,
        truncated=truncated,
        body="\n".join(lines[nxt + 1 :]),
    )
