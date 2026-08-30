"""Inverse of the server's ``formatMergedResults`` (src/domain/recall.ts) — the
frozen text contract for memory_search over corpus 'documents' and 'all'.

Known, inherent lossiness of that contract: dates are day-precision, a
document's ``<title> - <excerpt>`` join is not reliably splittable, and thought
content containing `` | tags: `` or a blank line followed by ``[i] `` mis-splits.
"""

from __future__ import annotations

import re
from typing import Literal

from ..errors import ParseError
from ..types import (
    DOCUMENT_TAINTS,
    DOCUMENTS_UNAVAILABLE_NOTE,
    DocumentHit,
    DocumentTaint,
    MergedHit,
    ParsedMergedResults,
    ThoughtHit,
)
from .shared import parse_number, split_records

EMPTY_SENTINEL = "No matching memories found."
_HEAD_RE = re.compile(
    r"^\[\d+\] corpus: (thoughts|documents) \| taint: ([a-z]+) \| rank: ([0-9.]+) \| sim: ([0-9.]+)"
    r" \| ([^|]+?) \| (.+)$"
)
_THOUGHT_TAIL_RE = re.compile(r"^agent: (.+) \| id: (\S+)$")
_DOCUMENT_TAIL_RE = re.compile(r"^source: (\S+) \| doc: (\S+) \| path: (.+)$")
_TAGS_SEPARATOR = " | tags: "
_BODY_INDENT = "\n    "


def _parse_taint(raw: str, text: str) -> DocumentTaint:
    if raw not in DOCUMENT_TAINTS:
        raise ParseError(f"Unknown taint '{raw}'", text)
    return raw


def _parse_record(record: str, text: str) -> MergedHit:
    body_sep = record.find(_BODY_INDENT)
    if body_sep == -1:
        raise ParseError("Merged-search record has no indented body line", text)
    head = _HEAD_RE.match(record[:body_sep])
    if head is None:
        raise ParseError("Merged-search record header did not match the frozen format", text)
    corpus, taint, rank, sim, date, tail = head.groups()
    body = record[body_sep + len(_BODY_INDENT) :]
    rank_n = parse_number(rank, text)
    sim_n = parse_number(sim, text)

    if corpus == "thoughts":
        t = _THOUGHT_TAIL_RE.match(tail)
        if t is None:
            raise ParseError("Thought record tail did not match the frozen format", text)
        if taint != "internal":
            raise ParseError(f"Thought record with non-internal taint '{taint}'", text)
        tags_at = body.rfind(_TAGS_SEPARATOR)
        return ThoughtHit(
            rank=rank_n,
            similarity=sim_n,
            date=date,
            agent_id=t.group(1),
            id=t.group(2),
            content=body if tags_at == -1 else body[:tags_at],
            tags=[] if tags_at == -1 else body[tags_at + len(_TAGS_SEPARATOR) :].split(", "),
        )

    d = _DOCUMENT_TAIL_RE.match(tail)
    if d is None:
        raise ParseError("Document record tail did not match the frozen format", text)
    return DocumentHit(
        taint=_parse_taint(taint, text),
        rank=rank_n,
        similarity=sim_n,
        date=date,
        source_kind=d.group(1),
        document_id=d.group(2),
        vault_path=d.group(3),
        snippet=body,
    )


def parse_merged_results(text: str) -> ParsedMergedResults:
    body = text
    degraded = False

    if body == DOCUMENTS_UNAVAILABLE_NOTE:
        return ParsedMergedResults(corpus=None, degraded=True, hits=[], text=text)
    note_suffix = f"\n\n{DOCUMENTS_UNAVAILABLE_NOTE}"
    if body.endswith(note_suffix):
        degraded = True
        body = body[: -len(note_suffix)]
    if body == EMPTY_SENTINEL:
        return ParsedMergedResults(corpus=None, degraded=degraded, hits=[], text=text)

    header_end = body.find("\n\n")
    if header_end == -1:
        raise ParseError("Merged-search text has no header/body separator", text)
    header = body[:header_end]
    corpus: Literal["documents", "all"]
    if header == "corpus: all (thoughts + documents)":
        corpus = "all"
    elif header == "corpus: documents":
        corpus = "documents"
    else:
        raise ParseError(f"Unrecognised merged-search header '{header}'", text)

    hits = [_parse_record(record, text) for record in split_records(body[header_end + 2 :])]
    return ParsedMergedResults(corpus=corpus, degraded=degraded, hits=hits, text=text)
