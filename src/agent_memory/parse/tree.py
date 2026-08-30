"""Inverses of the server's ``formatTreeList`` / ``formatTreeNode`` /
``formatTreeSearch`` (src/domain/recall.ts) — the frozen text contract for
memory_tree. List rows join with single newlines; search records with blank
lines; a node read keeps ``last appended:`` at full ISO precision."""

from __future__ import annotations

import re

from ..errors import ParseError
from ..types import ParsedTreeList, ParsedTreeNode, ParsedTreeSearch, TreeListEntry, TreeSearchHit
from .shared import parse_int, parse_number, parse_tree_state, parse_window, split_records

_LIST_EMPTY_RE = re.compile(r"^No tree nodes under (.+)\.$")
_LIST_HEADER_RE = re.compile(r"^tree under (.+) \(\d+\):$")
_LIST_ENTRY_RE = re.compile(r"^- (.+) \| (\S+) \| window: (\S+) \| docs: (\d+)(?: \| last: (\S+))?$")


def parse_tree_list(text: str) -> ParsedTreeList:
    empty = _LIST_EMPTY_RE.match(text)
    if empty is not None:
        return ParsedTreeList(scope=empty.group(1), entries=[])
    lines = text.split("\n")
    header = _LIST_HEADER_RE.match(lines[0])
    if header is None:
        raise ParseError("Tree-list text did not match the frozen format", text)
    entries: list[TreeListEntry] = []
    for line in lines[1:]:
        m = _LIST_ENTRY_RE.match(line)
        if m is None:
            raise ParseError(f"Tree-list entry did not match the frozen format: '{line}'", text)
        entries.append(
            TreeListEntry(
                path=m.group(1),
                state=parse_tree_state(m.group(2), text),
                window=parse_window(m.group(3), text),
                doc_count=parse_int(m.group(4), text),
                last_appended_date=m.group(5),
            )
        )
    return ParsedTreeList(scope=header.group(1), entries=entries)


_NODE_NO_SUMMARY_RE = re.compile(r"^\(No summary yet - this node is still \S+\.\)$")


def parse_tree_node(text: str) -> ParsedTreeNode:
    meta_end = text.find("\n\n")
    if meta_end == -1:
        raise ParseError("Tree-node text has no metadata/summary separator", text)
    meta: dict[str, str] = {}
    for line in text[:meta_end].split("\n"):
        sep = line.find(": ")
        if sep == -1:
            raise ParseError(f"Tree-node metadata line did not match the frozen format: '{line}'", text)
        meta[line[:sep]] = line[sep + 2 :]

    def required(key: str) -> str:
        value = meta.get(key)
        if value is None:
            raise ParseError(f"Tree-node metadata is missing '{key}'", text)
        return value

    summary = text[meta_end + 2 :]
    return ParsedTreeNode(
        path=required("path"),
        state=parse_tree_state(required("state"), text),
        window=parse_window(required("window"), text),
        doc_count=parse_int(required("docs"), text),
        last_appended_at=meta.get("last appended"),
        summary_md=None if _NODE_NO_SUMMARY_RE.match(summary) else summary,
    )


_SEARCH_EMPTY_SENTINEL = "No matching tree nodes found."
_SEARCH_HEADER_RE = re.compile(r"^tree search: (.+)$")
_SEARCH_HIT_RE = re.compile(r"^\[\d+\] (.+) \| (\S+) \| rank: ([0-9.]+) \| sim: ([0-9.]+) \| window: (\S+)$")
_EXCERPT_INDENT = "\n    "


def parse_tree_search(text: str) -> ParsedTreeSearch:
    if text == _SEARCH_EMPTY_SENTINEL:
        return ParsedTreeSearch(query=None, results=[])
    header_end = text.find("\n\n")
    if header_end == -1:
        raise ParseError("Tree-search text has no header/body separator", text)
    header = _SEARCH_HEADER_RE.match(text[:header_end])
    if header is None:
        raise ParseError("Tree-search header did not match the frozen format", text)
    results: list[TreeSearchHit] = []
    for record in split_records(text[header_end + 2 :]):
        line_end = record.find(_EXCERPT_INDENT)
        head = _SEARCH_HIT_RE.match(record if line_end == -1 else record[:line_end])
        if head is None:
            raise ParseError(f"Tree-search hit did not match the frozen format: '{record}'", text)
        results.append(
            TreeSearchHit(
                path=head.group(1),
                state=parse_tree_state(head.group(2), text),
                rank=parse_number(head.group(3), text),
                similarity=parse_number(head.group(4), text),
                window=parse_window(head.group(5), text),
                excerpt=None if line_end == -1 else record[line_end + len(_EXCERPT_INDENT) :],
            )
        )
    return ParsedTreeSearch(query=header.group(1), results=results)
