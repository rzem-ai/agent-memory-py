from __future__ import annotations

import re

from ..errors import ParseError
from ..types import TREE_NODE_STATES, TreeNodeState, TreeWindow

_RECORD_BOUNDARY = re.compile(r"\n\n(?=\[\d+\] )")


def parse_number(raw: str, text: str) -> float:
    try:
        n = float(raw)
    except ValueError:
        raise ParseError(f"Expected a number, got '{raw}'", text) from None
    if n != n or n in (float("inf"), float("-inf")):
        raise ParseError(f"Expected a number, got '{raw}'", text)
    return n


def parse_int(raw: str, text: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ParseError(f"Expected an integer, got '{raw}'", text) from None


def parse_tree_state(raw: str, text: str) -> TreeNodeState:
    if raw not in TREE_NODE_STATES:
        raise ParseError(f"Unknown tree node state '{raw}'", text)
    return raw


def parse_window(raw: str, text: str) -> TreeWindow:
    """Parse the formatter's ``<from>..<to>`` window rendering."""
    sep = raw.find("..")
    if sep == -1:
        raise ParseError(f"Expected a '<from>..<to>' window, got '{raw}'", text)
    return TreeWindow(from_=raw[:sep], to=raw[sep + 2 :])


def split_records(body: str) -> list[str]:
    """Split the ``\\n\\n``-joined records after a header, keeping ``[i] `` boundaries."""
    return _RECORD_BOUNDARY.split(body)
