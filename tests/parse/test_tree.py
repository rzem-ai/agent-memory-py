import pytest

from agent_memory import (
    ParsedTreeNode,
    ParseError,
    TreeListEntry,
    TreeSearchHit,
    TreeWindow,
    parse_tree_list,
    parse_tree_node,
    parse_tree_search,
)

from .helpers import fixture


def test_tree_list_parses_roots_with_and_without_last() -> None:
    parsed = parse_tree_list(fixture("tree-list-roots.txt"))
    assert parsed.scope == "roots"
    assert parsed.entries == [
        TreeListEntry(
            path="mail/2026",
            state="open",
            window=TreeWindow(from_="2026-01-01", to="2026-08-15"),
            doc_count=412,
            last_appended_date="2026-08-15",
        ),
        TreeListEntry(
            path="github/2026",
            state="summarised",
            window=TreeWindow(from_="2026-01-02", to="2026-08-10"),
            doc_count=178,
            last_appended_date=None,
        ),
    ]


def test_tree_list_empty_sentinel_yields_scope_and_no_entries() -> None:
    parsed = parse_tree_list(fixture("tree-list-empty.txt"))
    assert parsed.scope == "mail/2031"
    assert parsed.entries == []


def test_tree_list_malformed_raises() -> None:
    with pytest.raises(ParseError):
        parse_tree_list("nope")


def test_tree_node_summarised_with_multi_paragraph_summary() -> None:
    parsed = parse_tree_node(fixture("tree-node-summarised.txt"))
    assert parsed == ParsedTreeNode(
        path="mail/2026/07",
        state="summarised",
        window=TreeWindow(from_="2026-07-01", to="2026-07-31"),
        doc_count=58,
        last_appended_at="2026-07-31T22:14:09.000Z",
        summary_md="## July mail\n\n- Kickoff thread with the platform team.\n\nSecond paragraph with detail.",
    )


def test_tree_node_open_yields_no_summary_and_no_last_appended() -> None:
    parsed = parse_tree_node(fixture("tree-node-open.txt"))
    assert parsed == ParsedTreeNode(
        path="mail/2026/08",
        state="open",
        window=TreeWindow(from_="2026-08-01", to="2026-08-15"),
        doc_count=23,
        last_appended_at=None,
        summary_md=None,
    )


def test_tree_search_parses_hits_with_and_without_excerpt() -> None:
    parsed = parse_tree_search(fixture("tree-search.txt"))
    assert parsed.query == "kickoff planning"
    assert parsed.results == [
        TreeSearchHit(
            path="mail/2026/07",
            state="summarised",
            rank=0.912,
            similarity=0.801,
            window=TreeWindow(from_="2026-07-01", to="2026-07-31"),
            excerpt="July mail: kickoff thread with the platform team",
        ),
        TreeSearchHit(
            path="github/2026/06",
            state="summarised",
            rank=0.454,
            similarity=0.454,
            window=TreeWindow(from_="2026-06-01", to="2026-06-30"),
            excerpt=None,
        ),
    ]


def test_tree_search_empty_sentinel() -> None:
    parsed = parse_tree_search(fixture("tree-search-empty.txt"))
    assert parsed.results == []
    assert parsed.query is None
