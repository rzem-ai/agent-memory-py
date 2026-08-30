import pytest

from agent_memory import ParsedDocument, ParseError, parse_document

from .helpers import fixture


def test_parses_full_header_and_keeps_colon_lines_and_blank_lines_in_body() -> None:
    parsed = parse_document(fixture("document-full.txt"))
    assert parsed == ParsedDocument(
        id="mail-2026-07-15-0042",
        title="Project kickoff notes",
        source_kind="mail",
        external_id="<CAExample@mail.gmail.com>",
        taint="external",
        score=0.85,
        event_at="2026-07-15T09:30:00.000Z",
        ingested_at="2026-07-15T10:02:11.000Z",
        vault_path="mail/2026/07/15/kickoff.md",
        provenance={"from": "pm@example.com", "thread": "kickoff"},
        body_source="vault",
        truncated=False,
        body="# Kickoff\n\nAgenda:\n\n- item one\n\nevent_at: this line is body content, not a header",
    )


def test_parses_truncation_note_and_none_provenance() -> None:
    parsed = parse_document(fixture("document-truncated.txt"))
    assert parsed.truncated is True
    assert parsed.provenance == {}
    assert parsed.body_source == "chunks"
    assert parsed.body == "This body was cut off at exactly forty-four"


def test_malformed_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_document("not a document")
