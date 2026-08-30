import pytest

from agent_memory import DocumentHit, ParseError, ThoughtHit, parse_merged_results

from .helpers import fixture


def test_parses_corpus_all_with_one_thought_and_one_document_hit() -> None:
    parsed = parse_merged_results(fixture("merged-all.txt"))
    assert parsed.corpus == "all"
    assert parsed.degraded is False
    assert len(parsed.hits) == 2

    thought, doc = parsed.hits
    assert thought == ThoughtHit(
        corpus="thoughts",
        taint="internal",
        rank=0.812,
        similarity=0.903,
        date="2026-08-01",
        agent_id="angus",
        id="3f2a8c1e-5b7d-4e9a-8c3d-1a2b3c4d5e6f",
        content="Alex prefers pnpm for the angus2 workspace",
        tags=["claude-code", "project:angus2"],
    )
    assert doc == DocumentHit(
        corpus="documents",
        taint="external",
        rank=0.641,
        similarity=0.788,
        date="2026-07-15",
        source_kind="mail",
        document_id="mail-2026-07-15-0042",
        vault_path="mail/2026/07/15/kickoff.md",
        snippet="Project kickoff notes - Agenda and decisions from the kickoff call",
    )


def test_parses_corpus_documents_header() -> None:
    parsed = parse_merged_results(fixture("merged-documents.txt"))
    assert parsed.corpus == "documents"
    assert len(parsed.hits) == 1
    hit = parsed.hits[0]
    assert isinstance(hit, DocumentHit)
    assert hit.source_kind == "github"
    assert hit.document_id == "gh-123"


def test_thought_without_tags_suffix_yields_empty_tags() -> None:
    parsed = parse_merged_results(fixture("merged-no-tags-no-excerpt.txt"))
    thought, doc = parsed.hits
    assert isinstance(thought, ThoughtHit)
    assert thought.content == "A thought captured with no tags at all"
    assert thought.tags == []
    # Document with no excerpt: snippet is the bare title.
    assert isinstance(doc, DocumentHit)
    assert doc.snippet == "Untitled drop"


def test_empty_sentinel_yields_zero_hits() -> None:
    parsed = parse_merged_results(fixture("merged-empty.txt"))
    assert parsed.hits == []
    assert parsed.degraded is False
    assert parsed.corpus is None


def test_degraded_note_is_detected_and_stripped() -> None:
    parsed = parse_merged_results(fixture("merged-all-degraded.txt"))
    assert parsed.degraded is True
    assert len(parsed.hits) == 1
    assert isinstance(parsed.hits[0], ThoughtHit)
    assert parsed.hits[0].tags == ["claude-code"]


def test_degraded_empty_result_yields_zero_hits_and_degraded_flag() -> None:
    parsed = parse_merged_results(fixture("merged-empty-degraded.txt"))
    assert parsed.hits == []
    assert parsed.degraded is True


def test_multiline_thought_content_is_kept_verbatim() -> None:
    parsed = parse_merged_results(fixture("merged-multiline-content.txt"))
    assert len(parsed.hits) == 1
    hit = parsed.hits[0]
    assert isinstance(hit, ThoughtHit)
    assert hit.content == "First line of a multiline thought\nsecond line kept verbatim"
    assert hit.tags == ["notes"]


def test_raw_text_is_retained() -> None:
    text = fixture("merged-all.txt")
    assert parse_merged_results(text).text == text


def test_malformed_text_raises_parse_error_with_offending_text() -> None:
    with pytest.raises(ParseError) as info:
        parse_merged_results("utter garbage")
    assert info.value.text == "utter garbage"
