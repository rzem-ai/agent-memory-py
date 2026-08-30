"""Public vocabulary and result types for the agent-memory client.

The literal vocabularies mirror the server's wire contract (its
``src/tools/shared.ts`` and ``src/db/schema.ts``); the parsed shapes mirror its
frozen text formats in ``src/domain/recall.ts``. Change only in lockstep with
the server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, get_args

Corpus = Literal["thoughts", "documents", "all"]
CORPORA: tuple[Corpus, ...] = get_args(Corpus)

RelevanceMode = Literal["recency_weighted", "similarity", "recent", "since"]
RELEVANCE_MODES: tuple[RelevanceMode, ...] = get_args(RelevanceMode)

TreeOp = Literal["list", "read", "search"]
TREE_OPS: tuple[TreeOp, ...] = get_args(TreeOp)

TreeNodeState = Literal["open", "sealed", "summarised"]
TREE_NODE_STATES: tuple[TreeNodeState, ...] = get_args(TreeNodeState)

DocumentTaint = Literal["internal", "external"]
DOCUMENT_TAINTS: tuple[DocumentTaint, ...] = get_args(DocumentTaint)

SyncSourceKind = Literal["mail", "github", "folder", "articles", "calendar"]
"""Source kinds the server currently syncs. Parsed hits keep ``source_kind`` as
a plain ``str`` so a new kind on the server never breaks this client."""
SYNC_SOURCE_KINDS: tuple[SyncSourceKind, ...] = get_args(SyncSourceKind)

DOCUMENT_BODY_MAX_CHARS = 20_000
"""Default and hard-maximum body size for read_document (characters)."""

DOCUMENTS_UNAVAILABLE_NOTE = (
    "note: the documents corpus was unavailable (vault not mounted or its embedding backend was unreachable)."
)
"""The server's degraded-documents note, verbatim (a frozen sentinel)."""


class RankedThought(TypedDict):
    """A thoughts-corpus search row, verbatim from the server's structuredContent
    (snake_case wire shape; ``score`` is None for non-composite modes)."""

    id: str
    agent_id: str
    content: str
    tags: list[str]
    similarity: float
    score: float | None
    created_at: str


@dataclass(frozen=True, slots=True)
class ThoughtHit:
    """A thought hit parsed from the merged (documents/all) text format.
    ``date`` is day-precision — the text carries no finer resolution."""

    rank: float
    similarity: float
    date: str
    agent_id: str
    id: str
    content: str
    tags: list[str] = field(default_factory=list)
    corpus: Literal["thoughts"] = "thoughts"
    taint: Literal["internal"] = "internal"


@dataclass(frozen=True, slots=True)
class DocumentHit:
    """A document hit parsed from the merged text format. ``snippet`` is the
    formatter's ``<title> - <excerpt>`` join, which is not reliably splittable."""

    taint: DocumentTaint
    rank: float
    similarity: float
    date: str
    source_kind: str
    document_id: str
    vault_path: str
    snippet: str
    corpus: Literal["documents"] = "documents"


MergedHit = ThoughtHit | DocumentHit


@dataclass(frozen=True, slots=True)
class ParsedMergedResults:
    corpus: Literal["documents", "all"] | None
    """None when the empty sentinel left no header to read the corpus from."""
    degraded: bool
    hits: list[MergedHit]
    text: str
    """The raw frozen text, retained so no information is ever destroyed."""


@dataclass(frozen=True, slots=True)
class TreeWindow:
    from_: str
    to: str


@dataclass(frozen=True, slots=True)
class TreeListEntry:
    path: str
    state: TreeNodeState
    window: TreeWindow
    doc_count: int
    last_appended_date: str | None
    """Day-precision (``last:`` in the text); None when the node never appended."""


@dataclass(frozen=True, slots=True)
class ParsedTreeList:
    scope: str
    entries: list[TreeListEntry]


@dataclass(frozen=True, slots=True)
class ParsedTreeNode:
    path: str
    state: TreeNodeState
    window: TreeWindow
    doc_count: int
    last_appended_at: str | None
    """Full ISO timestamp (``last appended:`` keeps raw precision in the text)."""
    summary_md: str | None


@dataclass(frozen=True, slots=True)
class TreeSearchHit:
    path: str
    state: TreeNodeState
    rank: float
    similarity: float
    window: TreeWindow
    excerpt: str | None


@dataclass(frozen=True, slots=True)
class ParsedTreeSearch:
    query: str | None
    """None when the empty sentinel left no header to read the query from."""
    results: list[TreeSearchHit]


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    id: str
    title: str
    source_kind: str
    external_id: str
    taint: DocumentTaint
    score: float
    event_at: str
    ingested_at: str
    vault_path: str
    provenance: dict[str, Any]
    body_source: Literal["vault", "chunks"]
    truncated: bool
    body: str
