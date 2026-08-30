from agent_memory import AuthError, ParseError, ToolError, TransportError, classify_tool_error_text


def test_scope_denial() -> None:
    text = "Insufficient scope: 'memory_capture' requires 'memory:write' (caller 'readonly' has: memory:read)."
    assert classify_tool_error_text(text) == "scope_denied"


def test_validation_error() -> None:
    assert classify_tool_error_text("Error: 'query' must be a non-empty string.") == "validation"


def test_wildcard_only_identity() -> None:
    assert (
        classify_tool_error_text("This credential has no concrete namespace to capture into (wildcard-only identity).")
        == "no_namespace"
    )
    assert (
        classify_tool_error_text("This credential has no concrete namespace for KV access (wildcard-only identity).")
        == "no_namespace"
    )


def test_degraded_documents_corpus() -> None:
    text = "note: the documents corpus was unavailable (vault not mounted or its embedding backend was unreachable)."
    assert classify_tool_error_text(text) == "degraded"


def test_operational_failures() -> None:
    assert classify_tool_error_text("Search failed: connect ECONNREFUSED") == "failed"
    assert classify_tool_error_text("Capture failed: boom") == "failed"
    assert classify_tool_error_text("memory_tree failed: boom") == "failed"
    assert classify_tool_error_text("memory_read_document failed: boom") == "failed"


def test_anything_unrecognised_is_failed() -> None:
    assert classify_tool_error_text("some novel error text") == "failed"


def test_tool_error_carries_tool_text_and_kind() -> None:
    err = ToolError("memory_capture", "Insufficient scope: nope")
    assert err.tool == "memory_capture"
    assert err.text == "Insufficient scope: nope"
    assert err.kind == "scope_denied"
    assert str(err) == "memory_capture: Insufficient scope: nope"


def test_auth_error_is_a_transport_error() -> None:
    assert issubclass(AuthError, TransportError)


def test_parse_error_retains_offending_text() -> None:
    err = ParseError("bad", "the whole text")
    assert err.text == "the whole text"
    assert str(err) == "bad"
