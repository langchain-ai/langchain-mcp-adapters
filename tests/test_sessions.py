"""Tests for session helpers in ``langchain_mcp_adapters.sessions``."""

import pytest

from langchain_mcp_adapters.sessions import _expand_env_vars


def test_expand_env_vars_expands_braced_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``${VAR}`` reference is replaced with the environment value."""
    monkeypatch.setenv("MCP_TEST_TOKEN", "secret-value")
    assert _expand_env_vars("${MCP_TEST_TOKEN}") == "secret-value"


def test_expand_env_vars_expands_within_surrounding_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A braced reference embedded in a larger string is expanded in place."""
    monkeypatch.setenv("MCP_TEST_HOST", "example.com")
    assert _expand_env_vars("https://${MCP_TEST_HOST}/sse") == "https://example.com/sse"


def test_expand_env_vars_expands_multiple_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple braced references in one string are all expanded."""
    monkeypatch.setenv("MCP_TEST_USER", "alice")
    monkeypatch.setenv("MCP_TEST_PASS", "pw123")
    assert _expand_env_vars("${MCP_TEST_USER}:${MCP_TEST_PASS}") == "alice:pw123"


def test_expand_env_vars_leaves_bare_reference_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare ``$VAR`` (no braces) is preserved so literal ``$`` is never corrupted."""
    monkeypatch.setenv("MCP_TEST_TOKEN", "secret-value")
    assert _expand_env_vars("$MCP_TEST_TOKEN") == "$MCP_TEST_TOKEN"


def test_expand_env_vars_preserves_undefined_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undefined ``${MISSING}`` reference is kept verbatim."""
    monkeypatch.delenv("MCP_TEST_MISSING", raising=False)
    assert _expand_env_vars("${MCP_TEST_MISSING}") == "${MCP_TEST_MISSING}"


def test_expand_env_vars_no_reference_returns_value_unchanged() -> None:
    """A plain string with no references is returned unchanged."""
    assert _expand_env_vars("plain-value") == "plain-value"


def test_expand_env_vars_literal_dollar_in_password_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal ``$`` in a value (e.g. a password) is not treated as a variable."""
    monkeypatch.delenv("p", raising=False)
    assert _expand_env_vars("pa$$word") == "pa$$word"
