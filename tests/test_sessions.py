import warnings
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT,
    DEFAULT_STREAMABLE_HTTP_TIMEOUT,
    _create_streamable_http_session,
)
from tests.utils import run_streamable_http

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client_cm(session):
    """Return an async context manager that yields (read, write, get_session_id)."""
    read = MagicMock()
    write = MagicMock()
    get_session_id = MagicMock(return_value=None)

    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=(read, write, get_session_id))
    client_cm.__aexit__ = AsyncMock(return_value=False)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    return client_cm, session_cm, read, write


# ---------------------------------------------------------------------------
# Unit tests — mock streamable_http_client to inspect call arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uses_new_streamable_http_client_not_deprecated() -> None:
    """Test that _create_streamable_http_session calls the non-deprecated client."""
    mock_session = MagicMock()
    client_cm, session_cm, _, _ = _make_mock_client_cm(mock_session)

    with (
        patch(
            "langchain_mcp_adapters.sessions.streamable_http_client",
            return_value=client_cm,
        ) as mock_new,
        patch(
            "langchain_mcp_adapters.sessions.ClientSession",
            return_value=session_cm,
        ),
    ):
        async with _create_streamable_http_session(url="http://localhost/mcp"):
            pass

    mock_new.assert_called_once()
    # Paranoia: make sure we're not accidentally calling the old name anywhere
    assert "streamablehttp_client" not in str(mock_new.call_args)


@pytest.mark.asyncio
async def test_default_timeout_maps_to_httpx_timeout() -> None:
    """Test that default timeout and sse_read_timeout are forwarded as httpx.Timeout."""
    mock_session = MagicMock()
    client_cm, session_cm, _, _ = _make_mock_client_cm(mock_session)
    captured: list[httpx.AsyncClient] = []

    original_factory = __import__(
        "mcp.shared._httpx_utils", fromlist=["create_mcp_http_client"]
    ).create_mcp_http_client

    def capturing_factory(
        headers=None,
        timeout=None,
        auth=None,
    ) -> httpx.AsyncClient:
        client = original_factory(headers=headers, timeout=timeout, auth=auth)
        captured.append(client)
        return client

    with (
        patch(
            "langchain_mcp_adapters.sessions.streamable_http_client",
            return_value=client_cm,
        ),
        patch(
            "langchain_mcp_adapters.sessions.ClientSession",
            return_value=session_cm,
        ),
        patch(
            "langchain_mcp_adapters.sessions.create_mcp_http_client",
            side_effect=capturing_factory,
        ),
    ):
        async with _create_streamable_http_session(url="http://localhost/mcp"):
            pass

    assert len(captured) == 1
    client = captured[0]
    assert client.timeout.connect == DEFAULT_STREAMABLE_HTTP_TIMEOUT.total_seconds()
    assert (
        client.timeout.read == DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT.total_seconds()
    )


@pytest.mark.asyncio
async def test_headers_forwarded_to_http_client() -> None:
    """Test that custom headers are passed through to the httpx.AsyncClient."""
    mock_session = MagicMock()
    client_cm, session_cm, _, _ = _make_mock_client_cm(mock_session)
    received_kwargs: list[dict] = []

    def capturing_factory(headers=None, timeout=None, auth=None):
        received_kwargs.append({"headers": headers, "timeout": timeout, "auth": auth})
        return httpx.AsyncClient()

    with (
        patch(
            "langchain_mcp_adapters.sessions.streamable_http_client",
            return_value=client_cm,
        ),
        patch(
            "langchain_mcp_adapters.sessions.ClientSession",
            return_value=session_cm,
        ),
        patch(
            "langchain_mcp_adapters.sessions.create_mcp_http_client",
            side_effect=capturing_factory,
        ),
    ):
        async with _create_streamable_http_session(
            url="http://localhost/mcp",
            headers={"Authorization": "Bearer tok"},
        ):
            pass

    assert received_kwargs[0]["headers"] == {"Authorization": "Bearer tok"}


@pytest.mark.asyncio
async def test_custom_timeout_forwarded() -> None:
    """Test that custom timeout and sse_read_timeout are reflected in httpx.Timeout."""
    mock_session = MagicMock()
    client_cm, session_cm, _, _ = _make_mock_client_cm(mock_session)
    received_kwargs: list[dict] = []

    def capturing_factory(headers=None, timeout=None, auth=None):
        received_kwargs.append({"timeout": timeout})
        return httpx.AsyncClient()

    with (
        patch(
            "langchain_mcp_adapters.sessions.streamable_http_client",
            return_value=client_cm,
        ),
        patch(
            "langchain_mcp_adapters.sessions.ClientSession",
            return_value=session_cm,
        ),
        patch(
            "langchain_mcp_adapters.sessions.create_mcp_http_client",
            side_effect=capturing_factory,
        ),
    ):
        async with _create_streamable_http_session(
            url="http://localhost/mcp",
            timeout=timedelta(seconds=10),
            sse_read_timeout=timedelta(seconds=120),
        ):
            pass

    t = received_kwargs[0]["timeout"]
    assert isinstance(t, httpx.Timeout)
    assert t.connect == 10.0
    assert t.read == 120.0


@pytest.mark.asyncio
async def test_custom_httpx_client_factory_used_over_default() -> None:
    """Test that a provided httpx_client_factory replaces create_mcp_http_client."""
    mock_session = MagicMock()
    client_cm, session_cm, _, _ = _make_mock_client_cm(mock_session)
    custom_client = httpx.AsyncClient()
    factory_called = []

    def custom_factory(headers=None, timeout=None, auth=None):
        factory_called.append(True)
        return custom_client

    passed_http_client: list = []

    def capturing_streamable_http_client(
        url, *, http_client=None, terminate_on_close=True
    ):
        passed_http_client.append(http_client)
        return client_cm

    with (
        patch(
            "langchain_mcp_adapters.sessions.streamable_http_client",
            side_effect=capturing_streamable_http_client,
        ),
        patch(
            "langchain_mcp_adapters.sessions.ClientSession",
            return_value=session_cm,
        ),
    ):
        async with _create_streamable_http_session(
            url="http://localhost/mcp",
            httpx_client_factory=custom_factory,
        ):
            pass

    assert factory_called, "custom factory was not called"
    assert passed_http_client[0] is custom_client


# ---------------------------------------------------------------------------
# Smoke test — real FastMCP server, no DeprecationWarning
# ---------------------------------------------------------------------------


def _create_simple_server():
    """Minimal FastMCP server with one tool for smoke testing."""
    server = FastMCP("smoke-test")

    @server.tool()
    def ping() -> str:
        """Return pong."""
        return "pong"

    return server


@pytest.mark.asyncio
async def test_no_deprecation_warning_with_real_server(socket_enabled) -> None:
    """Test that streamable HTTP connection does not raise a DeprecationWarning."""
    with run_streamable_http(_create_simple_server, 8299):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            client = MultiServerMCPClient(
                {
                    "smoke": {
                        "url": "http://localhost:8299/mcp",
                        "transport": "streamable_http",
                    }
                }
            )
            tools = await client.get_tools(server_name="smoke")

        deprecation_warnings = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "streamablehttp_client" in str(w.message).lower()
        ]
        assert not deprecation_warnings, (
            f"DeprecationWarning about streamablehttp_client still raised: "
            f"{[str(w.message) for w in deprecation_warnings]}"
        )
        assert len(tools) == 1
        assert tools[0].name == "ping"
