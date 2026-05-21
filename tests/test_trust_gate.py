"""Tests for TrustGateInterceptor."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import ToolMessage

from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.trust_gate import TrustGateInterceptor


@pytest.fixture
def interceptor() -> TrustGateInterceptor:
    return TrustGateInterceptor(min_trust_score=60.0, cache_ttl_seconds=0)


@pytest.fixture
def request() -> MCPToolCallRequest:
    return MCPToolCallRequest(
        name="check_trust",
        args={"server_url": "https://example.com/mcp"},
        server_name="example-server",
    )


@pytest.fixture
def handler() -> AsyncMock:
    mock = AsyncMock()
    mock.return_value = ToolMessage(content="success", tool_call_id="check_trust")
    return mock


def _mock_response(score: float | None, status: int = 200):
    """Create a mock httpx response."""
    import httpx

    if score is not None:
        data = {"trust_score": score, "server": "example-server"}
    else:
        data = {"error": "not found"}

    resp = httpx.Response(
        status_code=status,
        json=data,
        request=httpx.Request("GET", "https://example.com"),
    )
    return resp


@pytest.mark.asyncio
async def test_trusted_server_passes(interceptor, request, handler):
    """Tool call proceeds when server trust score meets threshold."""
    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get.return_value = _mock_response(85.0)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        result = await interceptor(request, handler)

    handler.assert_called_once_with(request)
    assert isinstance(result, ToolMessage)
    assert result.content == "success"


@pytest.mark.asyncio
async def test_untrusted_server_blocked(interceptor, request, handler):
    """Tool call is blocked when server trust score is below threshold."""
    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get.return_value = _mock_response(30.0)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        result = await interceptor(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert "trust score of 30.0" in result.content
    assert "blocked" in result.content


@pytest.mark.asyncio
async def test_unreachable_oracle_blocks(interceptor, request, handler):
    """Tool call is blocked when trust oracle is unreachable."""
    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get.side_effect = Exception("Connection refused")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        result = await interceptor(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert "Could not verify trust" in result.content


@pytest.mark.asyncio
async def test_cache_reuses_score(request, handler):
    """Cached scores are reused within TTL."""
    interceptor = TrustGateInterceptor(
        min_trust_score=60.0, cache_ttl_seconds=300
    )

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get.return_value = _mock_response(85.0)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        # First call — hits the API
        await interceptor(request, handler)
        # Second call — should use cache
        await interceptor(request, handler)

    # httpx.AsyncClient should only be instantiated once (first call)
    assert mock_client.call_count == 1


@pytest.mark.asyncio
async def test_custom_observatory_url(request, handler):
    """Custom observatory URL is used for trust checks."""
    interceptor = TrustGateInterceptor(
        observatory_url="https://custom-trust.example.com",
        min_trust_score=50.0,
        cache_ttl_seconds=0,
    )

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get.return_value = _mock_response(75.0)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        await interceptor(request, handler)

    instance.get.assert_called_once()
    call_url = instance.get.call_args[0][0]
    assert call_url.startswith("https://custom-trust.example.com/")
