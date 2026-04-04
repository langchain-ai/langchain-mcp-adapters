from unittest.mock import AsyncMock

import pytest
from mcp.server import FastMCP
from mcp.types import InitializeResult

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.server_info import load_mcp_server_info
from tests.utils import run_streamable_http


def _create_server_with_instructions():
    server = FastMCP(
        "test-server",
        instructions="Use this server for testing purposes only.",
        port=8187,
    )

    @server.tool()
    def ping() -> str:
        """Ping the server."""
        return "pong"

    return server


def _create_server_without_instructions():
    server = FastMCP("no-instructions-server", port=8188)

    @server.tool()
    def ping() -> str:
        """Ping the server."""
        return "pong"

    return server


async def test_load_mcp_server_info_with_connection(socket_enabled) -> None:
    """Test loading server info using a connection config."""
    with run_streamable_http(_create_server_with_instructions, 8187):
        result = await load_mcp_server_info(
            connection={
                "url": "http://localhost:8187/mcp",
                "transport": "streamable_http",
            },
        )
        assert isinstance(result, InitializeResult)
        assert result.instructions == "Use this server for testing purposes only."
        assert result.serverInfo.name == "test-server"


async def test_load_mcp_server_info_no_instructions(socket_enabled) -> None:
    """Test loading server info when server has no instructions."""
    with run_streamable_http(_create_server_without_instructions, 8188):
        result = await load_mcp_server_info(
            connection={
                "url": "http://localhost:8188/mcp",
                "transport": "streamable_http",
            },
        )
        assert isinstance(result, InitializeResult)
        assert result.instructions is None


async def test_load_mcp_server_info_with_session() -> None:
    """Test loading server info with an uninitialized session."""
    mock_result = InitializeResult(
        protocolVersion="2025-03-26",
        capabilities={},
        serverInfo={"name": "mock-server", "version": "1.0"},
        instructions="Mock instructions",
    )
    session = AsyncMock()
    session.initialize.return_value = mock_result

    result = await load_mcp_server_info(session)

    session.initialize.assert_called_once()
    assert result.instructions == "Mock instructions"
    assert result.serverInfo.name == "mock-server"


async def test_load_mcp_server_info_raises_without_args() -> None:
    """Test that ValueError is raised when neither session nor connection."""
    with pytest.raises(ValueError, match="Either a session or a connection"):
        await load_mcp_server_info()


async def test_client_get_server_info(socket_enabled) -> None:
    """Test MultiServerMCPClient.get_server_info returns info for all servers."""
    with (
        run_streamable_http(_create_server_with_instructions, 8187),
        run_streamable_http(_create_server_without_instructions, 8188),
    ):
        client = MultiServerMCPClient(
            {
                "with_instructions": {
                    "url": "http://localhost:8187/mcp",
                    "transport": "streamable_http",
                },
                "without_instructions": {
                    "url": "http://localhost:8188/mcp",
                    "transport": "streamable_http",
                },
            },
        )
        info = await client.get_info()
        assert len(info) == 2
        assert info["with_instructions"].instructions == (
            "Use this server for testing purposes only."
        )
        assert info["with_instructions"].serverInfo.name == "test-server"
        assert info["without_instructions"].instructions is None
