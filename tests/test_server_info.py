import contextlib
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession
from mcp.server import FastMCP
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    InitializeResult,
    LoggingMessageNotificationParams,
    ServerCapabilities,
)

from langchain_mcp_adapters import server_info as server_info_module
from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.server_info import load_mcp_server_info
from tests.utils import run_streamable_http

# A port nothing listens on, used to exercise unreachable-server handling.
CLOSED_PORT = 8189


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


def _mock_initialize_result() -> InitializeResult:
    return InitializeResult(
        protocolVersion=LATEST_PROTOCOL_VERSION,
        capabilities={},
        serverInfo={"name": "mock-server", "version": "1.0"},
        instructions="Mock instructions",
    )


async def test_load_mcp_server_info_with_connection(socket_enabled) -> None:
    """Test loading server info using a connection config."""
    with run_streamable_http(_create_server_with_instructions, 8187):
        result = await load_mcp_server_info(
            None,
            connection={
                "url": "http://localhost:8187/mcp",
                "transport": "streamable_http",
            },
        )
        assert isinstance(result, InitializeResult)
        assert result.instructions == "Use this server for testing purposes only."
        assert result.serverInfo.name == "test-server"
        # The server registers a `ping` tool, so it must advertise tool support.
        assert result.capabilities.tools is not None
        assert result.protocolVersion


async def test_load_mcp_server_info_over_stdio() -> None:
    """Test loading server info over the stdio transport."""
    math_server_path = os.path.join(Path(__file__).parent, "servers/math_server.py")

    result = await load_mcp_server_info(
        None,
        connection={
            "command": "python3",
            "args": [math_server_path],
            "transport": "stdio",
        },
    )
    assert isinstance(result, InitializeResult)
    assert result.serverInfo.name == "Math"
    assert result.capabilities.tools is not None


async def test_load_mcp_server_info_no_instructions(socket_enabled) -> None:
    """Test loading server info when server has no instructions."""
    with run_streamable_http(_create_server_without_instructions, 8188):
        result = await load_mcp_server_info(
            None,
            connection={
                "url": "http://localhost:8188/mcp",
                "transport": "streamable_http",
            },
        )
        assert isinstance(result, InitializeResult)
        assert result.instructions is None
        assert result.serverInfo.name == "no-instructions-server"


async def test_load_mcp_server_info_with_session() -> None:
    """Test that a provided session is initialized and its result returned."""
    mock_result = _mock_initialize_result()
    # `spec` keeps sync methods (`get_server_capabilities`) sync and async ones
    # (`initialize`) async, matching the real `ClientSession`.
    session = AsyncMock(spec=ClientSession)
    # `None` capabilities means the session has not been initialized yet.
    session.get_server_capabilities.return_value = None
    session.initialize.return_value = mock_result

    result = await load_mcp_server_info(session)

    session.initialize.assert_called_once()
    assert result.instructions == "Mock instructions"
    assert result.serverInfo.name == "mock-server"


async def test_load_mcp_server_info_rejects_initialized_session() -> None:
    """Test that an already-initialized session is rejected, not re-initialized."""
    session = AsyncMock(spec=ClientSession)
    session.get_server_capabilities.return_value = ServerCapabilities()

    with pytest.raises(ValueError, match="already been initialized"):
        await load_mcp_server_info(session)

    session.initialize.assert_not_called()


async def test_load_mcp_server_info_rejects_initialized_real_session(
    socket_enabled,
) -> None:
    """Test the initialized-session guard on a real session, and the escape hatch."""
    with run_streamable_http(_create_server_with_instructions, 8187):
        client = MultiServerMCPClient(
            {
                "with_instructions": {
                    "url": "http://localhost:8187/mcp",
                    "transport": "streamable_http",
                },
            },
        )

        # `session()` initializes by default, so it must be rejected.
        async with client.session("with_instructions") as session:
            with pytest.raises(ValueError, match="already been initialized"):
                await load_mcp_server_info(session)

        # `auto_initialize=False` is the documented way to use the session path.
        async with client.session(
            "with_instructions", auto_initialize=False
        ) as session:
            result = await load_mcp_server_info(session)
            assert result.serverInfo.name == "test-server"
            # The session is usable afterwards, since it is now initialized.
            tools = await session.list_tools()
            assert [tool.name for tool in tools.tools] == ["ping"]


async def test_load_mcp_server_info_raises_without_args() -> None:
    """Test that ValueError is raised when neither session nor connection is given."""
    with pytest.raises(ValueError, match="Either a session or a connection"):
        await load_mcp_server_info(None)


async def test_load_mcp_server_info_passes_server_name_to_callbacks(
    monkeypatch,
) -> None:
    """Test that server_name reaches the CallbackContext handed to callbacks."""
    contexts: list[CallbackContext] = []

    async def logging_callback(params, context) -> None:
        contexts.append(context)

    captured = {}

    @contextlib.asynccontextmanager
    async def fake_create_session(connection, *, mcp_callbacks=None):
        captured["mcp_callbacks"] = mcp_callbacks
        session = AsyncMock()
        session.initialize.return_value = _mock_initialize_result()
        yield session

    monkeypatch.setattr(server_info_module, "create_session", fake_create_session)

    await load_mcp_server_info(
        None,
        connection={
            "url": "http://localhost:8187/mcp",
            "transport": "streamable_http",
        },
        callbacks=Callbacks(on_logging_message=logging_callback),
        server_name="my_server",
    )

    mcp_callbacks = captured["mcp_callbacks"]
    assert mcp_callbacks.logging_callback is not None
    await mcp_callbacks.logging_callback(
        LoggingMessageNotificationParams(level="info", data="hello")
    )
    assert [context.server_name for context in contexts] == ["my_server"]


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
        info = await client.get_server_info()
        assert len(info) == 2
        assert info["with_instructions"].instructions == (
            "Use this server for testing purposes only."
        )
        assert info["with_instructions"].serverInfo.name == "test-server"
        assert info["without_instructions"].instructions is None
        assert info["without_instructions"].serverInfo.name == "no-instructions-server"


async def test_client_get_server_info_single_server(socket_enabled) -> None:
    """Test that server_name scopes get_server_info to one server."""
    with run_streamable_http(_create_server_with_instructions, 8187):
        client = MultiServerMCPClient(
            {
                "with_instructions": {
                    "url": "http://localhost:8187/mcp",
                    "transport": "streamable_http",
                },
                # Unreachable, but must not be queried when scoping by name.
                "unreachable": {
                    "url": f"http://localhost:{CLOSED_PORT}/mcp",
                    "transport": "streamable_http",
                },
            },
        )
        info = await client.get_server_info(server_name="with_instructions")
        assert list(info) == ["with_instructions"]
        assert info["with_instructions"].serverInfo.name == "test-server"


async def test_client_get_server_info_unknown_server() -> None:
    """Test that an unknown server_name raises ValueError."""
    client = MultiServerMCPClient(
        {
            "with_instructions": {
                "url": "http://localhost:8187/mcp",
                "transport": "streamable_http",
            },
        },
    )
    with pytest.raises(ValueError, match="Couldn't find a server with name 'nope'"):
        await client.get_server_info(server_name="nope")


async def test_client_get_server_info_no_connections() -> None:
    """Test that get_server_info returns an empty dict with no connections."""
    assert await MultiServerMCPClient().get_server_info() == {}


async def test_client_get_server_info_names_failing_server(socket_enabled) -> None:
    """Test that a failing server is named in the error, and healthy ones aren't."""
    with run_streamable_http(_create_server_with_instructions, 8187):
        client = MultiServerMCPClient(
            {
                "with_instructions": {
                    "url": "http://localhost:8187/mcp",
                    "transport": "streamable_http",
                },
                "unreachable": {
                    "url": f"http://localhost:{CLOSED_PORT}/mcp",
                    "transport": "streamable_http",
                },
            },
        )
        with pytest.raises(RuntimeError, match="'unreachable'") as exc_info:
            await client.get_server_info()

        message = str(exc_info.value)
        assert "1 of 2" in message
        # The healthy server must not be blamed for the failure.
        assert "'with_instructions'" not in message
