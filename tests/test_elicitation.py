"""Tests for MCP elicitation callback support."""

import os
from pathlib import Path

import pytest
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from pydantic import BaseModel

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from tests.utils import run_streamable_http


class UserDetails(BaseModel):
    email: str
    age: int


def _create_elicitation_server():
    server = FastMCP(port=8184)

    @server.tool()
    async def create_profile(name: str, ctx: Context) -> str:
        """Create a user profile with elicitation."""
        result = await ctx.elicit(
            message=f"Please provide details for {name}'s profile:",
            schema=UserDetails,
        )
        if result.action == "accept" and result.data:
            return (
                f"Created profile for {name}: "
                f"email={result.data.email}, age={result.data.age}"
            )
        if result.action == "decline":
            return f"User declined. Created minimal profile for {name}."
        return "Profile creation cancelled."

    return server


async def test_elicitation_callback_accept(socket_enabled) -> None:
    """Test elicitation callback with user accepting and providing data."""
    elicitation_requests: list[
        tuple[RequestContext, ElicitRequestParams, CallbackContext]
    ] = []

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        elicitation_requests.append((mcp_context, params, context))
        return ElicitResult(
            action="accept",
            content={"email": "alice@example.com", "age": 28},
        )

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "create_profile"

        # Call the tool
        result = await tools[0].ainvoke(
            {"args": {"name": "Alice"}, "id": "call_1", "type": "tool_call"}
        )

        # Verify elicitation callback was called
        assert len(elicitation_requests) == 1
        _, params, context = elicitation_requests[0]
        assert "Alice" in params.message
        assert context.server_name == "test"
        assert context.tool_name == "create_profile"

        # Verify result
        assert "alice@example.com" in str(result.content)
        assert "28" in str(result.content)


async def test_elicitation_callback_decline(socket_enabled) -> None:
    """Test elicitation callback with user declining."""

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="decline")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "Bob"}, "id": "call_2", "type": "tool_call"}
        )

        assert "declined" in str(result.content).lower()


async def test_elicitation_callback_cancel(socket_enabled) -> None:
    """Test elicitation callback with user cancelling."""

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="cancel")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "Charlie"}, "id": "call_3", "type": "tool_call"}
        )

        assert "cancelled" in str(result.content).lower()


# --- STDIO Transport Tests ---


async def test_elicitation_callback_accept_stdio() -> None:
    """Test elicitation callback with user accepting via stdio transport."""
    current_dir = Path(__file__).parent
    elicitation_server_path = os.path.join(current_dir, "servers/elicitation_server.py")

    elicitation_requests: list[
        tuple[RequestContext, ElicitRequestParams, CallbackContext]
    ] = []

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        elicitation_requests.append((mcp_context, params, context))
        return ElicitResult(
            action="accept",
            content={"email": "stdio@example.com", "age": 35},
        )

    client = MultiServerMCPClient(
        {
            "test": {
                "command": "python3",
                "args": [elicitation_server_path, "stdio"],
                "transport": "stdio",
            }
        },
        callbacks=Callbacks(on_elicitation=on_elicitation),
    )

    tools = await client.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "create_profile"

    # Call the tool
    result = await tools[0].ainvoke(
        {"args": {"name": "StdioUser"}, "id": "call_stdio_1", "type": "tool_call"}
    )

    # Verify elicitation callback was called
    assert len(elicitation_requests) == 1
    _, params, context = elicitation_requests[0]
    assert "StdioUser" in params.message
    assert context.server_name == "test"
    assert context.tool_name == "create_profile"

    # Verify result
    assert "stdio@example.com" in str(result.content)
    assert "35" in str(result.content)


async def test_elicitation_callback_decline_stdio() -> None:
    """Test elicitation callback with user declining via stdio transport."""
    current_dir = Path(__file__).parent
    elicitation_server_path = os.path.join(current_dir, "servers/elicitation_server.py")

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="decline")

    client = MultiServerMCPClient(
        {
            "test": {
                "command": "python3",
                "args": [elicitation_server_path, "stdio"],
                "transport": "stdio",
            }
        },
        callbacks=Callbacks(on_elicitation=on_elicitation),
    )

    tools = await client.get_tools()
    result = await tools[0].ainvoke(
        {"args": {"name": "StdioDecline"}, "id": "call_stdio_2", "type": "tool_call"}
    )

    assert "declined" in str(result.content).lower()


async def test_elicitation_callback_cancel_stdio() -> None:
    """Test elicitation callback with user cancelling via stdio transport."""
    current_dir = Path(__file__).parent
    elicitation_server_path = os.path.join(current_dir, "servers/elicitation_server.py")

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="cancel")

    client = MultiServerMCPClient(
        {
            "test": {
                "command": "python3",
                "args": [elicitation_server_path, "stdio"],
                "transport": "stdio",
            }
        },
        callbacks=Callbacks(on_elicitation=on_elicitation),
    )

    tools = await client.get_tools()
    result = await tools[0].ainvoke(
        {"args": {"name": "StdioCancel"}, "id": "call_stdio_3", "type": "tool_call"}
    )

    assert "cancelled" in str(result.content).lower()


# --- HTTP Transport Tests (alias for streamable_http) ---


@pytest.mark.parametrize("transport", ["http", "streamable-http"])
async def test_elicitation_callback_accept_http_variants(
    socket_enabled, transport
) -> None:
    """Test elicitation callback with user accepting via http transport variants."""
    elicitation_requests: list[
        tuple[RequestContext, ElicitRequestParams, CallbackContext]
    ] = []

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        elicitation_requests.append((mcp_context, params, context))
        return ElicitResult(
            action="accept",
            content={"email": "http@example.com", "age": 42},
        )

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": transport,
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "create_profile"

        # Call the tool
        result = await tools[0].ainvoke(
            {"args": {"name": "HttpUser"}, "id": "call_http_1", "type": "tool_call"}
        )

        # Verify elicitation callback was called
        assert len(elicitation_requests) == 1
        _, params, context = elicitation_requests[0]
        assert "HttpUser" in params.message
        assert context.server_name == "test"
        assert context.tool_name == "create_profile"

        # Verify result
        assert "http@example.com" in str(result.content)
        assert "42" in str(result.content)


@pytest.mark.parametrize("transport", ["http", "streamable-http"])
async def test_elicitation_callback_decline_http_variants(
    socket_enabled, transport
) -> None:
    """Test elicitation callback with user declining via http transport variants."""

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="decline")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": transport,
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "HttpDecline"}, "id": "call_http_2", "type": "tool_call"}
        )

        assert "declined" in str(result.content).lower()


@pytest.mark.parametrize("transport", ["http", "streamable-http"])
async def test_elicitation_callback_cancel_http_variants(
    socket_enabled, transport
) -> None:
    """Test elicitation callback with user cancelling via http transport variants."""

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="cancel")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": transport,
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "HttpCancel"}, "id": "call_http_3", "type": "tool_call"}
        )

        assert "cancelled" in str(result.content).lower()
