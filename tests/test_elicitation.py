"""Tests for MCP elicitation callback support.

Two server-side styles behave differently across the protocol boundary:

- `ctx.elicit()` in a tool body sends a server-to-client request mid-call.
  2026-07-28 removed those, so such a server can only elicit on a handshake-era
  connection; those tests pin `protocol="legacy"`.
- A resolver returning `Elicit(...)` works on both eras — the SDK batches it
  into an `InputRequiredResult` at 2026-07-28 — so the same server and the same
  `Callbacks` cover both.
"""

from typing import Annotated

import pytest
from mcp.client import ClientRequestContext
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.resolve import Elicit, Resolve
from mcp.types import ElicitRequestParams, ElicitResult
from pydantic import BaseModel

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from tests.utils import run_streamable_http


def _create_elicitation_server():
    class UserDetails(BaseModel):
        email: str
        age: int

    server = MCPServer()

    # Track how many times code before elicit runs (should be exactly once)
    server._pre_elicit_call_count = 0

    @server.tool()
    async def create_profile(name: str, ctx: Context) -> str:
        """Create a user profile with elicitation."""
        # This code should only run once, not be re-executed after elicitation
        server._pre_elicit_call_count += 1

        result = await ctx.elicit(
            message=f"Please provide details for {name}'s profile:",
            schema=UserDetails,
        )
        if result.action == "accept" and result.data:
            return (
                f"Created profile for {name}: "
                f"email={result.data.email}, age={result.data.age}, "
                f"pre_elicit_calls={server._pre_elicit_call_count}"
            )
        if result.action == "decline":
            return (
                f"User declined. Created minimal profile for {name}. "
                f"pre_elicit_calls={server._pre_elicit_call_count}"
            )
        return (
            f"Profile creation cancelled. "
            f"pre_elicit_calls={server._pre_elicit_call_count}"
        )

    return server


async def test_elicitation_callback_accept(socket_enabled) -> None:
    """Test elicitation callback with user accepting and providing data."""
    elicitation_requests: list[
        tuple[ClientRequestContext, ElicitRequestParams, CallbackContext]
    ] = []

    async def on_elicitation(
        mcp_context: ClientRequestContext,
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
                    "transport": "http",
                    # `ctx.elicit()` needs a server-to-client back-channel,
                    # which 2026-07-28 removed.
                    "protocol": "legacy",
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

        # Verify code before ctx.elicit only ran once
        # (not re-executed after elicitation)
        assert "pre_elicit_calls=1" in str(result.content)


async def test_elicitation_callback_decline(socket_enabled) -> None:
    """Test elicitation callback with user declining."""

    async def on_elicitation(
        mcp_context: ClientRequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="decline")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "http",
                    # `ctx.elicit()` needs a server-to-client back-channel,
                    # which 2026-07-28 removed.
                    "protocol": "legacy",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "Bob"}, "id": "call_2", "type": "tool_call"}
        )

        assert "declined" in str(result.content).lower()
        # Verify code before ctx.elicit only ran once
        assert "pre_elicit_calls=1" in str(result.content)


async def test_elicitation_callback_cancel(socket_enabled) -> None:
    """Test elicitation callback with user cancelling."""

    async def on_elicitation(
        mcp_context: ClientRequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="cancel")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "http",
                    # `ctx.elicit()` needs a server-to-client back-channel,
                    # which 2026-07-28 removed.
                    "protocol": "legacy",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "Charlie"}, "id": "call_3", "type": "tool_call"}
        )

        assert "cancelled" in str(result.content).lower()
        # Verify code before ctx.elicit only ran once
        assert "pre_elicit_calls=1" in str(result.content)


# --- resolver-based elicitation, which spans both eras -----------------------


class _UserDetails(BaseModel):
    email: str
    age: int


def _ask_details(name: str) -> Elicit[_UserDetails]:
    return Elicit(f"Please provide details for {name}'s profile:", _UserDetails)


def _create_resolver_elicitation_server():
    server = MCPServer()

    @server.tool()
    async def create_profile(
        name: str,
        details: Annotated[_UserDetails, Resolve(_ask_details)],
    ) -> str:
        """Create a user profile, eliciting the details from the client."""
        return f"Created profile for {name}: email={details.email}, age={details.age}"

    return server


@pytest.mark.parametrize("protocol", ["auto", "legacy"])
async def test_resolver_elicitation_spans_both_eras(socket_enabled, protocol) -> None:
    """The same callback answers elicitation on either side of the boundary."""
    seen: list[ElicitRequestParams] = []

    async def on_elicitation(
        mcp_context: ClientRequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        seen.append(params)
        return ElicitResult(
            action="accept", content={"email": "alice@example.com", "age": 28}
        )

    with run_streamable_http(_create_resolver_elicitation_server, 8185):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8185/mcp",
                    "transport": "http",
                    "protocol": protocol,
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )
        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "Alice"}, "id": "1", "type": "tool_call"}
        )
        negotiated = (await client.get_server_info())["test"].protocol_version

    assert len(seen) == 1
    assert "Alice" in seen[0].message
    assert "alice@example.com" in str(result.content)
    # The "auto" run must really have been on 2026-07-28, not a silent fallback.
    assert negotiated == ("2026-07-28" if protocol == "auto" else "2025-11-25")
