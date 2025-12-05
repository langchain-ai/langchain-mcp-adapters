"""Tests for MCP elicitation callback support."""

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
            return f"Created profile for {name}: email={result.data.email}, age={result.data.age}"
        elif result.action == "decline":
            return f"User declined. Created minimal profile for {name}."
        else:
            return "Profile creation cancelled."

    return server


async def test_elicitation_callback_accept(socket_enabled) -> None:
    """Test elicitation callback with user accepting and providing data."""
    elicitation_requests: list[tuple[RequestContext, ElicitRequestParams, CallbackContext]] = []

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
