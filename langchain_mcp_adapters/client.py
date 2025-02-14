from typing import cast
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools from them."""

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.server_name_to_tools: dict[str, list[StructuredTool]] = {}

    async def connect_to_server(
        self, server_name: str, *, command: str, args: list[str], env: dict = None
    ):
        """Connect to a specific MCP server"""
        server_params = StdioServerParameters(command=command, args=args, env=env)

        # Create and store the connection
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(ClientSession(read, write)),
        )

        # Initialize the session
        await session.initialize()
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        self.server_name_to_tools[server_name] = server_tools

    def get_tools(self) -> list[StructuredTool]:
        """Get a list of all tools from all connected servers."""
        all_tools: list[StructuredTool] = []
        for server_tools in self.server_name_to_tools.values():
            all_tools.extend(server_tools)
        return all_tools

    async def __aenter__(self) -> "MultiServerMCPClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()
