from contextlib import AsyncExitStack
from types import TracebackType
from typing import Literal, cast, Optional, List

from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools from them."""

    def __init__(self) -> None:
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.server_name_to_tools: dict[str, list[BaseTool]] = {}

    async def connect_to_server(
        self,
        server_name: str,
        *,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        encoding: str = "utf-8",
        encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict",
    ) -> None:
        """Connect to a specific MCP server"""
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            encoding=encoding,
            encoding_error_handler=encoding_error_handler,
        )

        # Create and store the connection
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
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

    def get_tools(
        self,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None
    ) -> list[BaseTool]:
        """
        Get a filtered list of tools from all connected servers.
        
        Args:
            include_tools: Optional list of tool names to include. If None, all tools are included.
            exclude_tools: Optional list of tool names to exclude. If None, no tools are excluded.
        
        Returns:
            List of filtered tools
        """
                
        all_tools: list[BaseTool] = []
        for server_tools in self.server_name_to_tools.values():
            all_tools.extend(server_tools)
            
        exclude_set = set(exclude_tools or [])
        filtered_tools = [
            tool for tool in all_tools
            if tool.name not in exclude_set
            and (include_tools is None or tool.name in include_tools)
        ]
        
        return filtered_tools

    async def __aenter__(self) -> "MultiServerMCPClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.exit_stack.aclose()
