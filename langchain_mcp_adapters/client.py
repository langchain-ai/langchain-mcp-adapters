from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Optional, cast

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession

from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.sessions import ConnectionConfig, connect_to_server
from langchain_mcp_adapters.tools import load_mcp_tools


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools from them."""

    def __init__(
        self,
        connections: dict[str, ConnectionConfig] | None = None,
    ) -> None:
        """Initialize a MultiServerMCPClient with MCP servers connections.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                If None, no initial connections are established.

        Example:

        ```python
        async with MultiServerMCPClient(
            {
                "math": {
                    "command": "python",
                    # Make sure to update to the full absolute path to your math_server.py file
                    "args": ["/path/to/math_server.py"],
                    "transport": "stdio",
                },
                "weather": {
                    # make sure you start your weather server on port 8000
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                }
            }
        ) as client:
            all_tools = client.get_tools()
            ...
        ```
        """
        self.connections: dict[str, ConnectionConfig] = connections or {}
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.server_name_to_tools: dict[str, list[BaseTool]] = {}

    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        await session.initialize()
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        self.server_name_to_tools[server_name] = server_tools

    async def connect_to_server(
        self,
        server_name: str,
        **connection_config: dict[str, Any],
    ) -> None:
        """Connect to an MCP server.

        This is a generic method that calls individual connection methods
        based on the provided transport parameter
        (e.g., `connect_to_server_via_stdio`, etc.).

        Args:
            server_name: Name to identify this server connection
            **connection_config: Additional arguments to pass to the specific connection method

        Raises:
            ValueError: If transport is not recognized
            ValueError: If required parameters for the specified transport are missing
        """
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(connect_to_server(connection_config)),
        )
        await self._initialize_session_and_load_tools(server_name, session)

    async def connect_to_server_via_stdio(
        self,
        server_name: str,
        **connection_config: dict[str, Any],
    ) -> None:
        """Connect to a specific MCP server using stdio

        Args:
            server_name: Name to identify this server connection
            **connection_config: connection config, e.g. StdioConnectionConfig
        """
        if "transport" not in connection_config:
            connection_config["transport"] = "stdio"

        await self.connect_to_server(server_name, **connection_config)

    async def connect_to_server_via_sse(
        self,
        server_name: str,
        **connection_config: dict[str, Any],
    ) -> None:
        """Connect to a specific MCP server using SSE

        Args:
            server_name: Name to identify this server connection
            **connection_config: connection config, e.g. SSEConnectionConfig
        """
        if "transport" not in connection_config:
            connection_config["transport"] = "sse"

        await self.connect_to_server(server_name, **connection_config)

    async def connect_to_server_via_streamable_http(
        self,
        server_name: str,
        **connection_config: dict[str, Any],
    ) -> None:
        """Connect to a specific MCP server using Streamable HTTP

        Args:
            server_name: Name to identify this server connection
            **connection_config: connection config, e.g. StreamableHttpConnectionConfig
        """
        if "transport" not in connection_config:
            connection_config["transport"] = "streamable_http"

        await self.connect_to_server(server_name, **connection_config)

    async def connect_to_server_via_websocket(
        self,
        server_name: str,
        **connection_config: dict[str, Any],
    ):
        """Connect to a specific MCP server using Websockets

        Args:
            server_name: Name to identify this server connection
            **connection_config: connection config, e.g. WebsocketConnectionConfig
        """
        if "transport" not in connection_config:
            connection_config["transport"] = "websocket"

        await self.connect_to_server(server_name, **connection_config)

    def get_tools(self) -> list[BaseTool]:
        """Get a list of all tools from all connected servers."""
        all_tools: list[BaseTool] = []
        for server_tools in self.server_name_to_tools.values():
            all_tools.extend(server_tools)
        return all_tools

    async def get_prompt(
        self, server_name: str, prompt_name: str, arguments: Optional[dict[str, Any]]
    ) -> list[HumanMessage | AIMessage]:
        """Get a prompt from a given MCP server."""
        session = self.sessions[server_name]
        return await load_mcp_prompt(session, prompt_name, arguments)

    async def get_resources(
        self, server_name: str, uris: str | list[str] | None = None
    ) -> list[Blob]:
        """Get resources from a given MCP server.

        Args:
            server_name: Name of the server to get resources from
            uris: Optional resource URI or list of URIs to load. If not provided, all resources will be loaded.

        Returns:
            A list of LangChain Blobs
        """
        session = self.sessions[server_name]
        return await load_mcp_resources(session, uris)

    async def __aenter__(self) -> "MultiServerMCPClient":
        try:
            connections = self.connections or {}
            for server_name, connection in connections.items():
                await self.connect_to_server(server_name, **connection)

            return self
        except Exception:
            await self.exit_stack.aclose()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.exit_stack.aclose()
