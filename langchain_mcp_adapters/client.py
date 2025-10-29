"""Client for connecting to multiple MCP servers and loading LangChain tools/resources.

This module provides the MultiServerMCPClient class for managing connections to multiple
MCP servers and loading tools, prompts, and resources from them.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks
from langchain_mcp_adapters.interceptors import Interceptors
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.sessions import (
    Connection,
    McpHttpClientFactory,
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    WebsocketConnection,
    create_session,
)
from langchain_mcp_adapters.tools import load_mcp_tools

ASYNC_CONTEXT_MANAGER_ERROR = (
    "As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient cannot be used as a "
    "context manager (e.g., async with MultiServerMCPClient(...)). "
    "Instead, you can do one of the following:\n"
    "1. client = MultiServerMCPClient(...)\n"
    "   tools = await client.get_tools()\n"
    "2. client = MultiServerMCPClient(...)\n"
    "   async with client.session(server_name) as session:\n"
    "       tools = await load_mcp_tools(session)"
)


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers.

    Loads LangChain-compatible tools, prompts and resources from MCP servers.
    """

    def __init__(
        self,
        connections: dict[str, Connection] | None = None,
        *,
        callbacks: Callbacks | None = None,
        interceptors: Interceptors | None = None,
    ) -> None:
        """Initialize a MultiServerMCPClient with MCP servers connections.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                If None, no initial connections are established.
            callbacks: Optional callbacks for handling notifications and events.
            interceptors: Optional interceptors for modifying requests and responses.

        Example: basic usage (starting a new session on each tool call)

        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "math": {
                    "command": "python",
                    # Make sure to update to the full absolute path to your
                    # math_server.py file
                    "args": ["/path/to/math_server.py"],
                    "transport": "stdio",
                },
                "weather": {
                    # Make sure you start your weather server on port 8000
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                }
            }
        )
        all_tools = await client.get_tools()
        ```

        Example: explicitly starting a session

        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        client = MultiServerMCPClient({...})
        async with client.session("math") as session:
            tools = await load_mcp_tools(session)
        ```
        """
        self.connections: dict[str, Connection] = (
            connections if connections is not None else {}
        )
        self.callbacks = callbacks or Callbacks()
        self.interceptors = interceptors or Interceptors()

    @asynccontextmanager
    async def session(
        self,
        server_name: str,
        *,
        auto_initialize: bool = True,
    ) -> AsyncIterator[ClientSession]:
        """Create an MCP client session context manager.

        Args:
            server_name: Server connection identifier.
            auto_initialize: Auto-initialize session (default: True).

        Yields:
            Initialized ClientSession.

        Raises:
            ValueError: If server_name not in connections.
        """
        if server_name not in self.connections:
            msg = (
                f"Couldn't find a server with name '{server_name}', "
                f"expected one of '{list(self.connections.keys())}'"
            )
            raise ValueError(msg)

        mcp_callbacks = self.callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name)
        )

        async with create_session(
            self.connections[server_name], mcp_callbacks=mcp_callbacks
        ) as session:
            if auto_initialize:
                await session.initialize()
            yield session

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        """Get tools from MCP servers.

        Note: Creates new session per tool call. Tools from all servers
        returned if server_name is None.

        Args:
            server_name: Optional server to fetch tools from. If None,
                returns tools from all connected servers.

        Returns:
            List of LangChain BaseTool instances.

        Raises:
            ValueError: If specified server_name not in connections.
        """
        if server_name is not None:
            if server_name not in self.connections:
                msg = (
                    f"Couldn't find a server with name '{server_name}', "
                    f"expected one of '{list(self.connections.keys())}'"
                )
                raise ValueError(msg)
            return await load_mcp_tools(
                None,
                connection=self.connections[server_name],
                callbacks=self.callbacks,
                server_name=server_name,
                interceptors=self.interceptors,
            )

        all_tools: list[BaseTool] = []
        load_mcp_tool_tasks = []
        for name, connection in self.connections.items():
            load_mcp_tool_task = asyncio.create_task(
                load_mcp_tools(
                    None,
                    connection=connection,
                    callbacks=self.callbacks,
                    server_name=name,
                    interceptors=self.interceptors,
                )
            )
            load_mcp_tool_tasks.append(load_mcp_tool_task)
        tools_list = await asyncio.gather(*load_mcp_tool_tasks)
        for tools in tools_list:
            all_tools.extend(tools)
        return all_tools

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> list[HumanMessage | AIMessage]:
        """Get a prompt from an MCP server.

        Args:
            server_name: Name of the server to fetch prompt from.
            prompt_name: Name of the prompt to load.
            arguments: Optional prompt template arguments.

        Returns:
            List of LangChain messages (HumanMessage or AIMessage).

        Raises:
            ValueError: If server_name not found in connections.
        """
        async with self.session(server_name) as session:
            return await load_mcp_prompt(session, prompt_name, arguments=arguments)

    async def get_resources(
        self,
        server_name: str,
        *,
        uris: str | list[str] | None = None,
    ) -> list[Blob]:
        """Get resources from an MCP server.

        Args:
            server_name: Server to fetch resources from.
            uris: Optional URI(s) to load. If None, loads all resources.

        Returns:
            List of LangChain Blob objects.

        Raises:
            ValueError: If server_name not in connections.
        """
        async with self.session(server_name) as session:
            return await load_mcp_resources(session, uris=uris)

    async def __aenter__(self) -> "MultiServerMCPClient":
        """Async context manager entry point.

        Raises:
            NotImplementedError: Context manager support has been removed.
        """
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)

    def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit point.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Raises:
            NotImplementedError: Context manager support has been removed.
        """
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)


__all__ = [
    "Callbacks",
    "McpHttpClientFactory",
    "MultiServerMCPClient",
    "SSEConnection",
    "StdioConnection",
    "StreamableHttpConnection",
    "WebsocketConnection",
]
