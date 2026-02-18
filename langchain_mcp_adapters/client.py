"""Client for connecting to multiple MCP servers and loading LC tools/resources.

This module provides the `MultiServerMCPClient` class for managing connections
to multiple MCP servers and loading tools, prompts, and resources from them.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks
from langchain_mcp_adapters.interceptors import ToolCallInterceptor
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
        tool_interceptors: list[ToolCallInterceptor] | None = None,
        tool_name_prefix: bool = False,
    ) -> None:
        """Initialize a `MultiServerMCPClient` with MCP servers connections.

        Args:
            connections: A `dict` mapping server names to connection configurations. If
                `None`, no initial connections are established.
            callbacks: Optional callbacks for handling notifications and events.
            tool_interceptors: Optional list of tool call interceptors for modifying
                requests and responses.
            tool_name_prefix: If `True`, tool names are prefixed with the server name
                using an underscore separator (e.g., `"weather_search"` instead of
                `"search"`). This helps avoid conflicts when multiple servers have tools
                with the same name. Defaults to `False`.

        !!! example "Basic usage (starting a new session on each tool call)"

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
                        "transport": "http",
                    }
                }
            )
            all_tools = await client.get_tools()
            ```

        !!! example "Explicitly starting a session"

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
        self.tool_interceptors = tool_interceptors or []
        self.tool_name_prefix = tool_name_prefix

    @asynccontextmanager
    async def session(
        self,
        server_name: str,
        *,
        auto_initialize: bool = True,
    ) -> AsyncIterator[ClientSession]:
        """Connect to an MCP server and initialize a session.

        Args:
            server_name: Name to identify this server connection
            auto_initialize: Whether to automatically initialize the session

        Raises:
            ValueError: If the server name is not found in the connections

        Yields:
            An initialized `ClientSession`

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
        """Get a list of all tools from all connected servers.

        Args:
            server_name: Optional name of the server to get tools from.
                If `None`, all tools from all servers will be returned.

        !!! note

            A new session will be created for each tool call

        Returns:
            A list of LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools)

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
                tool_interceptors=self.tool_interceptors,
                tool_name_prefix=self.tool_name_prefix,
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
                    tool_interceptors=self.tool_interceptors,
                    tool_name_prefix=self.tool_name_prefix,
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
        """Get a prompt from a given MCP server."""
        async with self.session(server_name) as session:
            return await load_mcp_prompt(session, prompt_name, arguments=arguments)

    async def get_resources(
        self,
        server_name: str | None = None,
        *,
        uris: str | list[str] | None = None,
    ) -> list[Blob]:
        """Get resources from MCP server(s).

        Args:
            server_name: Optional name of the server to get resources from.
                If `None`, all resources from all servers will be returned.
            uris: Optional resource URI or list of URIs to load. If not provided,
                all resources will be loaded.

        Returns:
            A list of LangChain [Blob][langchain_core.documents.base.Blob] objects.

        """
        if server_name is not None:
            if server_name not in self.connections:
                msg = (
                    f"Couldn't find a server with name '{server_name}', "
                    f"expected one of '{list(self.connections.keys())}'"
                )
                raise ValueError(msg)
            async with self.session(server_name) as session:
                return await load_mcp_resources(session, uris=uris)

        async def _load_resources_from_server(name: str) -> list[Blob]:
            async with self.session(name) as session:
                return await load_mcp_resources(session, uris=uris)

        all_resources: list[Blob] = []
        load_tasks = [
            asyncio.create_task(_load_resources_from_server(name))
            for name in self.connections
        ]
        resources_list = await asyncio.gather(*load_tasks)
        for resources in resources_list:
            all_resources.extend(resources)
        return all_resources

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


class LongLivedMultiServerMCPClient(MultiServerMCPClient):
    """A `MultiServerMCPClient` variant that keeps sessions alive across tool calls.

    Use this class as an async context manager:

    ```python
    async with LongLivedMultiServerMCPClient(connections) as client:
        tools = await client.get_tools()
    ```

    In this mode, tools loaded by `get_tools()` will reuse long-lived server sessions.
    """

    def __init__(
        self,
        connections: dict[str, Connection] | None = None,
        *,
        callbacks: Callbacks | None = None,
        tool_interceptors: list[ToolCallInterceptor] | None = None,
        tool_name_prefix: bool = False,
    ) -> None:
        super().__init__(
            connections=connections,
            callbacks=callbacks,
            tool_interceptors=tool_interceptors,
            tool_name_prefix=tool_name_prefix,
        )
        self.sessions: dict[str, ClientSession] = {}
        self._session_stack: AsyncExitStack | None = None

    def _validate_server_name(self, server_name: str) -> None:
        if server_name not in self.connections:
            msg = (
                f"Couldn't find a server with name '{server_name}', "
                f"expected one of '{list(self.connections.keys())}'"
            )
            raise ValueError(msg)

    async def start(self) -> None:
        """Start long-lived sessions for all configured servers."""
        if self._session_stack is not None:
            return

        stack = AsyncExitStack()
        sessions: dict[str, ClientSession] = {}
        try:
            for server_name in self.connections:
                session = await stack.enter_async_context(
                    self.session(server_name, auto_initialize=True)
                )
                sessions[server_name] = session
        except Exception:
            await stack.aclose()
            raise

        self._session_stack = stack
        self.sessions = sessions

    async def stop(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        """Close all long-lived sessions."""
        if self._session_stack is not None:
            await self._session_stack.aclose()
            self._session_stack = None
        self.sessions.clear()

    @asynccontextmanager
    async def _session_for_server(self, server_name: str) -> AsyncIterator[ClientSession]:
        """Yield a persistent session when available, otherwise a short-lived one."""
        self._validate_server_name(server_name)
        persistent_session = self.sessions.get(server_name)
        if persistent_session is not None:
            yield persistent_session
            return

        async with self.session(server_name) as ephemeral_session:
            yield ephemeral_session

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        """Get tools, preferring long-lived sessions when available."""
        if server_name is not None:
            self._validate_server_name(server_name)

            session = self.sessions.get(server_name)
            if session is not None:
                return await load_mcp_tools(
                    session,
                    callbacks=self.callbacks,
                    server_name=server_name,
                    tool_interceptors=self.tool_interceptors,
                    tool_name_prefix=self.tool_name_prefix,
                )

            return await load_mcp_tools(
                None,
                connection=self.connections[server_name],
                callbacks=self.callbacks,
                server_name=server_name,
                tool_interceptors=self.tool_interceptors,
                tool_name_prefix=self.tool_name_prefix,
            )

        all_tools: list[BaseTool] = []
        load_mcp_tool_tasks = []
        for name, connection in self.connections.items():
            session = self.sessions.get(name)
            load_mcp_tool_task = asyncio.create_task(
                load_mcp_tools(
                    session,
                    connection=None if session is not None else connection,
                    callbacks=self.callbacks,
                    server_name=name,
                    tool_interceptors=self.tool_interceptors,
                    tool_name_prefix=self.tool_name_prefix,
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
        """Get a prompt, preferring long-lived sessions when available."""
        async with self._session_for_server(server_name) as session:
            return await load_mcp_prompt(session, prompt_name, arguments=arguments)

    async def get_resources(
        self,
        server_name: str | None = None,
        *,
        uris: str | list[str] | None = None,
    ) -> list[Blob]:
        """Get resources, preferring long-lived sessions when available."""
        if server_name is not None:
            async with self._session_for_server(server_name) as session:
                return await load_mcp_resources(session, uris=uris)

        async def _load_resources_from_server(name: str) -> list[Blob]:
            async with self._session_for_server(name) as session:
                return await load_mcp_resources(session, uris=uris)

        all_resources: list[Blob] = []
        load_tasks = [
            asyncio.create_task(_load_resources_from_server(name))
            for name in self.connections
        ]
        resources_list = await asyncio.gather(*load_tasks)
        for resources in resources_list:
            all_resources.extend(resources)
        return all_resources

    async def __aenter__(self) -> "LongLivedMultiServerMCPClient":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop(exc_type, exc_val, exc_tb)


__all__ = [
    "Callbacks",
    "LongLivedMultiServerMCPClient",
    "McpHttpClientFactory",
    "MultiServerMCPClient",
    "SSEConnection",
    "StdioConnection",
    "StreamableHttpConnection",
    "WebsocketConnection",
]
