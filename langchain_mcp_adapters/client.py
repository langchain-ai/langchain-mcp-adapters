import os
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, TypedDict, cast

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools

EncodingErrorHandler = Literal["strict", "ignore", "replace"]

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = "strict"

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5


class StdioConnection(TypedDict):
    transport: Literal["stdio"]

    command: str
    """The executable to run to start the server."""

    args: list[str]
    """Command line arguments to pass to the executable."""

    env: dict[str, str] | None
    """The environment to use when spawning the process."""

    cwd: str | Path | None
    """The working directory to use when spawning the process."""

    encoding: str
    """The text encoding used when sending/receiving messages to the server."""

    encoding_error_handler: EncodingErrorHandler
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values
    """

    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession"""


class SSEConnection(TypedDict):
    transport: Literal["sse"]

    url: str
    """The URL of the SSE endpoint to connect to."""

    headers: dict[str, Any] | None
    """HTTP headers to send to the SSE endpoint"""

    timeout: float
    """HTTP timeout"""

    sse_read_timeout: float
    """SSE read timeout"""

    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession"""


class WebsocketConnection(TypedDict):
    transport: Literal["websocket"]

    url: str
    """The URL of the Websocket endpoint to connect to."""

    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession"""


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools from them."""

    def __init__(
        self,
        connections: dict[str, StdioConnection | SSEConnection | WebsocketConnection] | None = None,
    ) -> None:
        """Initialize a MultiServerMCPClient with MCP servers connections.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                Each configuration can be a StdioConnection, SSEConnection or a WebsocketConnection.
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
        self.connections: dict[str, StdioConnection | SSEConnection | WebsocketConnection] = (
            connections or {}
        )
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
        *,
        transport: Literal["stdio", "sse", "websocket"] = "stdio",
        **kwargs,
    ) -> None:
        """Connect to an MCP server.

        This is a generic method that calls either of
        `connect_to_server_via_stdio` / `connect_to_server_via_sse` / `connect_to_server_via_websocket`
        based on the provided transport parameter.

        Args:
            server_name: Name to identify this server connection
            transport: Type of transport to use ("stdio" or "sse" or "websocket"), defaults to "stdio"
            **kwargs: Additional arguments to pass to the specific connection method

        Raises:
            ValueError: If transport is not recognized
            ValueError: If required parameters for the specified transport are missing
        """
        if transport == "sse":
            if "url" not in kwargs:
                raise ValueError("'url' parameter is required for SSE connection")
            await self.connect_to_server_via_sse(
                server_name,
                url=kwargs["url"],
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout", DEFAULT_HTTP_TIMEOUT),
                sse_read_timeout=kwargs.get("sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT),
                session_kwargs=kwargs.get("session_kwargs"),
            )
        elif transport == "stdio":
            if "command" not in kwargs:
                raise ValueError("'command' parameter is required for stdio connection")
            if "args" not in kwargs:
                raise ValueError("'args' parameter is required for stdio connection")
            await self.connect_to_server_via_stdio(
                server_name,
                command=kwargs["command"],
                args=kwargs["args"],
                env=kwargs.get("env"),
                encoding=kwargs.get("encoding", DEFAULT_ENCODING),
                encoding_error_handler=kwargs.get(
                    "encoding_error_handler", DEFAULT_ENCODING_ERROR_HANDLER
                ),
                session_kwargs=kwargs.get("session_kwargs"),
            )
        elif transport == "websocket":
            if "url" not in kwargs:
                raise ValueError("'url' parameter is required for Websocket connection")
            await self.connect_to_server_via_websocket(
                server_name,
                url=kwargs["url"],
                session_kwargs=kwargs.get("session_kwargs"),
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}. Must be 'stdio' or 'sse'")

    async def connect_to_server_via_stdio(
        self,
        server_name: str,
        *,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        encoding: str = DEFAULT_ENCODING,
        encoding_error_handler: Literal[
            "strict", "ignore", "replace"
        ] = DEFAULT_ENCODING_ERROR_HANDLER,
        session_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Connect to a specific MCP server using stdio

        Args:
            server_name: Name to identify this server connection
            command: Command to execute
            args: Arguments for the command
            env: Environment variables for the command
            encoding: Character encoding
            encoding_error_handler: How to handle encoding errors
            session_kwargs: Additional keyword arguments to pass to the ClientSession
        """
        # NOTE: execution commands (e.g., `uvx` / `npx`) require PATH envvar to be set.
        # To address this, we automatically inject existing PATH envvar into the `env` value,
        # if it's not already set.
        env = env or {}
        if "PATH" not in env:
            env["PATH"] = os.environ.get("PATH", "")

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
        session_kwargs = session_kwargs or {}
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(ClientSession(read, write, **session_kwargs)),
        )

        await self._initialize_session_and_load_tools(server_name, session)

    async def connect_to_server_via_sse(
        self,
        server_name: str,
        *,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
        session_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Connect to a specific MCP server using SSE

        Args:
            server_name: Name to identify this server connection
            url: URL of the SSE server
            headers: HTTP headers to send to the SSE endpoint
            timeout: HTTP timeout
            sse_read_timeout: SSE read timeout
            session_kwargs: Additional keyword arguments to pass to the ClientSession
        """
        # Create and store the connection
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url, headers, timeout, sse_read_timeout)
        )
        read, write = sse_transport
        session_kwargs = session_kwargs or {}
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(ClientSession(read, write, **session_kwargs)),
        )

        await self._initialize_session_and_load_tools(server_name, session)

    async def connect_to_server_via_websocket(
        self,
        server_name: str,
        *,
        url: str,
        session_kwargs: dict[str, Any] | None = None,
    ):
        """Connect to a specific MCP server using Websockets

        Args:
            server_name: Name to identify this server connection
            url: URL of the Websocket endpoint
            session_kwargs: Additional keyword arguments to pass to the ClientSession

        Raises:
            ImportError: If websockets package is not installed
        """
        try:
            from mcp.client.websocket import websocket_client
        except ImportError:
            raise ImportError(
                "Could not import websocket_client. ",
                "To use Websocket connections, please install the required dependency with: ",
                "'pip install mcp[ws]' or 'pip install websockets'",
            ) from None

        ws_transport = await self.exit_stack.enter_async_context(websocket_client(url))
        read, write = ws_transport
        session_kwargs = session_kwargs or {}
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(ClientSession(read, write, **session_kwargs)),
        )

        await self._initialize_session_and_load_tools(server_name, session)

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
