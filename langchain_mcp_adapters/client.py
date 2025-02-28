from contextlib import AsyncExitStack
from types import TracebackType
from typing import Literal, TypedDict, cast

from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER = "strict"


class StdioConnection(TypedDict):
    transport: Literal["stdio"]
    command: str
    args: list[str]
    env: dict[str, str] | None
    encoding: str
    encoding_error_handler: Literal["strict", "ignore", "replace"]


class SSEConnection(TypedDict):
    transport: Literal["sse"]
    url: str


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools from them."""

    def __init__(self, connections: dict[str, StdioConnection | SSEConnection] = None) -> None:
        self.connections = connections
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
        transport: Literal["stdio", "sse"] = "stdio",
        **kwargs,
    ) -> None:
        """Connect to an MCP server using either stdio or SSE.

        This is a generic method that calls either connect_to_server_via_stdio or connect_to_server_via_sse
        based on the provided transport parameter.

        Args:
            server_name: Name to identify this server connection
            transport: Type of transport to use ("stdio" or "sse"), defaults to "stdio"
            **kwargs: Additional arguments to pass to the specific connection method

        Raises:
            ValueError: If transport is not recognized
            ValueError: If required parameters for the specified transport are missing
        """
        if transport == "sse":
            if "url" not in kwargs:
                raise ValueError("'url' parameter is required for SSE connection")
            await self.connect_to_server_via_sse(server_name, url=kwargs["url"])
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
    ) -> None:
        """Connect to a specific MCP server using stdio

        Args:
            server_name: Name to identify this server connection
            command: Command to execute
            args: Arguments for the command
            env: Environment variables for the command
            encoding: Character encoding
            encoding_error_handler: How to handle encoding errors
        """
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

        await self._initialize_session_and_load_tools(server_name, session)

    async def connect_to_server_via_sse(
        self,
        server_name: str,
        *,
        url: str,
    ) -> None:
        """Connect to a specific MCP server using SSE

        Args:
            server_name: Name to identify this server connection
            url: URL of the SSE server
        """
        # Create and store the connection
        sse_transport = await self.exit_stack.enter_async_context(sse_client(url))
        read, write = sse_transport
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(ClientSession(read, write)),
        )

        await self._initialize_session_and_load_tools(server_name, session)

    def get_tools(self) -> list[BaseTool]:
        """Get a list of all tools from all connected servers."""
        all_tools: list[BaseTool] = []
        for server_tools in self.server_name_to_tools.values():
            all_tools.extend(server_tools)
        return all_tools

    async def __aenter__(self) -> "MultiServerMCPClient":
        try:
            connections = self.connections or {}
            for server_name, connection in connections.items():
                connection_dict = connection.copy()
                transport = connection_dict.pop("transport")
                if transport == "stdio":
                    await self.connect_to_server_via_stdio(server_name, **connection_dict)
                elif transport == "sse":
                    await self.connect_to_server_via_sse(server_name, **connection_dict)
                else:
                    raise ValueError(
                        f"Unsupported transport: {transport}. Must be 'stdio' or 'sse'"
                    )
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
