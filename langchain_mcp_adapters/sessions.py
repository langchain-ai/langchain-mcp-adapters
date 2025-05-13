import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Literal, TypedDict

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

EncodingErrorHandler = Literal["strict", "ignore", "replace"]

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = "strict"

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5

DEFAULT_STREAMABLE_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT = timedelta(seconds=60 * 5)


class BaseConnection(TypedDict):
    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession"""


class StdioConnection(BaseConnection):
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


class SSEConnection(BaseConnection):
    transport: Literal["sse"]

    url: str
    """The URL of the SSE endpoint to connect to."""

    headers: dict[str, Any] | None
    """HTTP headers to send to the SSE endpoint"""

    timeout: float
    """HTTP timeout"""

    sse_read_timeout: float
    """SSE read timeout"""


class StreamableHttpConnection(BaseConnection):
    transport: Literal["streamable_http"]

    url: str
    """The URL of the endpoint to connect to."""

    headers: dict[str, Any] | None
    """HTTP headers to send to the endpoint."""

    timeout: timedelta
    """HTTP timeout."""

    sse_read_timeout: timedelta
    """How long (in seconds) the client will wait for a new event before disconnecting.
    All other HTTP operations are controlled by `timeout`."""


class WebsocketConnection(BaseConnection):
    transport: Literal["websocket"]

    url: str
    """The URL of the Websocket endpoint to connect to."""


Connection = StdioConnection | SSEConnection | StreamableHttpConnection | WebsocketConnection


@asynccontextmanager
async def _connect_to_server_via_stdio(
    *,
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    encoding: str = DEFAULT_ENCODING,
    encoding_error_handler: Literal["strict", "ignore", "replace"] = DEFAULT_ENCODING_ERROR_HANDLER,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect to a specific MCP server using stdio

    Args:
        command: Command to execute
        args: Arguments for the command
        env: Environment variables for the command
        cwd: Working directory for the command
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
        cwd=cwd,
        encoding=encoding,
        encoding_error_handler=encoding_error_handler,
    )

    # Create and store the connection
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, **(session_kwargs or {})) as session:
            yield session


@asynccontextmanager
async def _connect_to_server_via_sse(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect to a specific MCP server using SSE

    Args:
        url: URL of the SSE server
        headers: HTTP headers to send to the SSE endpoint
        timeout: HTTP timeout
        sse_read_timeout: SSE read timeout
        session_kwargs: Additional keyword arguments to pass to the ClientSession
    """
    # Create and store the connection
    async with sse_client(url, headers, timeout, sse_read_timeout) as (read, write):
        async with ClientSession(read, write, **(session_kwargs or {})) as session:
            yield session


@asynccontextmanager
async def _connect_to_server_via_streamable_http(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: timedelta = DEFAULT_STREAMABLE_HTTP_TIMEOUT,
    sse_read_timeout: timedelta = DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect to a specific MCP server using Streamable HTTP

    Args:
        url: URL of the endpoint to connect to
        headers: HTTP headers to send to the endpoint
        timeout: HTTP timeout
        sse_read_timeout: How long (in seconds) the client will wait for a new event before disconnecting.
        session_kwargs: Additional keyword arguments to pass to the ClientSession
    """
    # Create and store the connection
    async with streamablehttp_client(url, headers, timeout, sse_read_timeout) as (read, write, _):
        async with ClientSession(read, write, **(session_kwargs or {})) as session:
            yield session


@asynccontextmanager
async def _connect_to_server_via_websocket(
    *,
    url: str,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect to a specific MCP server using Websockets

    Args:
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

    async with websocket_client(url) as (read, write):
        async with ClientSession(read, write, **(session_kwargs or {})) as session:
            yield session


@asynccontextmanager
async def connect_to_server(
    connection: Connection,
) -> AsyncIterator[ClientSession]:
    """Connect to an MCP server.

    Args:
        connection: Connection config to use to connect to the server

    Raises:
        ValueError: If transport is not recognized
        ValueError: If required parameters for the specified transport are missing

    Yields:
        A ClientSession
    """
    transport = connection["transport"]
    if transport == "sse":
        if "url" not in connection:
            raise ValueError("'url' parameter is required for SSE connection")
        async with _connect_to_server_via_sse(
            url=connection["url"],
            headers=connection.get("headers"),
            timeout=connection.get("timeout", DEFAULT_HTTP_TIMEOUT),
            sse_read_timeout=connection.get("sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT),
            session_kwargs=connection.get("session_kwargs"),
        ) as session:
            yield session
    elif transport == "streamable_http":
        if "url" not in connection:
            raise ValueError("'url' parameter is required for Streamable HTTP connection")
        async with _connect_to_server_via_streamable_http(
            url=connection["url"],
            headers=connection.get("headers"),
            timeout=connection.get("timeout", DEFAULT_STREAMABLE_HTTP_TIMEOUT),
            sse_read_timeout=connection.get(
                "sse_read_timeout", DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT
            ),
            session_kwargs=connection.get("session_kwargs"),
        ) as session:
            yield session
    elif transport == "stdio":
        if "command" not in connection:
            raise ValueError("'command' parameter is required for stdio connection")
        if "args" not in connection:
            raise ValueError("'args' parameter is required for stdio connection")
        async with _connect_to_server_via_stdio(
            command=connection["command"],
            args=connection["args"],
            env=connection.get("env"),
            cwd=connection.get("cwd"),
            encoding=connection.get("encoding", DEFAULT_ENCODING),
            encoding_error_handler=connection.get(
                "encoding_error_handler", DEFAULT_ENCODING_ERROR_HANDLER
            ),
            session_kwargs=connection.get("session_kwargs"),
        ) as session:
            yield session
    elif transport == "websocket":
        if "url" not in connection:
            raise ValueError("'url' parameter is required for Websocket connection")
        async with _connect_to_server_via_websocket(
            url=connection["url"],
            session_kwargs=connection.get("session_kwargs"),
        ) as session:
            yield session
    else:
        raise ValueError(
            f"Unsupported transport: {transport}. Must be one of: 'stdio', 'sse', 'websocket', 'streamable_http'"
        )
