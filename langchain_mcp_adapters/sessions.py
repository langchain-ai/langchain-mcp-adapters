"""Session management for different MCP transport types.

This module provides connection configurations and session management for various
MCP transport types including stdio, SSE, and Streamable HTTP.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client._probe import negotiate_auto
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.types import DiscoverResult, ServerCapabilities
from mcp.types.version import MODERN_PROTOCOL_VERSIONS
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from langchain_mcp_adapters.callbacks import _MCPCallbacks

logger = logging.getLogger(__name__)

_BRACED_VAR_RE = re.compile(r"\$\{([^}]+)\}")
"""Matches `${VAR}` style environment variable references."""


def _expand_env_vars(value: str) -> str:
    """Expand `${VAR}` references in *value* using the current environment.

    Only braced syntax is expanded; bare `$VAR` references are left untouched so
    that literal dollar signs in passwords or other values are never silently
    corrupted by an unrelated environment variable.

    Undefined variables are preserved as-is (e.g. `${MISSING}` stays
    `${MISSING}`).
    """
    return _BRACED_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


EncodingErrorHandler = Literal["strict", "ignore", "replace"]

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = "strict"

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5

DEFAULT_STREAMABLE_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT = timedelta(seconds=60 * 5)

# The `str` arm is forward-compat for revisions a newer SDK may add; the two
# literals are spelled out so editors complete them. Mirrors the SDK's own
# `mcp.client.client.ConnectMode`. Values are validated at use, not by the type.
ProtocolMode = Literal["auto", "legacy"] | str  # noqa: PYI051
"""How to negotiate the MCP protocol revision on a new session.

- `"auto"` (default): probe `server/discover` and adopt the newest mutually
  supported modern revision, falling back to the `initialize` handshake for
  servers that predate it. This is the only value that can reach 2026-07-28.
- `"legacy"`: force the `initialize` handshake. Caps negotiation at the newest
  handshake-era revision and never sends `server/discover`.
- A modern revision string (e.g. `"2026-07-28"`): adopt that revision directly
  without probing. Use this to pin against a server that could otherwise talk
  the connection down to a handshake-era revision.
"""

DEFAULT_PROTOCOL: ProtocolMode = "auto"
"""Negotiation policy used when a connection does not set `protocol`."""

_PROTOCOL_MODES: frozenset[str] = frozenset(
    {"auto", "legacy", *MODERN_PROTOCOL_VERSIONS}
)
"""Every accepted `protocol` value.

Validated eagerly so that a typo surfaces as an error rather than a silent
downgrade to a handshake-era revision.
"""


class McpHttpClientFactory(Protocol):
    """Protocol for creating httpx2.AsyncClient instances for MCP connections."""

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient:
        """Create an httpx2.AsyncClient instance.

        Args:
            headers: HTTP headers to include in requests.
            timeout: Request timeout configuration.
            auth: Authentication configuration.

        Returns:
            Configured httpx2.AsyncClient instance.
        """
        ...


class StdioConnection(TypedDict):
    """Configuration for stdio transport connections to MCP servers."""

    transport: Literal["stdio"]

    command: str
    """The executable to run to start the server."""

    args: list[str]
    """Command line arguments to pass to the executable."""

    env: NotRequired[dict[str, str] | None]
    """The environment to use when spawning the process.

    If not specified or set to None, a subset of the default environment
    variables from the current process will be used.

    Please refer to the MCP SDK documentation for details on which
    environment variables are included by default. The behavior
    varies by operating system.

    https://github.com/modelcontextprotocol/python-sdk/blob/c47c767ff437ee88a19e6b9001e2472cb6f7d5ed/src/mcp/client/stdio/__init__.py#L51
    """

    cwd: NotRequired[str | Path | None]
    """The working directory to use when spawning the process."""

    encoding: NotRequired[str]
    """The text encoding used when sending/receiving messages to the server.

    Default is 'utf-8'.
    """

    encoding_error_handler: NotRequired[EncodingErrorHandler]
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.

    Default is 'strict', which raises an error on encoding/decoding errors.
    """

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession."""

    protocol: NotRequired[ProtocolMode]
    """Which MCP protocol revision to negotiate.

    `"auto"` (the default) probes `server/discover` and falls back to the
    `initialize` handshake; `"legacy"` forces the handshake; a modern revision
    string such as `"2026-07-28"` adopts that revision without probing. See
    [`ProtocolMode`][langchain_mcp_adapters.sessions.ProtocolMode].
    """


class SSEConnection(TypedDict):
    """Configuration for Server-Sent Events (SSE) transport connections to MCP."""

    transport: Literal["sse"]

    url: str
    """The URL of the SSE endpoint to connect to."""

    headers: NotRequired[dict[str, Any] | None]
    """HTTP headers to send to the SSE endpoint."""

    timeout: NotRequired[float]
    """HTTP timeout.

    Default is 5 seconds. If the server takes longer to respond,
    you can increase this value.
    """

    sse_read_timeout: NotRequired[float]
    """SSE read timeout.

    Default is 300 seconds (5 minutes). This is how long the client will
    wait for a new event before disconnecting.
    """

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession."""

    httpx_client_factory: NotRequired[McpHttpClientFactory | None]
    """Custom factory for httpx2.AsyncClient (optional)."""

    auth: NotRequired[httpx2.Auth]
    """Optional authentication for the HTTP client."""

    protocol: NotRequired[ProtocolMode]
    """Which MCP protocol revision to negotiate.

    `"auto"` (the default) probes `server/discover` and falls back to the
    `initialize` handshake; `"legacy"` forces the handshake; a modern revision
    string such as `"2026-07-28"` adopts that revision without probing. See
    [`ProtocolMode`][langchain_mcp_adapters.sessions.ProtocolMode].
    """


class StreamableHttpConnection(TypedDict):
    """Connection configuration for Streamable HTTP transport."""

    transport: Literal["streamable_http"]

    url: str
    """The URL of the endpoint to connect to."""

    headers: NotRequired[dict[str, Any] | None]
    """HTTP headers to send to the endpoint."""

    timeout: NotRequired[float | timedelta]
    """HTTP timeout."""

    sse_read_timeout: NotRequired[float | timedelta]
    """How long (in seconds) the client will wait for a new event before disconnecting.
    All other HTTP operations are controlled by `timeout`."""

    terminate_on_close: NotRequired[bool]
    """Whether to terminate the session on close."""

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession."""

    httpx_client_factory: NotRequired[McpHttpClientFactory | None]
    """Custom factory for httpx2.AsyncClient (optional)."""

    auth: NotRequired[httpx2.Auth]
    """Optional authentication for the HTTP client."""

    protocol: NotRequired[ProtocolMode]
    """Which MCP protocol revision to negotiate.

    `"auto"` (the default) probes `server/discover` and falls back to the
    `initialize` handshake; `"legacy"` forces the handshake; a modern revision
    string such as `"2026-07-28"` adopts that revision without probing. See
    [`ProtocolMode`][langchain_mcp_adapters.sessions.ProtocolMode].
    """


class WebsocketConnection(TypedDict):
    """Deprecated. WebSocket is not supported as of langchain-mcp-adapters 0.4.0.

    MCP SDK v2 removed its WebSocket client and server modules; WebSocket was
    never an MCP-spec transport. This `TypedDict` remains importable so existing
    imports do not break, but it is no longer part of the `Connection` union and
    passing `transport="websocket"` to `create_session` raises `ValueError`.

    Use Streamable HTTP (`transport="http"`) instead.
    """

    transport: Literal["websocket"]

    url: str
    """The URL of the Websocket endpoint to connect to."""

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession"""


Connection = StdioConnection | SSEConnection | StreamableHttpConnection

WEBSOCKET_REMOVED_ERROR = (
    "The 'websocket' transport is not supported as of langchain-mcp-adapters "
    "0.4.0. MCP SDK v2 removed its WebSocket client, and WebSocket was never an "
    "MCP-spec transport. Use Streamable HTTP instead:\n"
    '    {"url": "http://localhost:8000/mcp", "transport": "http"}\n'
    "If your server only speaks WebSocket, it needs to expose a Streamable HTTP "
    "endpoint to be reachable from this version."
)


def _validate_protocol(protocol: ProtocolMode) -> ProtocolMode:
    """Return `protocol` if it names a known policy.

    Raises:
        ValueError: The value is not a recognized negotiation policy. Rejected
            eagerly rather than passed through, so that a typo surfaces as an
            error instead of silently leaving the connection on a handshake-era
            revision.
    """
    if protocol not in _PROTOCOL_MODES:
        accepted = ", ".join(repr(mode) for mode in sorted(_PROTOCOL_MODES))
        msg = (
            f"Unsupported protocol {protocol!r}. Must be one of: {accepted}. "
            "Use 'auto' to negotiate the newest revision the server supports, "
            "'legacy' to force the initialize handshake, or a modern revision "
            "string to pin it."
        )
        raise ValueError(msg)
    return protocol


def resolve_protocol(
    connection: Connection | None,
    *,
    default: ProtocolMode = DEFAULT_PROTOCOL,
) -> ProtocolMode:
    """Read and validate the `protocol` key of a connection config.

    Args:
        connection: Connection config to read `protocol` from. `None` and a
            config without the key both fall back to `default`.
        default: Policy to use when the connection does not specify one. A
            connection's own `protocol` always wins over this.

    Returns:
        A validated [`ProtocolMode`][langchain_mcp_adapters.sessions.ProtocolMode].

    Raises:
        ValueError: Either value is not a recognized negotiation policy.
    """
    _validate_protocol(default)
    return _validate_protocol((connection or {}).get("protocol", default))


async def negotiate_protocol(
    session: ClientSession,
    *,
    protocol: ProtocolMode = DEFAULT_PROTOCOL,
) -> None:
    """Negotiate the MCP protocol revision on a freshly created session.

    Replaces a bare `session.initialize()` call. `initialize()` only ever
    negotiates a handshake-era revision, so a client that calls it directly can
    never reach 2026-07-28 even against a server that speaks it.

    Args:
        session: An un-negotiated session from
            [`create_session`][langchain_mcp_adapters.sessions.create_session].
        protocol: Negotiation policy. See
            [`ProtocolMode`][langchain_mcp_adapters.sessions.ProtocolMode].

    Raises:
        ValueError: `protocol` is not a recognized policy.
        McpError: The server is modern-only and shares no revision with this
            client, or the handshake failed.
    """
    protocol = _validate_protocol(protocol)
    if protocol == "legacy":
        await session.initialize()
    elif protocol == "auto":
        # The SDK owns this policy: probe `server/discover`, and fall back to
        # `initialize` on anything that is not positive evidence of a modern
        # peer. Reimplementing the denylist here would drift from the SDK's.
        await negotiate_auto(session)
    else:
        # A pinned modern revision skips the probe entirely. `supported_versions`
        # is what the client is asserting, not something the server told us.
        session.adopt(
            DiscoverResult(
                supported_versions=[protocol],
                capabilities=ServerCapabilities(),
                result_type="complete",
                ttl_ms=0,
                cache_scope="public",
            )
        )


@asynccontextmanager
async def _create_stdio_session(
    *,
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    encoding: str = DEFAULT_ENCODING,
    encoding_error_handler: Literal[
        "strict", "ignore", "replace"
    ] = DEFAULT_ENCODING_ERROR_HANDLER,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using stdio.

    Args:
        command: Command to execute.
        args: Arguments for the command.
        env: Environment variables for the command. Values containing
            `${VAR}` references are expanded from the current environment. Only
            braced syntax is supported; bare `${VAR}` is **not** expanded so
            that literal dollar signs in passwords or other values are never
            silently corrupted. Only values (not keys) are expanded;
            `${command}` and `${args}` are passed through unchanged.

            If not specified, inherits a subset of the current environment.

            The details are implemented in the MCP sdk.
        cwd: Working directory for the command.
        encoding: Character encoding.
        encoding_error_handler: How to handle encoding errors.
        session_kwargs: Additional keyword arguments to pass to the ClientSession.

    Yields:
        An initialized ClientSession.
    """
    resolved_env = (
        {k: _expand_env_vars(v) for k, v in env.items()} if env is not None else None
    )
    if resolved_env is not None:
        for k, v in resolved_env.items():
            if _BRACED_VAR_RE.search(v):
                logger.warning(
                    "env[%r] contains unexpanded variable reference: %r", k, v
                )
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=resolved_env,
        cwd=cwd,
        encoding=encoding,
        encoding_error_handler=encoding_error_handler,
    )

    # Create and store the connection
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write, **(session_kwargs or {})) as session,
    ):
        yield session


@asynccontextmanager
async def _create_sse_session(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
    session_kwargs: dict[str, Any] | None = None,
    httpx_client_factory: McpHttpClientFactory | None = None,
    auth: httpx2.Auth | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using SSE.

    Args:
        url: URL of the SSE server.
        headers: HTTP headers to send to the SSE endpoint.
        timeout: HTTP timeout.
        sse_read_timeout: SSE read timeout.
        session_kwargs: Additional keyword arguments to pass to the ClientSession.
        httpx_client_factory: Custom factory for httpx2.AsyncClient (optional).
        auth: Authentication for the HTTP client.

    Yields:
        An initialized ClientSession.
    """
    # Create and store the connection
    kwargs = {}
    if httpx_client_factory is not None:
        kwargs["httpx_client_factory"] = httpx_client_factory

    async with (
        sse_client(url, headers, timeout, sse_read_timeout, auth=auth, **kwargs) as (
            read,
            write,
        ),
        ClientSession(read, write, **(session_kwargs or {})) as session,
    ):
        yield session


@asynccontextmanager
async def _create_streamable_http_session(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float | timedelta = DEFAULT_STREAMABLE_HTTP_TIMEOUT,
    sse_read_timeout: float | timedelta = DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT,
    terminate_on_close: bool = True,
    session_kwargs: dict[str, Any] | None = None,
    httpx_client_factory: McpHttpClientFactory | None = None,
    auth: httpx2.Auth | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using Streamable HTTP.

    Args:
        url: URL of the endpoint to connect to.
        headers: HTTP headers to send to the endpoint.
        timeout: HTTP timeout.
        sse_read_timeout: How long the client will wait for a new event before
            disconnecting.
        terminate_on_close: Whether to terminate the session on close.
        session_kwargs: Additional keyword arguments to pass to the ClientSession.
        httpx_client_factory: Custom factory for httpx2.AsyncClient (optional).
        auth: Authentication for the HTTP client.

    Yields:
        An initialized ClientSession.
    """
    # Create and store the connection
    client_factory = httpx_client_factory or create_mcp_http_client
    timeout_seconds = (
        timeout.total_seconds() if isinstance(timeout, timedelta) else timeout
    )
    sse_read_timeout_seconds = (
        sse_read_timeout.total_seconds()
        if isinstance(sse_read_timeout, timedelta)
        else sse_read_timeout
    )
    client = client_factory(
        headers=headers,
        timeout=httpx2.Timeout(timeout_seconds, read=sse_read_timeout_seconds),
        auth=auth,
    )

    async with (
        client,
        streamable_http_client(
            url,
            http_client=client,
            terminate_on_close=terminate_on_close,
        ) as (read, write),
        ClientSession(read, write, **(session_kwargs or {})) as session,
    ):
        yield session


@asynccontextmanager
async def create_session(
    connection: Connection, *, mcp_callbacks: _MCPCallbacks | None = None
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server.

    Args:
        connection: Connection config to use to connect to the server
        mcp_callbacks: mcp sdk compatible callbacks to use for the ClientSession

    Raises:
        ValueError: If transport is not recognized
        ValueError: If required parameters for the specified transport are missing

    Yields:
        A ClientSession
    """
    if "transport" not in connection:
        msg = (
            "Configuration error: Missing 'transport' key in server configuration. "
            "Each server must include 'transport' with one of: "
            "'stdio', 'sse', 'http'. "
            "Please refer to the langchain-mcp-adapters documentation for more details."
        )
        raise ValueError(msg)

    transport = connection["transport"]
    # `protocol` is negotiation policy, not transport configuration: it is
    # consumed by `negotiate_protocol` and must not reach a transport kwarg.
    params = {k: v for k, v in connection.items() if k not in ("transport", "protocol")}

    if mcp_callbacks is not None:
        params["session_kwargs"] = dict(params.get("session_kwargs") or {})
        if mcp_callbacks.logging_callback is not None:
            params["session_kwargs"]["logging_callback"] = (
                mcp_callbacks.logging_callback
            )
            # SEP-2577: a 2026-07-28 server emits `notifications/message` only
            # for requests that opt in via `_meta`, and only at or above the
            # requested level. Without this a registered logging callback goes
            # silent the moment a connection negotiates a modern revision.
            params["session_kwargs"].setdefault("log_level", mcp_callbacks.log_level)
        if mcp_callbacks.elicitation_callback is not None:
            params["session_kwargs"]["elicitation_callback"] = (
                mcp_callbacks.elicitation_callback
            )

    if transport == "sse":
        if "url" not in params:
            msg = "'url' parameter is required for SSE connection"
            raise ValueError(msg)
        async with _create_sse_session(**params) as session:
            yield session
    elif transport in {"streamable_http", "streamable-http", "http"}:
        if "url" not in params:
            msg = "'url' parameter is required for Streamable HTTP connection"
            raise ValueError(msg)
        async with _create_streamable_http_session(**params) as session:
            yield session
    elif transport == "stdio":
        if "command" not in params:
            msg = "'command' parameter is required for stdio connection"
            raise ValueError(msg)
        if "args" not in params:
            msg = "'args' parameter is required for stdio connection"
            raise ValueError(msg)
        async with _create_stdio_session(**params) as session:
            yield session
    elif transport == "websocket":
        raise ValueError(WEBSOCKET_REMOVED_ERROR)
    else:
        msg = (
            f"Unsupported transport: {transport}. "
            f"Must be one of: 'stdio', 'sse', 'http'"
        )
        raise ValueError(msg)
