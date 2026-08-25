"""Server info adapter for retrieving MCP server metadata.

This module retrieves the metadata a server reports when a connection is
established: its implementation identity, capabilities, instructions, and the
protocol revision that was negotiated.

Where that metadata comes from depends on the revision. Handshake-era
connections carry it in the `initialize` response
([`InitializeResult`][mcp.types.InitializeResult]); 2026-07-28 connections have
no `initialize` step at all and report it through `server/discover`
([`DiscoverResult`][mcp.types.DiscoverResult]) instead.
[`MCPServerInfo`][langchain_mcp_adapters.server_info.MCPServerInfo] normalizes
the two so callers do not have to branch on the era.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp.types import DiscoverResult, InitializeResult

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks, _MCPCallbacks
from langchain_mcp_adapters.sessions import (
    DEFAULT_PROTOCOL,
    Connection,
    ProtocolMode,
    create_session,
    negotiate_protocol,
    resolve_protocol,
)

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Implementation, ServerCapabilities

ALREADY_INITIALIZED_ERROR = (
    "The provided ClientSession has already been initialized. "
    "load_mcp_server_info() negotiates the connection itself, so it needs a "
    "session that has not been negotiated yet; re-negotiating a live session "
    "violates the MCP protocol, resets the server's initialization state, and can "
    "cause concurrent requests on that session to fail. Either pass an uninitialized "
    "session (e.g. MultiServerMCPClient.session(name, auto_initialize=False)), or "
    "omit `session` and pass `connection` to have a temporary session created."
)


@dataclass(frozen=True)
class MCPServerInfo:
    """Metadata a server reported when the connection was established.

    Normalizes the handshake-era `InitializeResult` and the 2026-07-28
    `DiscoverResult` into one shape. Reach for `raw` when you need a field
    specific to one of them.
    """

    protocol_version: str
    """The revision that was negotiated, e.g. `"2025-11-25"` or `"2026-07-28"`."""

    capabilities: ServerCapabilities
    """What the server advertises support for."""

    server_info: Implementation | None
    """The server's name and version.

    Always present on a handshake-era connection, where `InitializeResult`
    requires it. Optional on 2026-07-28, where identifying itself is a
    display-only `_meta` stamp: `None` means the server did not.
    """

    instructions: str | None
    """Free-text usage guidance from the server, if it offered any.

    Server-controlled text. Treat it the way you would any other untrusted
    model input before putting it in a prompt.
    """

    raw: InitializeResult | DiscoverResult
    """The underlying SDK result, for fields this dataclass does not surface."""

    @property
    def is_modern(self) -> bool:
        """Whether the connection negotiated a per-request (2026-07-28+) revision."""
        return isinstance(self.raw, DiscoverResult)

    @classmethod
    def from_session(cls, session: ClientSession) -> MCPServerInfo:
        """Read the negotiated metadata off a session.

        Raises:
            RuntimeError: The session has not negotiated a connection yet.
        """
        raw: InitializeResult | DiscoverResult | None = (
            session.initialize_result or session.discover_result
        )
        if raw is None or session.protocol_version is None:
            msg = (
                "The session has not negotiated a protocol revision yet; "
                "call negotiate_protocol() before reading server info."
            )
            raise RuntimeError(msg)
        capabilities = session.server_capabilities
        if capabilities is None:  # pragma: no cover - set by every adopt() path
            msg = "The session negotiated a revision but reported no capabilities."
            raise RuntimeError(msg)
        return cls(
            protocol_version=session.protocol_version,
            capabilities=capabilities,
            server_info=session.server_info,
            instructions=session.instructions,
            raw=raw,
        )


async def load_mcp_server_info(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
    server_name: str | None = None,
    protocol: ProtocolMode | None = None,
) -> MCPServerInfo:
    """Load the metadata a server reports when the connection is established.

    !!! note

        Unlike [`load_mcp_tools`][langchain_mcp_adapters.tools.load_mcp_tools],
        [`load_mcp_prompt`][langchain_mcp_adapters.prompts.load_mcp_prompt] and
        [`load_mcp_resources`][langchain_mcp_adapters.resources.load_mcp_resources],
        which expect an already-negotiated session, this function negotiates the
        connection itself and therefore requires a session that has *not* been
        negotiated yet. `MultiServerMCPClient.session()` negotiates by default,
        so pass `auto_initialize=False` when obtaining a session for this
        function. Passing an already-negotiated session raises `ValueError`
        rather than re-running negotiation.

    Args:
        session: An MCP client session that has **not** been negotiated yet.
            If `None`, a `connection` must be provided and a temporary session
            will be created automatically.
        connection: Connection config to create a new session if `session` is
            `None`.
        callbacks: Optional `Callbacks` for handling notifications and events.
            Only applied to sessions created from `connection`; ignored when
            `session` is provided, since the caller owns that session's
            callbacks.
        server_name: Name of the server, used for callback context. Ignored when
            `session` is provided, for the same reason as `callbacks`.
        protocol: Negotiation policy. Takes precedence over the connection's
            own `protocol` key; when omitted, that key is used, falling back to
            `"auto"`. See
            [`ProtocolMode`][langchain_mcp_adapters.sessions.ProtocolMode].

    Returns:
        The server's reported metadata.

    Raises:
        ValueError: If neither `session` nor `connection` is provided, if
            `session` has already been negotiated, if `connection` is missing
            `transport` or the parameters required by its transport, or if
            `protocol` is not a recognized policy.
        RuntimeError: If the server negotiates an unsupported MCP protocol
            revision, or if the session closes without completing negotiation.

    """
    if session is not None:
        if session.server_capabilities is not None:
            raise ValueError(ALREADY_INITIALIZED_ERROR)
        await negotiate_protocol(
            session, protocol=protocol if protocol is not None else DEFAULT_PROTOCOL
        )
        return MCPServerInfo.from_session(session)

    if connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    resolved_protocol = (
        protocol if protocol is not None else resolve_protocol(connection)
    )

    mcp_callbacks = (
        callbacks.to_mcp_format(context=CallbackContext(server_name=server_name))
        if callbacks is not None
        else _MCPCallbacks()
    )

    result: MCPServerInfo | None = None
    captured_exception: BaseException | None = None
    async with create_session(connection, mcp_callbacks=mcp_callbacks) as new_session:
        try:
            await negotiate_protocol(new_session, protocol=resolved_protocol)
            result = MCPServerInfo.from_session(new_session)
        except Exception as e:  # noqa: BLE001
            # Capture the exception to re-raise outside the context manager, which
            # may otherwise suppress it. Mirrors the work-around in `tools.py` for
            # an MCP SDK issue that swallows exceptions on client disconnect.
            captured_exception = e

    if captured_exception is not None:
        raise captured_exception

    if result is None:
        # Reachable only if the context manager suppressed an exception without
        # negotiation returning, which would otherwise return `None` in
        # violation of this function's return type.
        msg = (
            f"Negotiating an MCP connection with server "
            f"'{server_name or '[unknown server]'}' produced no result and raised "
            "no error; the session was closed by the transport. Check that the "
            "server is running and speaking MCP on the configured endpoint."
        )
        raise RuntimeError(msg)

    return result


__all__ = ["ALREADY_INITIALIZED_ERROR", "MCPServerInfo", "load_mcp_server_info"]
