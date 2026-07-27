"""Server info adapter for retrieving MCP server metadata.

This module provides functionality to retrieve server information
from the MCP initialize handshake, including server instructions,
capabilities, and implementation details.
"""

from mcp import ClientSession
from mcp.types import InitializeResult

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks, _MCPCallbacks
from langchain_mcp_adapters.sessions import Connection, create_session

ALREADY_INITIALIZED_ERROR = (
    "The provided ClientSession has already been initialized. "
    "load_mcp_server_info() performs the initialize handshake itself, so it needs a "
    "session that has not been initialized yet; re-initializing a live session "
    "violates the MCP protocol, resets the server's initialization state, and can "
    "cause concurrent requests on that session to fail. Either pass an uninitialized "
    "session (e.g. MultiServerMCPClient.session(name, auto_initialize=False)), or "
    "omit `session` and pass `connection` to have a temporary session created."
)


async def load_mcp_server_info(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
    server_name: str | None = None,
) -> InitializeResult:
    """Load server info from the MCP initialize handshake.

    Returns the full `InitializeResult` from the MCP protocol, which includes
    server instructions, capabilities, implementation details, and protocol
    version.

    !!! note

        Unlike [`load_mcp_tools`][langchain_mcp_adapters.tools.load_mcp_tools],
        [`load_mcp_prompt`][langchain_mcp_adapters.prompts.load_mcp_prompt] and
        [`load_mcp_resources`][langchain_mcp_adapters.resources.load_mcp_resources],
        which expect an already-initialized session, this function performs the
        `initialize()` handshake itself and therefore requires a session that has
        *not* been initialized yet. `MultiServerMCPClient.session()` initializes
        by default, so pass `auto_initialize=False` when obtaining a session for
        this function. Passing an already-initialized session raises `ValueError`
        rather than re-running the handshake.

        There is no way to recover an `InitializeResult` from a session that is
        already initialized: the MCP SDK caches only `capabilities` and discards
        `instructions`, `serverInfo` and `protocolVersion`.

    Args:
        session: An MCP client session that has **not** been initialized yet.
            If provided, this function calls `initialize()` on it. If `None`, a
            `connection` must be provided and a temporary session will be
            created automatically.
        connection: Connection config to create a new session if `session` is
            `None`.
        callbacks: Optional `Callbacks` for handling notifications and events.
            Only applied to sessions created from `connection`; ignored when
            `session` is provided, since the caller owns that session's
            callbacks.
        server_name: Name of the server, used for callback context. Ignored when
            `session` is provided, for the same reason as `callbacks`.

    Returns:
        The `InitializeResult` from the MCP server. Most callers want
            `.instructions` and `.serverInfo`; see `mcp.types.InitializeResult`
            for the full field set, which servers may extend with additional
            fields.

    Raises:
        ValueError: If neither `session` nor `connection` is provided, if
            `session` has already been initialized, or if `connection` is
            missing `transport` or the parameters required by its transport.
        RuntimeError: If the server negotiates an unsupported MCP protocol
            version, or if the session closes without completing the handshake.

    """
    if session is not None:
        if session.get_server_capabilities() is not None:
            raise ValueError(ALREADY_INITIALIZED_ERROR)
        return await session.initialize()

    if connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    mcp_callbacks = (
        callbacks.to_mcp_format(context=CallbackContext(server_name=server_name))
        if callbacks is not None
        else _MCPCallbacks()
    )

    result: InitializeResult | None = None
    captured_exception: BaseException | None = None
    async with create_session(connection, mcp_callbacks=mcp_callbacks) as new_session:
        try:
            result = await new_session.initialize()
        except Exception as e:  # noqa: BLE001
            # Capture the exception to re-raise outside the context manager, which
            # may otherwise suppress it. Mirrors the work-around in `tools.py` for
            # an MCP SDK issue that swallows exceptions on client disconnect.
            captured_exception = e

    if captured_exception is not None:
        raise captured_exception

    if result is None:
        # Reachable only if the context manager suppressed an exception without
        # `initialize()` returning, which would otherwise return `None` in
        # violation of this function's return type.
        msg = (
            f"The MCP initialize handshake with server '{server_name or '[unknown server]'}' "
            "produced no result and raised no error; the session was closed by the "
            "transport. Check that the server is running and speaking MCP on the "
            "configured endpoint."
        )
        raise RuntimeError(msg)

    return result
