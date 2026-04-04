"""Server info adapter for retrieving MCP server metadata.

This module provides functionality to retrieve server information
from the MCP initialize handshake, including server instructions,
capabilities, and implementation details.
"""

from mcp import ClientSession
from mcp.types import InitializeResult

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks, _MCPCallbacks
from langchain_mcp_adapters.sessions import Connection, create_session


async def load_mcp_server_info(
    session: ClientSession | None = None,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
    server_name: str | None = None,
) -> InitializeResult:
    """Load server info from the MCP initialize handshake.

    Returns the full `InitializeResult` from the MCP protocol, which includes
    server instructions, capabilities, implementation details, and protocol
    version.

    Args:
        session: An **uninitialized** MCP client session. If provided, this
            function will call ``initialize()`` on it. If ``None``, a
            ``connection`` must be provided and a temporary session will be
            created automatically.
        connection: Connection config to create a new session if ``session`` is
            ``None``.
        callbacks: Optional ``Callbacks`` for handling notifications and events.
        server_name: Name of the server (used for callback context).

    Returns:
        The ``InitializeResult`` from the MCP server, containing:
            - ``instructions``: Optional server instructions for the LLM.
            - ``serverInfo``: Server implementation details (name, version).
            - ``capabilities``: Server capabilities.
            - ``protocolVersion``: MCP protocol version.

    Raises:
        ValueError: If neither ``session`` nor ``connection`` is provided.

    """
    if session is not None:
        return await session.initialize()

    if connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    mcp_callbacks = (
        callbacks.to_mcp_format(context=CallbackContext(server_name=server_name))
        if callbacks is not None
        else _MCPCallbacks()
    )

    async with create_session(
        connection, mcp_callbacks=mcp_callbacks
    ) as new_session:
        return await new_session.initialize()
