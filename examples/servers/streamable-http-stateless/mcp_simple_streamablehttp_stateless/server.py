"""Simple MCP server example using streamable HTTP transport.

This module demonstrates a basic MCP server implementation using streamable HTTP
transport with basic math operations (add and multiply).
"""

import contextlib
import logging
from collections.abc import AsyncIterator

import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)


@click.command()
@click.option("--port", default=3000, help="Port to listen on for HTTP")
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--json-response",
    is_flag=True,
    default=False,
    help="Enable JSON responses instead of SSE streams",
)
def main(
    port: int,
    log_level: str,
    json_response: bool,
) -> int:
    """Run the MCP server with streamable HTTP transport.

    Args:
        port: Port to listen on for HTTP requests.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_response: Whether to enable JSON responses instead of SSE streams.

    Returns:
        Exit code (0 for success).
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Server("mcp-streamable-http-stateless-demo")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool calls for math operations.

        Args:
            name: Name of the tool to call.
            arguments: Dictionary of arguments for the tool.

        Returns:
            List of content objects with the tool result.

        Raises:
            ValueError: If the tool name is not recognized.
        """
        if name == "add":
            return [
                types.TextContent(
                    type="text",
                    text=str(arguments["a"] + arguments["b"])
                )
            ]
        elif name == "multiply":
            return [
                types.TextContent(
                    type="text",
                    text=str(arguments["a"] * arguments["b"])
                )
            ]
        else:
            raise ValueError(f"Tool {name} not found")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List all available tools provided by this server.

        Returns:
            List of tool definitions for add and multiply operations.
        """
        return [
            types.Tool(
                name="add",
                description="Adds two numbers",
                inputSchema={
                    "type": "object",
                    "required": ["a", "b"],
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number to add",
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number to add",
                        },
                    },
                },
            ),
            types.Tool(
                name="multiply",
                description="Multiplies two numbers",
                inputSchema={
                    "type": "object",
                    "required": ["a", "b"],
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number to multiply",
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number to multiply",
                        },
                    },
                },
            )
        ]

    # Create the session manager with true stateless mode
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Handle streamable HTTP requests through the session manager.

        Args:
            scope: ASGI scope object.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager lifecycle.

        Args:
            app: The Starlette application instance.

        Yields:
            None during the application lifetime.
        """
        async with session_manager.run():
            logger.info("Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logger.info("Application shutting down...")

    # Create an ASGI application using the transport
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    import uvicorn

    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

    return 0
