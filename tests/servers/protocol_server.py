"""Server used by the protocol-negotiation tests.

Deliberately plain: it speaks whatever revision the client negotiates, so the
tests can assert what the *client* chose rather than what the server forced.
"""

from mcp.server.mcpserver import Context, MCPServer


def create_protocol_server() -> MCPServer:
    server = MCPServer("protocol-server", instructions="Protocol test server.")

    @server.tool()
    async def echo(text: str) -> str:
        """Echo the input back."""
        return text

    @server.tool()
    async def noisy(text: str, ctx: Context) -> str:
        """Emit a log message and some progress, then echo."""
        await ctx.info(f"handling {text}")
        await ctx.report_progress(1.0, 1.0, "done")
        return text

    return server
