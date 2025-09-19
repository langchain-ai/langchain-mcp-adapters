"""Agent middleware for LangChain."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    StreamableHttpConnection,
)
from langchain_core.tools import BaseTool


def _sync_await(coro):
    """Run an async coroutine from sync code, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread: just run it.
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


class MCPMiddleware(AgentMiddleware):
    """MCP Middleware to register tools for use in an agent."""

    def __init__(self, connections: dict[str, StreamableHttpConnection]) -> None:
        """Initialize the middleware."""
        self.server_to_connection = connections
        for connection in connections.values():
            assert connection["transport"] == "streamable_http"
            # Check how it looks if we only support stateless connections
            # assert connection["is_stateful"] == False
        self._client = MultiServerMCPClient(connections)
        self._tools = None

    # Do we want to make tools a property? Or support an async method?
    @property
    def tools(self) -> list[BaseTool]:
        """No additional tools"""
        if self._tools is not None:
            return self._tools
        self._tools = _sync_await(self._client.get_tools())
        return self._tools
