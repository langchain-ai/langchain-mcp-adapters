"""Regression tests for MCPToolCallResult typing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_mcp_adapters.interceptors import MCPToolCallResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_mcp_adapters.interceptors import MCPToolCallRequest

    async def example_interceptor(
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        return await handler(request)


def test_mcp_tool_call_result_is_exported() -> None:
    assert MCPToolCallResult is not None
