"""Interceptor interfaces and types for MCP client tool call lifecycle management.

This module provides an interceptor interface for wrapping and controlling
MCP tool call execution with a handler callback pattern.

In the future, we might add more interceptors for other parts of the
request / result lifecycle, for example to support elicitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from mcp.types import CallToolResult as MCPCallToolResult
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Type aliases to avoid direct MCP type dependencies
CallToolResult = MCPCallToolResult


@dataclass
class ToolInterceptorContext:
    """Context passed to interceptors for tool execution.

    Attributes:
        server_name: Name of the MCP server handling the tool.
        tool_name: Name of the tool being executed.
        runtime: LangGraph runtime instance if available.
    """

    server_name: str
    tool_name: str
    # note: langgraph state is not yet available as part of the context.
    # it needs to be plumbed through.
    runtime: object | None = None


class ToolCallRequest(TypedDict, total=False):
    """TypedDict representing a modifiable tool call request.

    All fields are optional to allow partial modifications by interceptors.

    Attributes:
        name: Tool name to invoke.
        args: Tool arguments as key-value pairs.
        headers: HTTP headers for applicable transports (SSE, HTTP).
    """

    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    headers: NotRequired[dict[str, Any]]


class ToolCallInterceptor(Protocol):
    """Protocol for tool call interceptors using handler callback pattern.

    Interceptors wrap tool execution to enable request/response modification,
    retry logic, caching, rate limiting, and other cross-cutting concerns.
    Multiple interceptors compose in "onion" pattern (first is outermost).

    The handler can be called multiple times (retry), skipped (caching/short-circuit),
    or wrapped with error handling. Each handler call is independent.
    """

    async def __call__(
        self,
        request: ToolCallRequest,
        context: ToolInterceptorContext,
        handler: Callable[[ToolCallRequest], Awaitable[CallToolResult]],
    ) -> CallToolResult:
        """Intercept tool execution with control over handler invocation.

        Args:
            request: Tool call request (name, args, headers).
            context: Execution context with server/tool info and LangGraph state.
            handler: Async callable executing the tool. Can be called multiple
                times, skipped, or wrapped for error handling.

        Returns:
            Final CallToolResult from tool execution or interceptor logic.
        ...
        """


@dataclass
class Interceptors:
    """Container for MCP client interceptors.

    Interceptors compose in order with first as outermost layer.
    For [A, B, C], execution order is A -> B -> C  tool_call.

    Future: Will support resource_interceptors and prompt_interceptors.

    Attributes:
        tools: List of tool call interceptors applied in order.
    """

    tools: list[ToolCallInterceptor] = field(default_factory=list)
