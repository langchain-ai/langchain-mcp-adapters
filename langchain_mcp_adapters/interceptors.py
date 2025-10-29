"""Interceptor interfaces and types for MCP client tool call lifecycle management.

This module provides an interceptor interface for wrapping and controlling
MCP tool call execution with a handler callback pattern.

In the future, we might add more interceptors for other parts of the
request / result lifecycle, for example to support elicitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from mcp.types import CallToolResult as MCPCallToolResult
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

# Type aliases to avoid direct MCP type dependencies
CallToolResult = MCPCallToolResult


@dataclass
class ToolInterceptorContext:
    """Context object passed to interceptors containing state and server information."""

    server_name: str
    tool_name: str

    # we'll add state eventually when we have a context manager like get_state()
    # state: object | None = None
    config: RunnableConfig | None = None
    runtime: object | None = None


class ToolCallRequest(TypedDict, total=False):
    """Tool call request that can be modified by interceptors."""

    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    headers: NotRequired[dict[str, Any]]


class ToolCallInterceptor(Protocol):
    """Protocol for tool call interceptor functions.

    Interceptors wrap tool execution with a handler callback pattern, enabling:
    - Request/response modification
    - Retry logic
    - Caching
    - Rate limiting
    - Circuit breaking
    - Logging and monitoring
    - Short-circuiting execution

    The handler callback executes the actual tool call. Interceptors can:
    - Call handler once with original or modified request
    - Call handler multiple times (e.g., for retry logic)
    - Skip calling handler entirely (e.g., return cached result)
    - Wrap handler call in try/catch for error handling

    Multiple interceptors compose in an "onion" pattern where the first
    interceptor in the list is the outermost layer.
    """

    async def __call__(
        self,
        request: ToolCallRequest,
        context: ToolInterceptorContext,
        handler: Callable[[ToolCallRequest], Awaitable[CallToolResult]],
    ) -> CallToolResult:
        """Intercept and control tool execution via handler callback.

        The handler callback executes the tool call and returns a CallToolResult.
        Interceptors can call the handler multiple times for retry logic, skip
        calling it to short-circuit, or modify the request/response. Multiple
        interceptors compose with first in list as outermost layer.

        Args:
            request: Tool call request with name, args, and optional headers.
            context: Interceptor context with server/tool info, config, and runtime.
            handler: Async callable to execute the tool and return CallToolResult.
                Can be called multiple times for retry logic. Can skip calling
                it to short-circuit execution.

        Returns:
            CallToolResult (the final result).

        The handler callable can be invoked multiple times for retry logic.
        Each call to handler is independent and stateless.

        Examples:
            Async retry on error:
            ```python
            async def __call__(self, request, context, handler):
                for attempt in range(3):
                    try:
                        result = await handler(request)
                        if not result.isError:
                            return result
                    except Exception:
                        if attempt == 2:
                            raise
                return result
            ```

            Caching:
            ```python
            async def __call__(self, request, context, handler):
                cache_key = f"{request['name']}:{request['args']}"
                if cached := await get_cache_async(cache_key):
                    return cached
                result = await handler(request)
                await save_cache_async(cache_key, result)
                return result
            ```

            Request modification:
            ```python
            async def __call__(self, request, context, handler):
                modified_request = request.copy()
                modified_request["args"]["verbose"] = True
                return await handler(modified_request)
            ```
        """
        ...

