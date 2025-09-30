"""Hook interfaces and types for MCP client lifecycle management.

This module provides hook interfaces for intercepting and extending
MCP client behavior before and after tool calls.

In the future, we might add more hooks for other parts of the
request / result lifecycle, for example to support elicitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from mcp.types import CallToolResult as MCPCallToolResult
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

# Type aliases to avoid direct MCP type dependencies
CallToolResult = MCPCallToolResult


@dataclass
class ToolHookContext:
    """Context object passed to hooks containing state and server information."""

    server_name: str
    tool_name: str

    # we'll add state eventually when we have a context manager like get_state()
    # state: object | None = None
    config: RunnableConfig | None = None
    runtime: object | None = None


class CallToolRequestSpec(TypedDict, total=False):
    """Result of before tool call hook."""

    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    headers: NotRequired[dict[str, Any]]


class BeforeToolCallHook(Protocol):
    """Protocol for before_tool_call hook functions.

    Allows modification of tool call arguments and headers before execution.
    Return None to proceed with original request.
    """

    async def __call__(
        self,
        request: CallToolRequestSpec,
        context: ToolHookContext,
    ) -> CallToolRequestSpec | None:
        """Execute before tool call.

        Args:
            request: The original tool call request
            context: Hook context with server/tool info and shared state

        Returns:
            Modified CallToolRequest or None to use original request
        """
        ...


class AfterToolCallHook(Protocol):
    """Protocol for after_tool_call hook functions.

    Allows modification of tool call results after execution.
    Return None to proceed with original result processing.
    Return CallToolResult to use the modified result.
    """

    async def __call__(
        self,
        result: CallToolResult,
        context: ToolHookContext,
    ) -> CallToolResult | None:
        """Execute after tool call.

        Args:
            result: The original tool call result
            context: Hook context with server/tool info and shared state

        Returns:
            - CallToolResult to use the modified result
            - None to use original result
        """
        ...


class Hooks:
    """Container for MCP client hook functions."""

    def __init__(
        self,
        *,
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
    ) -> None:
        """Initialize hooks.

        Args:
            before_tool_call: Hook called before tool execution
            after_tool_call: Hook called after tool execution
        """
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
