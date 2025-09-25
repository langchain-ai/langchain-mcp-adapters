"""Hook interfaces and types for MCP client lifecycle management.

This module provides hook interfaces for intercepting and extending
MCP client behavior before and after tool calls.

In the future, we might add more hooks for other parts of the
request / result lifecycle, for example to support elicitation.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from mcp.types import CallToolRequest, CallToolResult


@dataclass
class ToolHookContext:
    """Context object passed to hooks containing state and server information."""

    server_name: str

    state: dict[str, Any] = field(default_factory=dict)
    runnable_config: RunnableConfig = field(default_factory=dict)
    runtime: object = None


class CallToolSpecs(TypedDict, total=False):
    headers: dict[str, Any]


class BeforeToolCallResult(TypedDict, total=False):
    """Result returned by before_tool_call hook."""

    name: str
    args: dict[str, Any]
    headers: dict[str, Any]


class BeforeToolCallHook(Protocol):
    """Protocol for before_tool_call hook functions."""

    async def __call__(
        self,
        request: CallToolRequest,
        context: ToolHookContext,
    ) -> CallToolRequest | tuple[CallToolRequest, CallToolSpecs]:
        """Execute before tool call."""
        ...


UpdatedContent = str | list[str | dict[str, Any]]


class AfterToolCallHook(Protocol):
    """Protocol for after_tool_call hook functions."""

    async def __call__(
        self,
        result: CallToolResult,
        context: ToolHookContext,
    ) -> UpdatedContent | ToolMessage | None:
        """Execute after tool call."""
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
