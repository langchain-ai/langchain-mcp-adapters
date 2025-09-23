"""Interceptor types and interfaces for MCP tool call interception.

This module provides callback interfaces and types for intercepting and modifying
MCP tool calls at runtime, enabling runtime personalization, policy enforcement,
and enhanced observability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage
    from mcp.types import CallToolResult


class ToolCallArgs(TypedDict):
    """Arguments for a tool call that can be modified by interceptors."""

    name: str
    """The name of the tool being called."""

    arguments: dict[str, Any]
    """The arguments to pass to the tool."""

    headers: NotRequired[dict[str, str]]
    """Optional headers to include with the tool call."""


class ToolCallResult(TypedDict):
    """Result from a tool call that can be modified by interceptors."""

    content: str | list[str]
    """The text content returned by the tool."""

    artifacts: NotRequired[list[Any]]
    """Optional artifacts/non-text content returned by the tool."""


class ProgressEvent(TypedDict):
    """Progress event information."""

    progress: NotRequired[float]
    """Progress percentage (0-100)."""

    source: NotRequired[str]
    """Source of the progress event."""

    total: NotRequired[int]
    """Total number of items/steps."""

    completed: NotRequired[int]
    """Number of completed items/steps."""


class MessageEvent(TypedDict):
    """Server message event information."""

    level: str
    """Log level (info, warning, error, etc.)."""

    message: str
    """The message content."""

    source: NotRequired[str]
    """Source of the message."""


# Callback type definitions
OnMessageCallback = Callable[[MessageEvent], None | Awaitable[None]]
"""Callback for server messages and diagnostic information."""

OnProgressCallback = Callable[[ProgressEvent], None | Awaitable[None]]
"""Callback for progress events during tool execution."""

OnToolsListChangedCallback = Callable[[list[str]], None | Awaitable[None]]
"""Callback when the list of available tools changes."""

OnPromptsListChangedCallback = Callable[[list[str]], None | Awaitable[None]]
"""Callback when the list of available prompts changes."""

OnResourcesListChangedCallback = Callable[[list[str]], None | Awaitable[None]]
"""Callback when the list of available resources changes."""


class BeforeToolCallHook(Protocol):
    """Protocol for before-tool-call interceptor hooks.

    This hook is called before a tool is executed and allows modification
    of the tool arguments and headers.
    """

    def __call__(
        self,
        tool_call: ToolCallArgs,
    ) -> ToolCallArgs | Awaitable[ToolCallArgs]:
        """Modify tool call arguments before execution.

        Args:
            tool_call: The tool call arguments that can be modified.

        Returns:
            The modified tool call arguments.
        """
        ...


class AfterToolCallHook(Protocol):
    """Protocol for after-tool-call interceptor hooks.

    This hook is called after a tool is executed and allows transformation
    of the tool result. The result can be returned in multiple formats:
    - ToolCallResult dict with content and optional artifacts
    - 2-tuple [content, artifacts]
    - ToolMessage instance
    - String or other result types
    """

    def __call__(
        self,
        tool_call: ToolCallArgs,
        result: CallToolResult,
    ) -> (
        ToolCallResult
        | tuple[str | list[str], list[Any] | None]
        | ToolMessage
        | str
        | Any
        | Awaitable[
            ToolCallResult
            | tuple[str | list[str], list[Any] | None]
            | ToolMessage
            | str
            | Any
        ]
    ):
        """Transform tool call result after execution.

        Args:
            tool_call: The original tool call arguments.
            result: The result from the MCP tool call.

        Returns:
            The transformed result in one of several supported formats:
            - ToolCallResult dict with content and optional artifacts
            - 2-tuple [content, artifacts]
            - ToolMessage instance
            - String or other result types
        """
        ...


class InterceptorConfig(TypedDict, total=False):
    """Configuration for tool call interceptors and callbacks."""

    # Hook functions
    before_tool_call: BeforeToolCallHook
    """Hook called before tool execution to modify arguments."""

    after_tool_call: AfterToolCallHook
    """Hook called after tool execution to transform results."""

    # Notification callbacks
    on_message: OnMessageCallback
    """Callback for server log and diagnostic messages."""

    on_progress: OnProgressCallback
    """Callback for progress events during tool execution."""

    on_tools_list_changed: OnToolsListChangedCallback
    """Callback when the list of available tools changes."""

    on_prompts_list_changed: OnPromptsListChangedCallback
    """Callback when the list of available prompts changes."""

    on_resources_list_changed: OnResourcesListChangedCallback
    """Callback when the list of available resources changes."""


__all__ = [
    # Types
    "ToolCallArgs",
    "ToolCallResult",
    "ProgressEvent",
    "MessageEvent",
    "InterceptorConfig",

    # Protocols
    "BeforeToolCallHook",
    "AfterToolCallHook",

    # Callback types
    "OnMessageCallback",
    "OnProgressCallback",
    "OnToolsListChangedCallback",
    "OnPromptsListChangedCallback",
    "OnResourcesListChangedCallback",
]