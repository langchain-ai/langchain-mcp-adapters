"""Callback interfaces and types for MCP client lifecycle management.

This module provides callback interfaces for handling notifications
from MCP servers during client operation.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from mcp.types import LoggingMessageNotification, ProgressNotification


@dataclass
class CallbackContext:
    """Context object passed to callbacks containing state information."""

    server_name: str
    state: dict[str, Any] = field(default_factory=dict)
    runnable_config: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


class OnLoggingMessageCallback(Protocol):
    """Protocol for logging message callback functions."""

    async def __call__(
        self,
        notification: LoggingMessageNotification,
        context: CallbackContext,
    ) -> None:
        """Handle logging message notification."""
        ...


class OnProgressNotificationCallback(Protocol):
    """Protocol for progress notification callback functions."""

    async def __call__(
        self,
        notification: ProgressNotification,
        context: CallbackContext,
    ) -> None:
        """Handle progress notification."""
        ...


class Callbacks:
    """Container for MCP client callback functions."""

    def __init__(
        self,
        *,
        on_logging_message: OnLoggingMessageCallback | None = None,
        on_progress_notification: OnProgressNotificationCallback | None = None,
    ) -> None:
        """Initialize callbacks.

        Args:
            on_logging_message: Callback for logging message notifications
            on_progress_notification: Callback for progress notifications
        """
        self.on_logging_message = on_logging_message
        self.on_progress_notification = on_progress_notification
