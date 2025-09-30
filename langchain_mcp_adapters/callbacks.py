"""Types for callbacks."""

from dataclasses import dataclass
from typing import Protocol

from mcp.client.session import LoggingFnT
from mcp.shared.session import ProgressFnT
from mcp.types import LoggingMessageNotificationParams


@dataclass
class CallbackContext:
    """LangChain MCP client callback context."""

    server_name: str
    tool_name: str | None = None


class LoggingMessageCallback(Protocol):
    """Light wrapper around the mcp.client.session.LoggingFnT.

    Injects callback context as the last argument.
    """

    async def __call__(
        self,
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
    ) -> None:
        """Execute callback on logging message notification."""
        ...


class ProgressCallback(Protocol):
    """Light wrapper around the mcp.shared.session.ProgressFnT.

    Injects callback context as the last argument.
    """

    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        """Execute callback on progress notification."""
        ...


@dataclass
class _MCPCallbacks:
    """Callbacks compatible with the MCP SDK. For internal use only."""

    logging_callback: LoggingFnT | None = None
    progress_callback: ProgressFnT | None = None


@dataclass
class Callbacks:
    """Callbacks for the LangChain MCP client."""

    on_logging_message: LoggingMessageCallback | None = None
    on_progress: ProgressCallback | None = None

    def to_mcp_format(self, *, context: CallbackContext) -> _MCPCallbacks:
        """Convert the LangChain MCP client callbacks to MCP SDK callbacks.

        Injects the LangChain CallbackContext as the last argument.
        """
        if (on_logging_message := self.on_logging_message) is not None:

            async def mcp_logging_callback(
                params: LoggingMessageNotificationParams,
            ) -> None:
                await on_logging_message(params, context)
        else:
            mcp_logging_callback = None

        if (on_progress := self.on_progress) is not None:

            async def mcp_progress_callback(
                progress: float, total: float | None, message: str | None
            ) -> None:
                await on_progress(progress, total, message, context)
        else:
            mcp_progress_callback = None

        return _MCPCallbacks(
            logging_callback=mcp_logging_callback,
            progress_callback=mcp_progress_callback,
        )
