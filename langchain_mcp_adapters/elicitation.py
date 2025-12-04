"""Elicitation support for MCP tools via LangGraph interrupts.

This module provides types and utilities for handling MCP elicitation requests
by leveraging LangGraph's interrupt() function. When an MCP server requests
information from a user during tool execution, this module translates the
request into a LangGraph interrupt, allowing the graph to pause, collect
user input, and resume with the response.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from mcp.types import ElicitRequestParams, ElicitResult

if TYPE_CHECKING:
    from mcp.shared.context import RequestContext


class ElicitationAction(str, Enum):
    """User's response action to an elicitation request."""

    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"


@dataclass(frozen=True)
class ElicitationRequest:
    """Elicitation request from MCP server.

    This dataclass represents an elicitation request that will be passed
    to LangGraph's interrupt() function. It contains all the information
    needed for a UI to present the request to the user.

    Attributes:
        message: Human-readable explanation of why information is needed.
        requested_schema: JSON Schema defining expected response structure.
        server_name: Name of the MCP server requesting elicitation.
        tool_name: Name of the tool that triggered elicitation.
    """

    message: str
    requested_schema: dict[str, Any]
    server_name: str
    tool_name: str


@dataclass
class ElicitationResponse:
    """User's response to an elicitation request.

    Attributes:
        action: The user's response action (accept/decline/cancel).
        content: Form data if action is accept.
    """

    action: ElicitationAction
    content: dict[str, Any] | None = None

    @classmethod
    def accept(
        cls, content: dict[str, Any] | None = None
    ) -> ElicitationResponse:
        """Create an accept response with optional form data."""
        return cls(action=ElicitationAction.ACCEPT, content=content)

    @classmethod
    def decline(cls) -> ElicitationResponse:
        """Create a decline response."""
        return cls(action=ElicitationAction.DECLINE)

    @classmethod
    def cancel(cls) -> ElicitationResponse:
        """Create a cancel response."""
        return cls(action=ElicitationAction.CANCEL)


@dataclass
class _PendingElicitation:
    """Internal: A pending elicitation request waiting for a response."""

    request: ElicitationRequest
    response_event: asyncio.Event
    response: ElicitationResponse | None = None


class ElicitationBridge:
    """Bridges MCP elicitation callbacks to LangGraph interrupts.

    This class solves the task-boundary problem where MCP's elicitation
    callback runs in a different task (the receive loop) than LangGraph's
    tool execution task.

    Architecture:
    - Task B (MCP receive loop): Calls elicitation_callback, which puts
      the request in _pending and waits on response_event
    - Task A (tool execution): Monitors _pending via poll_and_interrupt(),
      calls interrupt(), sets response and signals response_event
    """

    def __init__(self, server_name: str, tool_name: str) -> None:
        """Initialize the elicitation bridge.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool being executed.
        """
        self.server_name = server_name
        self.tool_name = tool_name
        self._pending: _PendingElicitation | None = None
        self._lock = asyncio.Lock()

    async def elicitation_callback(
        self,
        context: RequestContext[Any, Any],
        params: ElicitRequestParams,
    ) -> ElicitResult:
        """MCP SDK callback - runs in receive loop task (Task B).

        This method is called by the MCP SDK when the server sends an
        elicitation request. It stores the request and waits for the
        tool execution task to provide a response via set_response().

        Args:
            context: MCP request context.
            params: Elicitation request parameters from the server.

        Returns:
            ElicitResult to send back to the MCP server.
        """
        # Convert MCP request to our internal format
        request = ElicitationRequest(
            message=params.message,
            requested_schema=params.requestedSchema or {},
            server_name=self.server_name,
            tool_name=self.tool_name,
        )

        # Create pending elicitation with an event to wait on
        async with self._lock:
            self._pending = _PendingElicitation(
                request=request,
                response_event=asyncio.Event(),
            )

        # Wait for the tool execution task to provide a response
        # This will block until set_response() is called
        await self._pending.response_event.wait()

        # Get the response and clear pending state
        async with self._lock:
            response = self._pending.response
            self._pending = None

        if response is None:
            # This shouldn't happen, but handle gracefully
            return ElicitResult(action="cancel")

        # Convert back to MCP format
        if response.action == ElicitationAction.ACCEPT and response.content:
            return ElicitResult(action="accept", content=response.content)
        return ElicitResult(action=response.action.value)

    def has_pending_elicitation(self) -> bool:
        """Check if there's a pending elicitation request."""
        return self._pending is not None and self._pending.response is None

    def get_pending_request(self) -> ElicitationRequest | None:
        """Get the pending elicitation request, if any."""
        if self._pending is not None and self._pending.response is None:
            return self._pending.request
        return None

    async def set_response(self, response: ElicitationResponse) -> None:
        """Set the response for the pending elicitation.

        Called from Task A (tool execution) after interrupt() returns.

        Args:
            response: The user's response to the elicitation.
        """
        async with self._lock:
            if self._pending is not None:
                self._pending.response = response
                self._pending.response_event.set()
