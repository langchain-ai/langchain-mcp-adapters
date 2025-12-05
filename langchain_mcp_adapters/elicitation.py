"""Elicitation support for MCP tools via LangGraph interrupts.

This module provides types for handling MCP elicitation requests via LangGraph's
interrupt() function. When an MCP server requests information from a user during
tool execution, these types facilitate the request/response flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mcp.types import ElicitResult


@dataclass
class ElicitationRequest:
    """Elicitation request passed to LangGraph's interrupt().

    This dataclass wraps the MCP elicitation params with additional context
    about which server and tool triggered the elicitation.

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
    """Response to an elicitation request.

    Use this when resuming after an elicitation interrupt.

    Attributes:
        action: The user's action - "accept", "decline", or "cancel".
        content: Form data when action is "accept", None otherwise.

    Example:
        ```python
        # User accepts with data
        response = ElicitationResponse("accept", {"email": "user@example.com"})

        # User declines
        response = ElicitationResponse("decline")

        # User cancels
        response = ElicitationResponse("cancel")

        await graph.astream(Command(resume=response), thread)
        ```
    """

    action: Literal["accept", "decline", "cancel"]
    content: dict[str, Any] | None = field(default=None)

    def to_result(self) -> ElicitResult:
        """Convert to MCP ElicitResult."""
        return ElicitResult(action=self.action, content=self.content)
