"""Elicitation support for MCP clients.

This module provides handlers for MCP elicitation requests, allowing servers
to request structured user input during interactions.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from mcp.types import ElicitResult, ErrorData

if TYPE_CHECKING:
    from mcp.shared.context import RequestContext
    from mcp.types import ElicitRequestParams


class ElicitationResponse(TypedDict):
    """Response from an elicitation handler."""

    action: Literal["accept", "decline", "cancel"]
    """The user's action: accept with data, decline, or cancel."""

    data: dict[str, Any] | None
    """The user's response data when action is 'accept'."""

    reason: str | None
    """Optional reason for decline or cancel."""


class ElicitationHandler(Protocol):
    """Protocol for handling elicitation requests from MCP servers.

    Elicitation allows servers to request structured user input during interactions.
    Handlers implementing this protocol determine how to gather the requested data
    from users or other sources.
    """

    async def handle_elicitation(
        self,
        message: str,
        schema: dict[str, Any],
        server_name: str | None = None,
    ) -> ElicitationResponse:
        """Handle an elicitation request.

        Args:
            message: Human-readable prompt explaining the request
            schema: JSON Schema defining the expected response structure
            server_name: Optional name of the server making the request

        Returns:
            Response containing the action (accept/decline/cancel) and optional data
        """
        ...


class BaseElicitationHandler(ABC):
    """Base class for elicitation handlers with common functionality."""

    @abstractmethod
    async def handle_elicitation(
        self,
        message: str,
        schema: dict[str, Any],
        server_name: str | None = None,
    ) -> ElicitationResponse:
        """Handle an elicitation request.

        Args:
            message: Human-readable prompt explaining the request
            schema: JSON Schema defining the expected response structure
            server_name: Optional name of the server making the request

        Returns:
            Response containing the action (accept/decline/cancel) and optional data
        """
        ...

    def validate_schema_response(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> bool:
        """Validate that the data matches the expected schema.

        Args:
            data: The data to validate
            schema: The JSON Schema to validate against

        Returns:
            True if valid, False otherwise

        Note:
            This is a basic validation that checks required fields and types.
            For full JSON Schema validation, consider using jsonschema library.
        """
        if schema.get("type") != "object":
            return False

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field in required:
            if field not in data:
                return False

        # Check field types (basic validation)
        for field, value in data.items():
            if field not in properties:
                # Additional properties not allowed by default
                if not schema.get("additionalProperties", False):
                    return False
                continue

            field_schema = properties[field]
            field_type = field_schema.get("type")

            # Basic type checking
            if field_type == "string" and not isinstance(value, str):
                return False
            elif field_type == "number" and not isinstance(value, (int, float)):
                return False
            elif field_type == "integer" and not isinstance(value, int):
                return False
            elif field_type == "boolean" and not isinstance(value, bool):
                return False
            elif field_type == "array" and not isinstance(value, list):
                return False
            elif field_type == "object" and not isinstance(value, dict):
                return False

        return True


class DefaultElicitationHandler(BaseElicitationHandler):
    """Handler that returns default values from the schema.

    This handler automatically accepts elicitation requests using default
    values specified in the schema, or empty/zero values if no defaults
    are provided.
    """

    def __init__(self, auto_accept: bool = True):
        """Initialize the default handler.

        Args:
            auto_accept: If True, automatically accept with defaults.
                        If False, decline all requests.
        """
        self.auto_accept = auto_accept

    async def handle_elicitation(
        self,
        message: str,
        schema: dict[str, Any],
        server_name: str | None = None,
    ) -> ElicitationResponse:
        """Handle elicitation by returning default values.

        Args:
            message: Human-readable prompt explaining the request
            schema: JSON Schema defining the expected response structure
            server_name: Optional name of the server making the request

        Returns:
            Response with default values if auto_accept is True, otherwise decline
        """
        if not self.auto_accept:
            return {
                "action": "decline",
                "data": None,
                "reason": "Automatic decline (DefaultElicitationHandler)",
            }

        # Extract default values from schema
        data = self._extract_defaults(schema)

        return {
            "action": "accept",
            "data": data,
            "reason": None,
        }

    def _extract_defaults(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Extract default values from a JSON Schema.

        Args:
            schema: The JSON Schema to extract defaults from

        Returns:
            Dictionary with default values for each field
        """
        if schema.get("type") != "object":
            return {}

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        result = {}

        for field, field_schema in properties.items():
            # Use explicit default if provided
            if "default" in field_schema:
                result[field] = field_schema["default"]
            # For required fields without defaults, use type-appropriate empty values
            elif field in required:
                field_type = field_schema.get("type")
                if field_type == "string":
                    result[field] = ""
                elif field_type in ("number", "integer"):
                    result[field] = 0
                elif field_type == "boolean":
                    result[field] = False
                elif field_type == "array":
                    result[field] = []
                elif field_type == "object":
                    result[field] = {}

        return result


class DeclineElicitationHandler(BaseElicitationHandler):
    """Handler that always declines elicitation requests.

    Useful for non-interactive environments where user input is not available.
    """

    def __init__(self, reason: str = "User input not available"):
        """Initialize the decline handler.

        Args:
            reason: The reason to provide when declining
        """
        self.reason = reason

    async def handle_elicitation(
        self,
        message: str,
        schema: dict[str, Any],
        server_name: str | None = None,
    ) -> ElicitationResponse:
        """Decline the elicitation request.

        Args:
            message: Human-readable prompt explaining the request
            schema: JSON Schema defining the expected response structure
            server_name: Optional name of the server making the request

        Returns:
            Response declining the request
        """
        return {
            "action": "decline",
            "data": None,
            "reason": self.reason,
        }


class FunctionElicitationHandler(BaseElicitationHandler):
    """Handler that delegates elicitation to a custom function.

    This allows for flexible integration with existing systems or
    custom logic for handling elicitation requests.
    """

    def __init__(
        self,
        handler_func: Any,  # Callable[[str, dict, str | None], Awaitable[ElicitationResponse]]
    ):
        """Initialize with a custom handler function.

        Args:
            handler_func: Async function that takes (message, schema, server_name)
                         and returns an ElicitationResponse
        """
        self.handler_func = handler_func

    async def handle_elicitation(
        self,
        message: str,
        schema: dict[str, Any],
        server_name: str | None = None,
    ) -> ElicitationResponse:
        """Delegate to the custom handler function.

        Args:
            message: Human-readable prompt explaining the request
            schema: JSON Schema defining the expected response structure
            server_name: Optional name of the server making the request

        Returns:
            Response from the custom handler function
        """
        return await self.handler_func(message, schema, server_name)


async def convert_to_mcp_result(
    response: ElicitationResponse,
) -> ElicitResult | ErrorData:
    """Convert an ElicitationResponse to MCP's ElicitResult format.

    Args:
        response: The elicitation response to convert

    Returns:
        MCP-compatible ElicitResult or ErrorData
    """
    action = response["action"]

    if action == "accept":
        # Accept with data
        return ElicitResult(
            _meta={},
            action="accept",
            content=response["data"] or {},
        )
    elif action == "decline":
        # Explicit decline
        return ElicitResult(
            _meta={},
            action="decline",
            content=None,
        )
    elif action == "cancel":
        # User cancelled (dismissed dialog, etc.)
        return ElicitResult(
            _meta={},
            action="cancel",
            content=None,
        )
    else:
        # Invalid action - return error
        return ErrorData(
            code=-32602,  # Invalid params
            message=f"Invalid elicitation action: {action}",
        )


async def create_mcp_elicitation_callback(
    handler: ElicitationHandler,
    server_name: str | None = None,
) -> Any:  # Returns ElicitationFnT
    """Create an MCP SDK-compatible elicitation callback from a handler.

    Args:
        handler: The elicitation handler to use
        server_name: Optional name of the server for context

    Returns:
        Async function compatible with MCP SDK's elicitation_callback parameter
    """
    async def mcp_callback(
        context: RequestContext[Any, Any, Any],
        params: ElicitRequestParams,
    ) -> ElicitResult | ErrorData:
        """MCP SDK elicitation callback."""
        # Extract message and schema from params
        message = params.message
        schema = params.requestedSchema

        # Call our handler
        response = await handler.handle_elicitation(
            message=message,
            schema=schema,
            server_name=server_name,
        )

        # Convert response to MCP format
        return await convert_to_mcp_result(response)

    return mcp_callback


__all__ = [
    "ElicitationHandler",
    "ElicitationResponse",
    "BaseElicitationHandler",
    "DefaultElicitationHandler",
    "DeclineElicitationHandler",
    "FunctionElicitationHandler",
    "create_mcp_elicitation_callback",
    "convert_to_mcp_result",
]