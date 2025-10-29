"""Elicitation support for MCP clients.

This module provides handlers for MCP elicitation requests, allowing servers
to request structured user input during interactions.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from mcp.types import ElicitResult, ErrorData

try:
    import jsonschema
except ImportError:
    jsonschema = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from mcp.shared.context import RequestContext
    from mcp.types import ElicitRequestParams


# Security limits (configurable via handler __init__)
DEFAULT_MAX_SCHEMA_SIZE = 100_000  # 100KB
DEFAULT_MAX_PROPERTIES = 100


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

    def __init__(
        self,
        *,
        max_schema_size: int = DEFAULT_MAX_SCHEMA_SIZE,
        max_properties: int = DEFAULT_MAX_PROPERTIES,
        use_jsonschema: bool = True,
    ) -> None:
        """Initialize the base handler with security limits.

        Args:
            max_schema_size: Maximum schema size in characters
            max_properties: Maximum number of properties per object
            use_jsonschema: Whether to use jsonschema library for validation
        """
        self.max_schema_size = max_schema_size
        self.max_properties = max_properties
        self.use_jsonschema = use_jsonschema and jsonschema is not None
        self._jsonschema_module = jsonschema if self.use_jsonschema else None

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

    def _check_schema_size(self, schema: dict[str, Any]) -> bool:
        """Check if schema size is within limits.

        Args:
            schema: The schema to check

        Returns:
            True if within limits, False otherwise
        """
        schema_str = json.dumps(schema)
        return len(schema_str) <= self.max_schema_size

    def _check_property_count(self, schema: dict[str, Any]) -> bool:
        """Check if property count is within limits.

        Args:
            schema: The schema to check

        Returns:
            True if within limits, False otherwise
        """
        if schema.get("type") != "object":
            return True

        properties = schema.get("properties", {})
        return len(properties) <= self.max_properties

    def validate_schema_limits(self, schema: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate that schema is within security limits.

        Args:
            schema: The schema to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._check_schema_size(schema):
            return False, f"Schema exceeds maximum size of {self.max_schema_size}"

        if not self._check_property_count(schema):
            return (
                False,
                f"Schema exceeds maximum properties of {self.max_properties}",
            )

        return True, None

    def validate_schema_response(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> bool:
        """Validate that the data matches the expected schema.

        Args:
            data: The data to validate
            schema: The JSON Schema to validate against

        Returns:
            True if valid, False otherwise
        """
        # Use jsonschema if available
        if self.use_jsonschema and self._jsonschema_module is not None:
            try:
                self._jsonschema_module.validate(instance=data, schema=schema)
            except Exception:  # noqa: BLE001
                return False
            else:
                return True

        # Fall back to basic validation
        return self._basic_validate(data, schema)

    def _basic_validate(self, data: dict[str, Any], schema: dict[str, Any]) -> bool:
        """Basic validation for when jsonschema is not available.

        Args:
            data: The data to validate
            schema: The JSON Schema to validate against

        Returns:
            True if valid, False otherwise
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
            type_checks: list[tuple[bool, type | tuple[type, ...]]] = [
                (field_type == "string", str),
                (field_type == "number", (int, float)),
                (field_type == "integer", int),
                (field_type == "boolean", bool),
                (field_type == "array", list),
                (field_type == "object", dict),
            ]

            for condition, expected_type in type_checks:
                if condition and not isinstance(value, expected_type):
                    return False

        return True


class DefaultElicitationHandler(BaseElicitationHandler):
    """Handler that returns default values from the schema.

    This handler automatically accepts elicitation requests using default
    values specified in the schema, or empty/zero values if no defaults
    are provided.
    """

    def __init__(self, *, auto_accept: bool = True, **kwargs: Any) -> None:
        """Initialize the default handler.

        Args:
            auto_accept: If True, automatically accept with defaults.
                        If False, decline all requests.
            **kwargs: Additional arguments passed to BaseElicitationHandler
        """
        super().__init__(**kwargs)
        self.auto_accept = auto_accept

    async def handle_elicitation(
        self,
        message: str,  # noqa: ARG002
        schema: dict[str, Any],
        server_name: str | None = None,  # noqa: ARG002
    ) -> ElicitationResponse:
        """Handle elicitation by returning default values.

        Args:
            message: Human-readable prompt explaining the request (unused but required)
            schema: JSON Schema defining the expected response structure
            server_name: Optional name of the server (unused but required)

        Returns:
            Response with default values if auto_accept is True, otherwise decline
        """
        # Validate schema limits
        is_valid, error_msg = self.validate_schema_limits(schema)
        if not is_valid:
            return {
                "action": "decline",
                "data": None,
                "reason": f"Schema validation failed: {error_msg}",
            }

        if not self.auto_accept:
            return {
                "action": "decline",
                "data": None,
                "reason": "Automatic decline (DefaultElicitationHandler)",
            }

        # Extract and validate default values
        data = self._extract_defaults(schema)

        # Validate the defaults against the schema
        if not self.validate_schema_response(data, schema):
            return {
                "action": "decline",
                "data": None,
                "reason": "Generated defaults do not match schema",
            }

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
            # Validate default value if provided
            if "default" in field_schema:
                default_value = field_schema["default"]
                # Ensure default matches the field type
                field_type = field_schema.get("type")
                if self._is_valid_default(default_value, field_type):
                    result[field] = default_value
                elif field in required:
                    # Use type-appropriate empty value for invalid defaults
                    result[field] = self._get_empty_value(field_type)
            # For required fields without defaults, use type-appropriate empty values
            elif field in required:
                field_type = field_schema.get("type")
                result[field] = self._get_empty_value(field_type)

        return result

    def _is_valid_default(self, value: Any, field_type: str | None) -> bool:
        """Check if a default value is valid for the given type.

        Args:
            value: The default value to check
            field_type: The expected JSON Schema type

        Returns:
            True if valid, False otherwise
        """
        if field_type is None:
            return True

        type_mapping: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if field_type not in type_mapping:
            return True  # Unknown type, allow it

        expected_type = type_mapping[field_type]
        return isinstance(value, expected_type)

    def _get_empty_value(self, field_type: str | None) -> Any:
        """Get an appropriate empty value for a field type.

        Args:
            field_type: The JSON Schema type

        Returns:
            An appropriate empty value
        """
        if field_type is None:
            return None

        empty_values: dict[str, Any] = {
            "string": "",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {},
        }
        return empty_values.get(field_type)


class DeclineElicitationHandler(BaseElicitationHandler):
    """Handler that always declines elicitation requests.

    Useful for non-interactive environments where user input is not available.
    """

    def __init__(
        self, *, reason: str = "User input not available", **kwargs: Any
    ) -> None:
        """Initialize the decline handler.

        Args:
            reason: The reason to provide when declining
            **kwargs: Additional arguments passed to BaseElicitationHandler
        """
        super().__init__(**kwargs)
        self.reason = reason

    async def handle_elicitation(
        self,
        message: str,  # noqa: ARG002
        schema: dict[str, Any],  # noqa: ARG002
        server_name: str | None = None,  # noqa: ARG002
    ) -> ElicitationResponse:
        """Decline the elicitation request.

        Args:
            message: Human-readable prompt (unused but required)
            schema: JSON Schema (unused but required)
            server_name: Optional name of the server (unused but required)

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
        handler_func: Callable[
            [str, dict[str, Any], str | None], Awaitable[ElicitationResponse]
        ],
        **kwargs: Any,
    ) -> None:
        """Initialize with a custom handler function.

        Args:
            handler_func: Async function that takes (message, schema, server_name)
                         and returns an ElicitationResponse
            **kwargs: Additional arguments passed to BaseElicitationHandler
        """
        super().__init__(**kwargs)
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
        # Validate schema limits before delegating
        is_valid, error_msg = self.validate_schema_limits(schema)
        if not is_valid:
            return {
                "action": "decline",
                "data": None,
                "reason": f"Schema validation failed: {error_msg}",
            }

        # Call user's handler
        response = await self.handler_func(message, schema, server_name)

        # Validate response if accepting
        if (
            response["action"] == "accept"
            and response["data"] is not None
            and not self.validate_schema_response(response["data"], schema)
        ):
            return {
                "action": "decline",
                "data": None,
                "reason": "Response data does not match schema",
            }

        return response


class RateLimitedElicitationHandler(BaseElicitationHandler):
    """Handler that enforces rate limiting on elicitation requests.

    This wrapper can be used with any other handler to add rate limiting.
    """

    def __init__(
        self,
        wrapped_handler: ElicitationHandler,
        *,
        max_requests: int = 10,
        time_window: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the rate-limited handler.

        Args:
            wrapped_handler: The underlying handler to wrap
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
            **kwargs: Additional arguments passed to BaseElicitationHandler
        """
        super().__init__(**kwargs)
        self.wrapped_handler = wrapped_handler
        self.max_requests = max_requests
        self.time_window = time_window
        self._request_times: deque[float] = deque()

    async def handle_elicitation(
        self,
        message: str,
        schema: dict[str, Any],
        server_name: str | None = None,
    ) -> ElicitationResponse:
        """Handle elicitation with rate limiting.

        Args:
            message: Human-readable prompt explaining the request
            schema: JSON Schema defining the expected response structure
            server_name: Optional name of the server making the request

        Returns:
            Response from wrapped handler or decline if rate limited
        """
        current_time = time.time()

        # Remove old requests outside the time window
        cutoff_time = current_time - self.time_window
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()

        # Check if rate limit exceeded
        if len(self._request_times) >= self.max_requests:
            return {
                "action": "decline",
                "data": None,
                "reason": (
                    f"Rate limit exceeded: {self.max_requests} "
                    f"requests per {self.time_window}s"
                ),
            }

        # Record this request
        self._request_times.append(current_time)

        # Delegate to wrapped handler
        return await self.wrapped_handler.handle_elicitation(
            message, schema, server_name
        )


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
            action="accept",
            content=response["data"] or {},
        )
    if action == "decline":
        # Explicit decline
        return ElicitResult(
            action="decline",
            content=None,
        )
    if action == "cancel":
        # User cancelled (dismissed dialog, etc.)
        return ElicitResult(
            action="cancel",
            content=None,
        )

    # Invalid action - return error
    return ErrorData(
        code=-32602,  # Invalid params
        message=f"Invalid elicitation action: {action}",
    )


async def create_mcp_elicitation_callback(
    handler: ElicitationHandler,
    server_name: str | None = None,
) -> Callable[
    [RequestContext[Any, Any, Any], ElicitRequestParams],
    Awaitable[ElicitResult | ErrorData],
]:
    """Create an MCP SDK-compatible elicitation callback from a handler.

    Args:
        handler: The elicitation handler to use
        server_name: Optional name of the server for context

    Returns:
        Async function compatible with MCP SDK's elicitation_callback parameter
    """

    async def mcp_callback(
        context: RequestContext[Any, Any, Any],  # noqa: ARG001
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
    "BaseElicitationHandler",
    "DeclineElicitationHandler",
    "DefaultElicitationHandler",
    "ElicitationHandler",
    "ElicitationResponse",
    "FunctionElicitationHandler",
    "RateLimitedElicitationHandler",
    "convert_to_mcp_result",
    "create_mcp_elicitation_callback",
]
