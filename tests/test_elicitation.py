"""Tests for elicitation functionality."""

from typing import Any
from unittest.mock import MagicMock

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult, ErrorData

from langchain_mcp_adapters.elicitation import (
    BaseElicitationHandler,
    DeclineElicitationHandler,
    DefaultElicitationHandler,
    ElicitationResponse,
    FunctionElicitationHandler,
    convert_to_mcp_result,
    create_mcp_elicitation_callback,
)


class TestElicitationHandlers:
    """Test various elicitation handler implementations."""

    async def test_default_handler_auto_accept(self):
        """Test that DefaultElicitationHandler accepts with defaults."""
        handler = DefaultElicitationHandler(auto_accept=True)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "John"},
                "age": {"type": "integer"},
                "active": {"type": "boolean", "default": True},
            },
            "required": ["age"],
        }

        response = await handler.handle_elicitation(
            "Please provide your details",
            schema,
            "test_server"
        )

        assert response["action"] == "accept"
        assert response["data"] == {
            "name": "John",
            "age": 0,
            "active": True,
        }
        assert response["reason"] is None

    async def test_default_handler_decline(self):
        """Test that DefaultElicitationHandler can decline."""
        handler = DefaultElicitationHandler(auto_accept=False)

        schema = {"type": "object", "properties": {}}

        response = await handler.handle_elicitation(
            "Please provide input",
            schema,
            "test_server"
        )

        assert response["action"] == "decline"
        assert response["data"] is None
        assert "Automatic decline" in response["reason"]

    async def test_decline_handler(self):
        """Test DeclineElicitationHandler always declines."""
        handler = DeclineElicitationHandler(reason="Testing decline")

        schema = {"type": "object", "properties": {"field": {"type": "string"}}}

        response = await handler.handle_elicitation(
            "Provide input",
            schema,
            "test_server"
        )

        assert response["action"] == "decline"
        assert response["data"] is None
        assert response["reason"] == "Testing decline"

    async def test_function_handler(self):
        """Test FunctionElicitationHandler delegates correctly."""
        async def custom_handler(
            message: str, schema: dict, server_name: str | None
        ) -> ElicitationResponse:
            return {
                "action": "accept",
                "data": {"custom": "response"},
                "reason": None,
            }

        handler = FunctionElicitationHandler(custom_handler)

        response = await handler.handle_elicitation(
            "Provide input",
            {"type": "object"},
            "test_server"
        )

        assert response["action"] == "accept"
        assert response["data"] == {"custom": "response"}

    def test_schema_validation_valid(self):
        """Test schema validation with valid data."""
        class TestHandler(BaseElicitationHandler):
            async def handle_elicitation(self, message, schema, server_name=None):
                return {"action": "accept", "data": {}, "reason": None}

        handler = TestHandler()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["name"],
        }

        valid_data = {
            "name": "Alice",
            "age": 30,
            "active": True,
        }

        assert handler.validate_schema_response(valid_data, schema) is True

    def test_schema_validation_invalid(self):
        """Test schema validation with invalid data."""
        class TestHandler(BaseElicitationHandler):
            async def handle_elicitation(self, message, schema, server_name=None):
                return {"action": "accept", "data": {}, "reason": None}

        handler = TestHandler()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        # Missing required field
        invalid_data1 = {"name": "Alice"}
        assert handler.validate_schema_response(invalid_data1, schema) is False

        # Wrong type
        invalid_data2 = {"name": "Alice", "age": "thirty"}
        assert handler.validate_schema_response(invalid_data2, schema) is False

        # Additional property when not allowed
        schema["additionalProperties"] = False
        invalid_data3 = {"name": "Alice", "age": 30, "extra": "field"}
        assert handler.validate_schema_response(invalid_data3, schema) is False


class TestConversion:
    """Test conversion between formats."""

    async def test_convert_accept_response(self):
        """Test converting accept response to MCP format."""
        response: ElicitationResponse = {
            "action": "accept",
            "data": {"field": "value"},
            "reason": None,
        }

        result = await convert_to_mcp_result(response)

        assert isinstance(result, ElicitResult)
        assert result.action == "accept"
        assert result.content == {"field": "value"}

    async def test_convert_decline_response(self):
        """Test converting decline response to MCP format."""
        response: ElicitationResponse = {
            "action": "decline",
            "data": None,
            "reason": "User declined",
        }

        result = await convert_to_mcp_result(response)

        assert isinstance(result, ElicitResult)
        assert result.action == "decline"
        assert result.content is None

    async def test_convert_cancel_response(self):
        """Test converting cancel response to MCP format."""
        response: ElicitationResponse = {
            "action": "cancel",
            "data": None,
            "reason": None,
        }

        result = await convert_to_mcp_result(response)

        assert isinstance(result, ElicitResult)
        assert result.action == "cancel"
        assert result.content is None

    async def test_convert_invalid_response(self):
        """Test converting invalid response returns error."""
        response: ElicitationResponse = {
            "action": "invalid",  # type: ignore
            "data": None,
            "reason": None,
        }

        result = await convert_to_mcp_result(response)

        assert isinstance(result, ErrorData)
        assert result.code == -32602
        assert "Invalid elicitation action" in result.message

    async def test_create_mcp_callback(self):
        """Test creating MCP SDK-compatible callback."""
        # Create a simple handler
        handler = DefaultElicitationHandler(auto_accept=True)

        # Create the callback
        callback = await create_mcp_elicitation_callback(handler, "test_server")

        # Create mock context and params
        context = MagicMock(spec=RequestContext)
        params = ElicitRequestParams(
            message="Test elicitation",
            requestedSchema={
                "type": "object",
                "properties": {
                    "field": {"type": "string", "default": "test"},
                },
            },
        )

        # Call the callback
        result = await callback(context, params)

        # Verify result
        assert isinstance(result, ElicitResult)
        assert result.action == "accept"
        assert result.content == {"field": "test"}


class TestDefaultHandlerExtraction:
    """Test default value extraction in DefaultElicitationHandler."""

    async def test_extract_defaults_with_explicit_defaults(self):
        """Test extracting explicit default values from schema."""
        handler = DefaultElicitationHandler()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "Unknown"},
                "count": {"type": "integer", "default": 42},
                "active": {"type": "boolean", "default": True},
                "items": {"type": "array", "default": ["a", "b"]},
                "config": {"type": "object", "default": {"key": "value"}},
            },
        }

        defaults = handler._extract_defaults(schema)

        assert defaults == {
            "name": "Unknown",
            "count": 42,
            "active": True,
            "items": ["a", "b"],
            "config": {"key": "value"},
        }

    async def test_extract_defaults_for_required_fields(self):
        """Test that required fields get type-appropriate defaults."""
        handler = DefaultElicitationHandler()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
                "items": {"type": "array"},
                "config": {"type": "object"},
            },
            "required": ["name", "count", "active", "items", "config"],
        }

        defaults = handler._extract_defaults(schema)

        assert defaults == {
            "name": "",
            "count": 0,
            "active": False,
            "items": [],
            "config": {},
        }

    async def test_extract_defaults_mixed(self):
        """Test extraction with mix of explicit defaults and required fields."""
        handler = DefaultElicitationHandler()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "Bob"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["age"],
        }

        defaults = handler._extract_defaults(schema)

        assert defaults == {
            "name": "Bob",
            "age": 0,  # Required but no default, so uses type default
            # email is not included (not required, no default)
        }

    async def test_extract_defaults_non_object_schema(self):
        """Test that non-object schemas return empty dict."""
        handler = DefaultElicitationHandler()

        # Array schema
        assert handler._extract_defaults({"type": "array"}) == {}

        # String schema
        assert handler._extract_defaults({"type": "string"}) == {}

        # No type specified
        assert handler._extract_defaults({}) == {}


