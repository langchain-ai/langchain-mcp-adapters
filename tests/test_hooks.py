"""Tests for the interceptor system functionality."""

import os
from pathlib import Path

import pytest
from mcp.types import (
    CallToolResult,
    TextContent,
)

from langchain_mcp_adapters.interceptors import (
    Interceptors,
    ToolCallRequest,
    ToolInterceptorContext,
)
from langchain_mcp_adapters.tools import load_mcp_tools


def _get_math_server_config():
    """Get configuration for math server."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    return {
        "command": "python3",
        "args": [math_server_path],
        "transport": "stdio",
    }


class TestInterceptorModifiesRequest:
    """Tests for interceptors that modify the request."""

    async def test_interceptor_modifies_arguments(self):
        """Test that interceptor can modify tool arguments."""

        async def modify_args_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            # Double the arguments
            args = request.get("args", {})
            modified_args = {k: v * 2 for k, v in args.items()}
            modified_request = ToolCallRequest(
                name=request.get("name"),
                args=modified_args,
                headers=request.get("headers", {}),
            )
            return await handler(modified_request)

        interceptors = Interceptors(tools=[modify_args_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        # Original call would be 2 + 3 = 5, but interceptor doubles it to 4 + 6 = 10
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert "10" in str(result)

    async def test_interceptor_modifies_tool_name(self):
        """Test that interceptor can redirect to different tool."""

        async def redirect_tool_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            # Redirect add to multiply
            if request.get("name") == "add":
                modified_request = ToolCallRequest(
                    name="multiply",
                    args=request.get("args", {}),
                    headers=request.get("headers", {}),
                )
                return await handler(modified_request)
            return await handler(request)

        interceptors = Interceptors(tools=[redirect_tool_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        # Call add but interceptor redirects to multiply: 5 * 2 = 10
        result = await add_tool.ainvoke({"a": 5, "b": 2})
        assert result == "10"

    async def test_interceptor_modifies_headers(self):
        """Test that interceptor can modify headers (for applicable transports)."""

        async def modify_headers_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            modified_request = ToolCallRequest(
                name=request.get("name"),
                args=request.get("args", {}),
                headers={"X-Custom-Header": "test-value"},
            )
            return await handler(modified_request)

        interceptors = Interceptors(tools=[modify_headers_interceptor])
        # For stdio transport, headers won't apply, but interceptor should still work
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result == "5"


class TestInterceptorModifiesResponse:
    """Tests for interceptors that modify the response."""

    async def test_interceptor_modifies_result(self):
        """Test that interceptor can modify tool result."""

        async def modify_result_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            # Execute the tool first
            result = await handler(request)

            # Prepend "Modified: " to all text content
            modified_content = []
            for content in result.content:
                if isinstance(content, TextContent):
                    modified_content.append(
                        TextContent(type="text", text=f"Modified: {content.text}")
                    )
                else:
                    modified_content.append(content)

            return CallToolResult(
                content=modified_content,
                isError=result.isError,
            )

        interceptors = Interceptors(tools=[modify_result_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        # The interceptor modifies the result
        assert result == "Modified: 5"

    async def test_interceptor_returns_custom_result(self):
        """Test that interceptor can return a completely custom CallToolResult."""

        async def return_custom_result_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            # Don't call handler, just return custom result
            return CallToolResult(
                content=[TextContent(type="text", text="Custom tool response")],
                isError=False,
            )

        interceptors = Interceptors(tools=[return_custom_result_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        # The interceptor returns a custom result without calling handler
        assert result == "Custom tool response"


class TestInterceptorAdvancedPatterns:
    """Tests for advanced interceptor patterns like retry and caching."""

    async def test_interceptor_retry_on_error(self):
        """Test that interceptor can implement retry logic."""
        call_count = 0

        async def retry_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            nonlocal call_count
            for attempt in range(3):
                call_count += 1
                result = await handler(request)
                if not result.isError:
                    return result
            return result

        interceptors = Interceptors(tools=[retry_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        # Should succeed on first try since there's no error
        assert result == "5"
        assert call_count == 1

    async def test_interceptor_caching(self):
        """Test that interceptor can implement caching."""
        cache = {}
        call_count = 0

        async def caching_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            nonlocal call_count
            cache_key = f"{request.get('name')}:{request.get('args')}"

            if cache_key in cache:
                return cache[cache_key]

            call_count += 1
            result = await handler(request)
            cache[cache_key] = result
            return result

        interceptors = Interceptors(tools=[caching_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")

        # First call - should execute
        result1 = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result1 == "5"
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result2 == "5"
        assert call_count == 1  # Should not increment

        # Third call with different args - should execute
        result3 = await add_tool.ainvoke({"a": 5, "b": 7})
        assert result3 == "12"
        assert call_count == 2


class TestInterceptorComposition:
    """Tests for composing multiple interceptors."""

    async def test_multiple_interceptors_compose(self):
        """Test that multiple interceptors compose in the correct order."""
        execution_order = []

        async def logging_interceptor_1(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            execution_order.append("before_1")
            result = await handler(request)
            execution_order.append("after_1")
            return result

        async def logging_interceptor_2(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            execution_order.append("before_2")
            result = await handler(request)
            execution_order.append("after_2")
            return result

        # First in list should be outermost layer
        interceptors = Interceptors(
            interceptors=[logging_interceptor_1, logging_interceptor_2]
        )
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result == "5"

        # Should execute in onion order: 1 before, 2 before, execute, 2 after, 1 after
        assert execution_order == ["before_1", "before_2", "after_2", "after_1"]


class TestInterceptorErrorHandling:
    """Tests for interceptor error handling."""

    async def test_interceptor_exception_propagates(self):
        """Test that exceptions in interceptors propagate correctly."""

        async def failing_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            raise ValueError("Interceptor failed")

        interceptors = Interceptors(tools=[failing_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        with pytest.raises(ValueError, match="Interceptor failed"):
            await add_tool.ainvoke({"a": 2, "b": 3})

    async def test_interceptor_exception_after_handler_propagates(self):
        """Test that exceptions after handler execution propagate correctly."""

        async def failing_after_interceptor(
            request: ToolCallRequest,
            context: ToolInterceptorContext,
            handler,
        ) -> CallToolResult:
            await handler(request)
            raise RuntimeError("After handler failed")

        interceptors = Interceptors(tools=[failing_after_interceptor])
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            interceptors=interceptors,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        with pytest.raises(RuntimeError, match="After handler failed"):
            await add_tool.ainvoke({"a": 2, "b": 3})
