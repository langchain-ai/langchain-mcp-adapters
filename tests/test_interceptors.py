"""Tests for the interceptor system functionality."""

import pytest
from langchain_core.messages import ToolMessage
from mcp.server import FastMCP
from mcp.types import (
    CallToolResult,
    TextContent,
)

from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from tests.utils import IsLangChainID, run_streamable_http


class TestMetaPassthrough:
    """Tests for _meta parameter passthrough to MCP servers."""

    async def test_meta_passed_via_arguments(self, socket_enabled):
        """Test that _meta in tool arguments is passed through to MCP server."""

        def _create_meta_echo_server(port: int = 8220):
            """Create a server that echoes the received _meta."""
            server = FastMCP(port=port)

            @server.tool()
            def echo_meta(message: str) -> str:
                """Echo message and return any received meta."""
                # The server receives _meta through the MCP protocol
                # We'll verify it was passed by checking the request context
                return f"Message: {message}"

            return server

        with run_streamable_http(_create_meta_echo_server, 8220):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8220/mcp",
                    "transport": "streamable_http",
                },
            )

            echo_tool = next(tool for tool in tools if tool.name == "echo_meta")
            # Pass _meta as part of the arguments - it should be extracted
            # and passed to MCP call_tool
            result = await echo_tool.ainvoke(
                {"message": "hello", "_meta": {"session_id": "abc123"}}
            )
            # Tool executes successfully (meta doesn't affect response in this test)
            assert "Message: hello" in str(result)

    async def test_interceptor_can_modify_meta(self, socket_enabled):
        """Test that interceptors can modify the meta field."""
        captured_meta = []

        async def add_meta_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            # Add or modify meta through the interceptor
            modified_request = request.override(
                meta={"added_by_interceptor": True, "user_id": "test-user"}
            )
            return await handler(modified_request)

        async def capture_meta_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            # Capture the meta after modification to verify it was set
            captured_meta.append(request.meta)
            return await handler(request)

        def _create_simple_server(port: int = 8221):
            server = FastMCP(port=port)

            @server.tool()
            def simple_tool(value: int) -> int:
                """Return doubled value."""
                return value * 2

            return server

        with run_streamable_http(_create_simple_server, 8221):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8221/mcp",
                    "transport": "streamable_http",
                },
                # First interceptor modifies, second captures to verify
                tool_interceptors=[add_meta_interceptor, capture_meta_interceptor],
            )

            simple_tool = next(tool for tool in tools if tool.name == "simple_tool")
            result = await simple_tool.ainvoke({"value": 5})
            assert "10" in str(result)

            # Verify the meta was modified by the first interceptor
            assert len(captured_meta) == 1
            assert captured_meta[0] == {
                "added_by_interceptor": True,
                "user_id": "test-user",
            }

    async def test_meta_and_args_passed_separately(self, socket_enabled):
        """Test that _meta is extracted from args and passed as separate param."""
        captured_requests = []

        async def capture_request_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            # Capture the request to verify _meta was extracted from args
            captured_requests.append(
                {
                    "args": dict(request.args),
                    "meta": request.meta,
                }
            )
            return await handler(request)

        def _create_capture_server(port: int = 8222):
            server = FastMCP(port=port)

            @server.tool()
            def capture_tool(a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

            return server

        with run_streamable_http(_create_capture_server, 8222):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8222/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[capture_request_interceptor],
            )

            capture_tool = next(tool for tool in tools if tool.name == "capture_tool")
            result = await capture_tool.ainvoke(
                {"a": 3, "b": 7, "_meta": {"session_id": "abc123"}}
            )

            # Verify the tool executed correctly
            assert "10" in str(result)

            # Verify _meta was extracted from args and placed in meta field
            assert len(captured_requests) == 1
            assert "_meta" not in captured_requests[0]["args"]
            assert captured_requests[0]["args"] == {"a": 3, "b": 7}
            assert captured_requests[0]["meta"] == {"session_id": "abc123"}


def _create_math_server(port: int = 8200):
    """Create a math server with add and multiply tools."""
    server = FastMCP(port=port)

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @server.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    return server


class TestInterceptorModifiesRequest:
    """Tests for interceptors that modify the request."""

    async def test_interceptor_modifies_arguments(self, socket_enabled):
        """Test that interceptor can modify tool arguments."""

        async def modify_args_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            # Double the arguments
            modified_args = {k: v * 2 for k, v in request.args.items()}
            modified_request = request.override(args=modified_args)
            return await handler(modified_request)

        with run_streamable_http(_create_math_server, 8200):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8200/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[modify_args_interceptor],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")
            # Original call would be 2 + 3 = 5, but interceptor doubles it to 4 + 6 = 10
            result = await add_tool.ainvoke({"a": 2, "b": 3})
            assert "10" in str(result)

    async def test_interceptor_modifies_tool_name(self, socket_enabled):
        """Test that interceptor can redirect to different tool."""

        async def redirect_tool_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            # Redirect add to multiply
            if request.name == "add":
                modified_request = request.override(name="multiply")
                return await handler(modified_request)
            return await handler(request)

        with run_streamable_http(_create_math_server, 8201):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8201/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[redirect_tool_interceptor],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")
            # Call add but interceptor redirects to multiply: 5 * 2 = 10
            result = await add_tool.ainvoke({"a": 5, "b": 2})
            assert result == [{"type": "text", "text": "10", "id": IsLangChainID}]


class TestInterceptorModifiesResponse:
    """Tests for interceptors that modify the response."""

    async def test_interceptor_modifies_result(self, socket_enabled):
        """Test that interceptor can modify tool result."""

        async def modify_result_interceptor(
            request: MCPToolCallRequest,
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

        with run_streamable_http(_create_math_server, 8203):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8203/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[modify_result_interceptor],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")
            result = await add_tool.ainvoke({"a": 2, "b": 3})
            assert result == [
                {"type": "text", "text": "Modified: 5", "id": IsLangChainID}
            ]

    async def test_interceptor_returns_custom_result(self, socket_enabled):
        """Test that interceptor can return a completely custom CallToolResult."""

        async def return_custom_result_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            # Don't call handler, just return custom result
            return CallToolResult(
                content=[TextContent(type="text", text="Custom tool response")],
                isError=False,
            )

        with run_streamable_http(_create_math_server, 8204):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8204/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[return_custom_result_interceptor],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")
            result = await add_tool.ainvoke({"a": 2, "b": 3})
            assert result == [
                {"type": "text", "text": "Custom tool response", "id": IsLangChainID}
            ]


class TestInterceptorAdvancedPatterns:
    """Tests for advanced interceptor patterns like caching."""

    async def test_interceptor_caching(self, socket_enabled):
        """Test that interceptor can implement caching."""
        cache = {}
        call_count = 0

        async def caching_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            nonlocal call_count
            cache_key = f"{request.name}:{request.args}"

            if cache_key in cache:
                return cache[cache_key]

            call_count += 1
            result = await handler(request)
            cache[cache_key] = result
            return result

        with run_streamable_http(_create_math_server, 8206):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8206/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[caching_interceptor],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")

            # First call - should execute
            result1 = await add_tool.ainvoke({"a": 2, "b": 3})
            assert result1 == [{"type": "text", "text": "5", "id": IsLangChainID}]
            assert call_count == 1

            # Second call with same args - should use cache
            result2 = await add_tool.ainvoke({"a": 2, "b": 3})
            assert result2 == [{"type": "text", "text": "5", "id": IsLangChainID}]
            assert call_count == 1  # Should not increment

            # Third call with different args - should execute
            result3 = await add_tool.ainvoke({"a": 5, "b": 7})
            assert result3 == [{"type": "text", "text": "12", "id": IsLangChainID}]
            assert call_count == 2


class TestInterceptorComposition:
    """Tests for composing multiple interceptors."""

    async def test_multiple_interceptors_compose(self, socket_enabled):
        """Test that multiple interceptors compose in the correct order."""
        execution_order = []

        async def logging_interceptor_1(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            execution_order.append("before_1")
            result = await handler(request)
            execution_order.append("after_1")
            return result

        async def logging_interceptor_2(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            execution_order.append("before_2")
            result = await handler(request)
            execution_order.append("after_2")
            return result

        # First in list should be outermost layer
        with run_streamable_http(_create_math_server, 8207):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8207/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[logging_interceptor_1, logging_interceptor_2],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")
            result = await add_tool.ainvoke({"a": 2, "b": 3})
            assert result == [{"type": "text", "text": "5", "id": IsLangChainID}]

            # Should execute in onion order: 1 before, 2 before, execute, 2 after,
            # 1 after
            assert execution_order == ["before_1", "before_2", "after_2", "after_1"]


class TestInterceptorErrorHandling:
    """Tests for interceptor error handling."""

    async def test_interceptor_exception_propagates(self, socket_enabled):
        """Test that exceptions in interceptors propagate correctly."""

        async def failing_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> CallToolResult:
            raise ValueError("Interceptor failed")

        with run_streamable_http(_create_math_server, 8208):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8208/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[failing_interceptor],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")
            with pytest.raises(ValueError, match="Interceptor failed"):
                await add_tool.ainvoke({"a": 2, "b": 3})

    async def test_interceptor_returns_tool_message(self, socket_enabled):
        """Test that interceptor can return a ToolMessage directly."""

        async def tool_message_interceptor(
            request: MCPToolCallRequest,
            handler,
        ) -> ToolMessage:
            # Return a ToolMessage directly instead of CallToolResult
            return ToolMessage(
                content="Custom ToolMessage response",
                name=request.name,
                tool_call_id="test-call-id",
            )

        with run_streamable_http(_create_math_server, 8209):
            tools = await load_mcp_tools(
                None,
                connection={
                    "url": "http://localhost:8209/mcp",
                    "transport": "streamable_http",
                },
                tool_interceptors=[tool_message_interceptor],
            )

            add_tool = next(tool for tool in tools if tool.name == "add")
            result = await add_tool.ainvoke(
                {"args": {"a": 2, "b": 3}, "id": "test-call-id", "type": "tool_call"}
            )

            # The interceptor returns a ToolMessage which should be returned as-is
            assert isinstance(result, ToolMessage)
            assert result.content == "Custom ToolMessage response"
            assert result.name == "add"
            assert result.tool_call_id == "test-call-id"
