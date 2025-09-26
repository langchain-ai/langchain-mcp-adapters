"""Tests for the hooks system functionality."""

import os
from pathlib import Path

import pytest
from mcp.types import (
    CallToolResult,
    TextContent,
)

from langchain_mcp_adapters.hooks import CallToolRequestSpec, Hooks, ToolHookContext
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


class TestBeforeToolCallHook:
    """Tests for before tool call hook functionality."""

    async def test_before_hook_modifies_arguments(self):
        """Test that before hook can modify tool arguments."""

        async def modify_args_hook(
            request: CallToolRequestSpec, context: ToolHookContext
        ) -> CallToolRequestSpec:
            # Double the arguments
            args = request.get("args", {})
            modified_args = {k: v * 2 for k, v in args.items()}

            return {
                "name": request.get("name"),
                "args": modified_args,
                "headers": request.get("headers", {}),
            }

        hooks = Hooks(before_tool_call=modify_args_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        # Original call would be 2 + 3 = 5, but hook doubles it to 4 + 6 = 10
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert "10" in str(result)

    async def test_before_hook_modifies_tool_name(self):
        """Test that before hook can redirect to different tool."""

        async def redirect_tool_hook(
            request: CallToolRequestSpec, context: ToolHookContext
        ) -> CallToolRequestSpec:
            # Redirect add to multiply
            if request.get("name") == "add":
                return {
                    "name": "multiply",
                    "args": request.get("args", {}),
                    "headers": request.get("headers", {}),
                }
            return request

        hooks = Hooks(before_tool_call=redirect_tool_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        # Call add but hook redirects to multiply: 5 * 2 = 10
        result = await add_tool.ainvoke({"a": 5, "b": 2})
        assert result == "10"

    async def test_before_hook_returns_none(self):
        """Test that before hook returning None uses original request."""

        async def no_op_hook(
            request: CallToolRequestSpec, context: ToolHookContext
        ) -> CallToolRequestSpec | None:
            return None

        hooks = Hooks(before_tool_call=no_op_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result == "5"

    async def test_before_hook_modifies_headers(self):
        """Test that before hook can modify headers (for applicable transports)."""

        async def modify_headers_hook(
            request: CallToolRequestSpec, context: ToolHookContext
        ) -> CallToolRequestSpec:
            return {
                "name": request.get("name"),
                "args": request.get("args", {}),
                "headers": {"X-Custom-Header": "test-value"},
            }

        hooks = Hooks(before_tool_call=modify_headers_hook)
        # For stdio transport, headers won't apply, but hook should still work
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result == "5"


class TestAfterToolCallHook:
    """Tests for after tool call hook functionality."""

    async def test_after_hook_modifies_result(self):
        """Test that after hook can modify tool result."""

        async def modify_result_hook(
            result: CallToolResult, context: ToolHookContext
        ) -> CallToolResult:
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

        hooks = Hooks(after_tool_call=modify_result_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        # The after hook modifies the result
        assert result == "Modified: 5"

    async def test_after_hook_returns_modified_result(self):
        """Test that after hook can return a completely custom CallToolResult."""

        async def return_custom_result_hook(
            result: CallToolResult, context: ToolHookContext
        ) -> CallToolResult:
            return CallToolResult(
                content=[TextContent(type="text", text="Custom tool response")],
                isError=False,
            )

        hooks = Hooks(after_tool_call=return_custom_result_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        # The after hook returns a custom result
        assert result == "Custom tool response"

    async def test_after_hook_returns_none(self):
        async def no_op_hook(
            result: CallToolResult, context: ToolHookContext
        ) -> CallToolResult | None:
            return None

        hooks = Hooks(after_tool_call=no_op_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result == "5"


class TestHookErrorHandling:
    """Tests for hook error handling."""

    async def test_before_hook_exception_propagates(self):
        """Test that exceptions in before hooks propagate correctly."""

        async def failing_hook(
            request: CallToolRequestSpec, context: ToolHookContext
        ) -> CallToolRequestSpec | None:
            raise ValueError("Hook failed")

        hooks = Hooks(before_tool_call=failing_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        with pytest.raises(ValueError, match="Hook failed"):
            await add_tool.ainvoke({"a": 2, "b": 3})

    async def test_after_hook_exception_propagates(self):
        """Test that exceptions in after hooks propagate correctly."""

        async def failing_hook(
            result: CallToolResult, context: ToolHookContext
        ) -> CallToolResult | None:
            raise RuntimeError("After hook failed")

        hooks = Hooks(after_tool_call=failing_hook)
        tools = await load_mcp_tools(
            None,
            connection=_get_math_server_config(),
            hooks=hooks,
        )

        add_tool = next(tool for tool in tools if tool.name == "add")
        with pytest.raises(RuntimeError, match="After hook failed"):
            await add_tool.ainvoke({"a": 2, "b": 3})
