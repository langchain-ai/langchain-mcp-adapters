"""Integration tests for hooks and callbacks in the MCP client."""

from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import (
    CallToolResult,
    LoggingMessageNotification,
    LoggingMessageNotificationParams,
    ProgressNotification,
    ProgressNotificationParams,
    TextContent,
)
from mcp.types import (
    Tool as MCPTool,
)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.lifecycle import (
    CallbackContext,
    Callbacks,
    Hooks,
)


class TestClientHooksIntegration:
    """Test hooks integration with actual tool calls."""

    @pytest.mark.asyncio
    async def test_before_tool_call_hook_modification(self):
        """Test that before_tool_call hook can modify tool call parameters."""
        # Track hook calls
        hook_calls = []

        async def before_hook(request, context):
            hook_calls.append(
                {
                    "tool_name": request.params.name,
                    "server_name": context.server_name,
                    "original_args": request.params.arguments.copy(),
                }
            )
            # Modify the arguments
            return {"args": {"injected_param": "test_value"}}

        hooks = Hooks(before_tool_call=before_hook)

        # Mock the MCP session and its methods
        with patch(
            "langchain_mcp_adapters.tools.create_session"
        ) as mock_create_session:
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.call_tool = AsyncMock()

            # Mock the tool call result
            mock_result = CallToolResult(
                content=[TextContent(type="text", text="test result")], isError=False
            )
            mock_session.call_tool.return_value = mock_result

            # Mock list_tools to return a test tool

            test_tool = MCPTool(
                name="test_tool",
                description="A test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"param1": {"type": "string"}},
                },
            )

            with patch(
                "langchain_mcp_adapters.tools._list_all_tools"
            ) as mock_list_tools:
                mock_list_tools.return_value = [test_tool]

                mock_create_session.return_value.__aenter__.return_value = mock_session
                mock_create_session.return_value.__aexit__.return_value = None

                # Create client with hooks
                client = MultiServerMCPClient(
                    connections={
                        "test_server": {"command": "echo", "transport": "stdio"}
                    },
                    hooks=hooks,
                )

                # Get tools (this creates the LangChain tools with hooks)
                tools = await client.get_tools(server_name="test_server")

                # Execute the tool
                if tools:
                    tool = tools[0]
                    await tool.ainvoke({"param1": "original_value"})

                    # Verify the hook was called
                    assert len(hook_calls) == 1
                    assert hook_calls[0]["tool_name"] == "test_tool"
                    assert hook_calls[0]["server_name"] == "test_server"
                    assert hook_calls[0]["original_args"] == {
                        "param1": "original_value"
                    }

                    # Verify the tool was called with modified arguments
                    mock_session.call_tool.assert_called_once()
                    call_args = mock_session.call_tool.call_args
                    assert (
                        "injected_param" in call_args[0][1]
                    )  # Second argument is the arguments dict
                    assert call_args[0][1]["injected_param"] == "test_value"

    @pytest.mark.asyncio
    async def test_after_tool_call_hook_modification(self):
        """Test that after_tool_call hook can modify tool results."""
        # Track hook calls
        hook_calls = []

        async def after_hook(result, context):
            hook_calls.append(
                {
                    "server_name": context.server_name,
                    "original_content": [
                        c.text for c in result.content if hasattr(c, "text")
                    ],
                }
            )
            # Return modified result
            return "Modified result from hook"

        hooks = Hooks(after_tool_call=after_hook)

        # Mock the MCP session and its methods
        with patch(
            "langchain_mcp_adapters.tools.create_session"
        ) as mock_create_session:
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.call_tool = AsyncMock()

            # Mock the tool call result

            mock_result = CallToolResult(
                content=[TextContent(type="text", text="original result")],
                isError=False,
            )
            mock_session.call_tool.return_value = mock_result

            # Mock list_tools to return a test tool

            test_tool = MCPTool(
                name="test_tool",
                description="A test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"param1": {"type": "string"}},
                },
            )

            with patch(
                "langchain_mcp_adapters.tools._list_all_tools"
            ) as mock_list_tools:
                mock_list_tools.return_value = [test_tool]

                mock_create_session.return_value.__aenter__.return_value = mock_session
                mock_create_session.return_value.__aexit__.return_value = None

                # Create client with hooks
                client = MultiServerMCPClient(
                    connections={
                        "test_server": {"command": "echo", "transport": "stdio"}
                    },
                    hooks=hooks,
                )

                # Get tools
                tools = await client.get_tools(server_name="test_server")

                # Execute the tool
                if tools:
                    tool = tools[0]
                    result = await tool.ainvoke({"param1": "test_value"})

                    # Verify the hook was called
                    assert len(hook_calls) == 1
                    assert hook_calls[0]["server_name"] == "test_server"
                    assert hook_calls[0]["original_content"] == ["original result"]

                    # Verify the result was modified
                    # When using response_format='content_and_artifact',
                    # LangChain returns just the content part
                    assert result == "Modified result from hook"

    @pytest.mark.asyncio
    async def test_callbacks_integration(self):
        """Test that callbacks are properly integrated."""
        # Track callback calls
        log_calls = []
        progress_calls = []

        async def on_logging_message(notification, context):
            log_calls.append(
                {
                    "level": notification.params.level,
                    "message": notification.params.data,
                    "context_state": context.state,
                }
            )

        async def on_progress_notification(notification, context):
            progress_calls.append(
                {
                    "progress": notification.params.progress,
                    "total": notification.params.total,
                    "context_state": context.state,
                }
            )

        callbacks = Callbacks(
            on_logging_message=on_logging_message,
            on_progress_notification=on_progress_notification,
        )

        # Create client with callbacks
        client = MultiServerMCPClient(
            connections={"test_server": {"command": "echo", "transport": "stdio"}},
            callbacks=callbacks,
        )

        # Test callback handling directly

        # Test logging callback
        log_notification = LoggingMessageNotification(
            method="notifications/message",
            params=LoggingMessageNotificationParams(
                level="info",
                logger="test",
                data="test message",
            ),
        )
        context = CallbackContext(server_name="test-server", state={"test": "value"})
        await client.handle_logging_message(log_notification, context)

        # Test progress callback
        progress_notification = ProgressNotification(
            method="notifications/progress",
            params=ProgressNotificationParams(
                progressToken="test-token",
                progress=50,
                total=100,
            ),
        )
        await client.handle_progress_notification(progress_notification, context)

        # Verify callbacks were called
        assert len(log_calls) == 1
        assert log_calls[0]["level"] == "info"
        assert log_calls[0]["message"] == "test message"
        assert log_calls[0]["context_state"] == {"test": "value"}

        assert len(progress_calls) == 1
        assert progress_calls[0]["progress"] == 50
        assert progress_calls[0]["total"] == 100
        assert progress_calls[0]["context_state"] == {"test": "value"}

    def test_hook_error_resilience(self):
        """Test that hook errors don't break the client."""

        async def failing_hook(request, context):
            raise RuntimeError("Hook failed")

        hooks = Hooks(before_tool_call=failing_hook)

        # Should not raise an exception during initialization
        client = MultiServerMCPClient(
            connections={"test_server": {"command": "echo", "transport": "stdio"}},
            hooks=hooks,
        )

        assert client.hooks.before_tool_call == failing_hook

    def test_callback_error_resilience(self):
        """Test that callback errors don't break the client."""

        async def failing_callback(notification, context):
            raise RuntimeError("Callback failed")

        callbacks = Callbacks(on_logging_message=failing_callback)

        # Should not raise an exception during initialization
        client = MultiServerMCPClient(
            connections={"test_server": {"command": "echo", "transport": "stdio"}},
            callbacks=callbacks,
        )

        assert client.callbacks.on_logging_message == failing_callback
