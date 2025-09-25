"""Tests for callback functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, LoggingMessageNotificationParams, TextContent
from mcp.types import Tool as MCPTool

from langchain_mcp_adapters.callbacks import (
    CallbackContext,
    Callbacks,
    LoggingMessageCallback,
    ProgressCallback,
    _MCPCallbacks,
)
from langchain_mcp_adapters.tools import (
    convert_mcp_tool_to_langchain_tool,
    load_mcp_tools,
)


@pytest.mark.asyncio
async def test_to_mcp_format_with_both_callbacks() -> None:
    """Test converting to MCP format with both callbacks."""
    logging_callback = AsyncMock(spec=LoggingMessageCallback)
    progress_callback = AsyncMock(spec=ProgressCallback)
    callbacks = Callbacks(
        on_logging_message=logging_callback,
        on_progress=progress_callback,
    )
    context = CallbackContext(server_name="test_server", tool_name="test_tool")

    mcp_callbacks = callbacks.to_mcp_format(context=context)

    assert isinstance(mcp_callbacks, _MCPCallbacks)
    assert mcp_callbacks.logging_callback is not None
    assert mcp_callbacks.progress_callback is not None

    # Test logging callback
    params = LoggingMessageNotificationParams(
        level="info", data={"message": "test log"}
    )
    await mcp_callbacks.logging_callback(params)
    logging_callback.assert_called_once_with(params, context)

    # Test progress callback
    await mcp_callbacks.progress_callback(0.75, 1.0, "Almost done...")
    progress_callback.assert_called_once_with(0.75, 1.0, "Almost done...", context)


@pytest.mark.asyncio
async def test_progress_callback_execution():
    """Test that progress callbacks are properly executed during tool calls."""
    # Track progress calls
    progress_calls = []

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append(
            {
                "progress": progress,
                "total": total,
                "message": message,
                "context": context,
            }
        )

    callbacks = Callbacks(on_progress=progress_callback)
    context = CallbackContext(server_name="test_server", tool_name="test_tool")

    mcp_callbacks = callbacks.to_mcp_format(context=context)

    # Simulate progress updates
    await mcp_callbacks.progress_callback(0.0, 1.0, "Starting...")
    await mcp_callbacks.progress_callback(0.5, 1.0, "Halfway done...")
    await mcp_callbacks.progress_callback(1.0, 1.0, "Complete!")

    # Give the async tasks time to complete

    await asyncio.sleep(0.01)

    # Verify all progress calls were recorded
    assert len(progress_calls) == 3

    # Check first call
    assert progress_calls[0]["progress"] == 0.0
    assert progress_calls[0]["total"] == 1.0
    assert progress_calls[0]["message"] == "Starting..."
    assert progress_calls[0]["context"] == context

    # Check second call
    assert progress_calls[1]["progress"] == 0.5
    assert progress_calls[1]["total"] == 1.0
    assert progress_calls[1]["message"] == "Halfway done..."
    assert progress_calls[1]["context"] == context

    # Check third call
    assert progress_calls[2]["progress"] == 1.0
    assert progress_calls[2]["total"] == 1.0
    assert progress_calls[2]["message"] == "Complete!"
    assert progress_calls[2]["context"] == context


@pytest.mark.asyncio
async def test_progress_callback_with_none_values():
    """Test progress callbacks with None values."""
    progress_calls = []

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append(
            {
                "progress": progress,
                "total": total,
                "message": message,
                "context": context,
            }
        )

    callbacks = Callbacks(on_progress=progress_callback)
    context = CallbackContext(server_name="test_server")

    mcp_callbacks = callbacks.to_mcp_format(context=context)

    # Test with None total and message
    await mcp_callbacks.progress_callback(0.3, None, None)

    # Give the async task time to complete

    await asyncio.sleep(0.01)

    assert len(progress_calls) == 1
    assert progress_calls[0]["progress"] == 0.3
    assert progress_calls[0]["total"] is None
    assert progress_calls[0]["message"] is None
    assert progress_calls[0]["context"] == context


@pytest.mark.asyncio
async def test_logging_callback_execution():
    """Test that logging callbacks are properly executed."""
    # Track logging calls
    logging_calls = []

    async def logging_callback(
        params: LoggingMessageNotificationParams, context: CallbackContext
    ):
        logging_calls.append(
            {
                "params": params,
                "context": context,
            }
        )

    callbacks = Callbacks(on_logging_message=logging_callback)
    context = CallbackContext(server_name="test_server", tool_name="test_tool")

    mcp_callbacks = callbacks.to_mcp_format(context=context)

    # Simulate different log levels
    info_params = LoggingMessageNotificationParams(
        level="info", data={"message": "Info message"}
    )
    warning_params = LoggingMessageNotificationParams(
        level="warning", data={"message": "Warning message"}
    )
    error_params = LoggingMessageNotificationParams(
        level="error", data={"message": "Error message", "error": "Some error"}
    )

    await mcp_callbacks.logging_callback(info_params)
    await mcp_callbacks.logging_callback(warning_params)
    await mcp_callbacks.logging_callback(error_params)

    # Give the async tasks time to complete

    await asyncio.sleep(0.01)

    # Verify all logging calls were recorded
    assert len(logging_calls) == 3

    # Check info call
    assert logging_calls[0]["params"] == info_params
    assert logging_calls[0]["context"] == context

    # Check warning call
    assert logging_calls[1]["params"] == warning_params
    assert logging_calls[1]["context"] == context

    # Check error call
    assert logging_calls[2]["params"] == error_params
    assert logging_calls[2]["context"] == context


@pytest.mark.asyncio
async def test_logging_callback_with_complex_data():
    """Test logging callbacks with complex data structures."""
    logging_calls = []

    async def logging_callback(
        params: LoggingMessageNotificationParams, context: CallbackContext
    ):
        logging_calls.append(
            {
                "params": params,
                "context": context,
            }
        )

    callbacks = Callbacks(on_logging_message=logging_callback)
    context = CallbackContext(server_name="test_server")

    mcp_callbacks = callbacks.to_mcp_format(context=context)

    # Test with complex data structure
    complex_params = LoggingMessageNotificationParams(
        level="debug",
        data={
            "message": "Complex log entry",
            "metadata": {
                "user_id": 123,
                "action": "tool_execution",
                "duration": 1.5,
                "nested": {"key": "value"},
            },
            "timestamp": "2024-01-01T12:00:00Z",
        },
    )

    await mcp_callbacks.logging_callback(complex_params)

    # Give the async task time to complete

    await asyncio.sleep(0.01)

    assert len(logging_calls) == 1
    assert logging_calls[0]["params"] == complex_params
    assert logging_calls[0]["context"] == context


@pytest.mark.asyncio
async def test_callbacks_with_mcp_tool_execution():
    """Test that callbacks are properly called during MCP tool execution."""
    # Track all callback invocations
    progress_calls = []
    logging_calls = []

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append(
            {
                "progress": progress,
                "total": total,
                "message": message,
                "server_name": context.server_name,
                "tool_name": context.tool_name,
            }
        )

    async def logging_callback(
        params: LoggingMessageNotificationParams, context: CallbackContext
    ):
        logging_calls.append(
            {
                "level": params.level,
                "data": params.data,
                "server_name": context.server_name,
                "tool_name": context.tool_name,
            }
        )

    callbacks = Callbacks(
        on_progress=progress_callback,
        on_logging_message=logging_callback,
    )

    # Create a mock session that simulates progress and logging
    session = AsyncMock()

    async def mock_call_tool(tool_name, arguments, progress_callback=None):
        # Simulate progress updates
        if progress_callback:
            await progress_callback(0.0, 1.0, "Starting tool execution...")
            await progress_callback(0.5, 1.0, "Processing arguments...")
            await progress_callback(1.0, 1.0, "Tool execution complete!")

        return CallToolResult(
            content=[TextContent(type="text", text="Tool executed successfully")],
            isError=False,
        )

    session.call_tool.side_effect = mock_call_tool

    # Create an MCP tool
    mcp_tool = MCPTool(
        name="test_tool",
        description="A test tool for callback integration",
        inputSchema={
            "properties": {"input": {"title": "Input", "type": "string"}},
            "required": ["input"],
            "title": "TestToolSchema",
            "type": "object",
        },
    )

    # Convert to LangChain tool with callbacks
    lc_tool = convert_mcp_tool_to_langchain_tool(
        session,
        mcp_tool,
        callbacks=callbacks,
        server_name="test_server",
    )

    # Execute the tool
    result = await lc_tool.ainvoke({"input": "test input"})

    # Verify the tool executed successfully
    assert result == "Tool executed successfully"

    # Verify progress callbacks were called
    assert len(progress_calls) == 3
    assert progress_calls[0]["progress"] == 0.0
    assert progress_calls[0]["message"] == "Starting tool execution..."
    assert progress_calls[0]["server_name"] == "test_server"
    assert progress_calls[0]["tool_name"] == "test_tool"

    assert progress_calls[1]["progress"] == 0.5
    assert progress_calls[1]["message"] == "Processing arguments..."
    assert progress_calls[2]["progress"] == 1.0
    assert progress_calls[2]["message"] == "Tool execution complete!"


@pytest.mark.asyncio
async def test_callbacks_with_client_session():
    """Test callbacks integration with MultiServerMCPClient."""
    # Track callback invocations
    progress_calls = []
    logging_calls = []

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append(
            {
                "progress": progress,
                "total": total,
                "message": message,
                "server_name": context.server_name,
                "tool_name": context.tool_name,
            }
        )

    async def logging_callback(
        params: LoggingMessageNotificationParams, context: CallbackContext
    ):
        logging_calls.append(
            {
                "level": params.level,
                "data": params.data,
                "server_name": context.server_name,
                "tool_name": context.tool_name,
            }
        )

    callbacks = Callbacks(
        on_progress=progress_callback,
        on_logging_message=logging_callback,
    )

    # Create a mock session
    session = AsyncMock()

    # Mock list_tools response
    mcp_tools = [
        MCPTool(
            name="tool1",
            description="First tool",
            inputSchema={
                "properties": {"param": {"title": "Param", "type": "string"}},
                "required": ["param"],
                "title": "Tool1Schema",
                "type": "object",
            },
        ),
        MCPTool(
            name="tool2",
            description="Second tool",
            inputSchema={
                "properties": {"param": {"title": "Param", "type": "string"}},
                "required": ["param"],
                "title": "Tool2Schema",
                "type": "object",
            },
        ),
    ]
    session.list_tools.return_value = MagicMock(tools=mcp_tools, nextCursor=None)

    # Mock call_tool with progress simulation
    async def mock_call_tool(tool_name, arguments, progress_callback=None):
        if progress_callback:
            await progress_callback(0.0, 1.0, f"Starting {tool_name}...")
            await progress_callback(1.0, 1.0, f"{tool_name} complete!")

        return CallToolResult(
            content=[TextContent(type="text", text=f"{tool_name} result")],
            isError=False,
        )

    session.call_tool.side_effect = mock_call_tool

    # Load tools with callbacks
    tools = await load_mcp_tools(
        session,
        callbacks=callbacks,
        server_name="test_server",
    )

    # Verify tools were loaded
    assert len(tools) == 2

    # Execute first tool
    result1 = await tools[0].ainvoke({"param": "test1"})
    assert result1 == "tool1 result"

    # Execute second tool
    result2 = await tools[1].ainvoke({"param": "test2"})
    assert result2 == "tool2 result"

    # Verify progress callbacks were called for both tools
    assert len(progress_calls) == 4  # 2 progress updates per tool

    # Check first tool callbacks
    assert progress_calls[0]["message"] == "Starting tool1..."
    assert progress_calls[0]["server_name"] == "test_server"
    assert progress_calls[0]["tool_name"] == "tool1"
    assert progress_calls[1]["message"] == "tool1 complete!"

    # Check second tool callbacks
    assert progress_calls[2]["message"] == "Starting tool2..."
    assert progress_calls[2]["server_name"] == "test_server"
    assert progress_calls[2]["tool_name"] == "tool2"
    assert progress_calls[3]["message"] == "tool2 complete!"
