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
async def test_to_mcp_format_with_callbacks() -> None:
    """Test converting to MCP format with callbacks."""
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
async def test_progress_callback_execution() -> None:
    """Test progress callback execution with various values."""
    progress_calls = []

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append((progress, total, message, context.server_name))

    callbacks = Callbacks(on_progress=progress_callback)
    context = CallbackContext(server_name="test_server", tool_name="test_tool")
    mcp_callbacks = callbacks.to_mcp_format(context=context)

    await mcp_callbacks.progress_callback(0.0, 1.0, "Starting...")
    await mcp_callbacks.progress_callback(0.5, None, None)
    await mcp_callbacks.progress_callback(1.0, 1.0, "Complete!")

    await asyncio.sleep(0.01)

    assert len(progress_calls) == 3
    assert progress_calls[0] == (0.0, 1.0, "Starting...", "test_server")
    assert progress_calls[1] == (0.5, None, None, "test_server")
    assert progress_calls[2] == (1.0, 1.0, "Complete!", "test_server")


@pytest.mark.asyncio
async def test_logging_callback_execution() -> None:
    """Test logging callback execution with different levels."""
    logging_calls = []

    async def logging_callback(
        params: LoggingMessageNotificationParams, context: CallbackContext
    ):
        logging_calls.append((params.level, params.data, context.server_name))

    callbacks = Callbacks(on_logging_message=logging_callback)
    context = CallbackContext(server_name="test_server", tool_name="test_tool")
    mcp_callbacks = callbacks.to_mcp_format(context=context)

    await mcp_callbacks.logging_callback(
        LoggingMessageNotificationParams(level="info", data={"message": "Info"})
    )
    await mcp_callbacks.logging_callback(
        LoggingMessageNotificationParams(
            level="error", data={"message": "Error", "metadata": {"key": "value"}}
        )
    )

    await asyncio.sleep(0.01)

    assert len(logging_calls) == 2
    assert logging_calls[0] == ("info", {"message": "Info"}, "test_server")
    assert logging_calls[1] == (
        "error",
        {"message": "Error", "metadata": {"key": "value"}},
        "test_server",
    )


@pytest.mark.asyncio
async def test_callbacks_with_mcp_tool_execution() -> None:
    """Test callbacks integration during MCP tool execution."""
    progress_calls = []
    logging_calls = []

    async def progress_callback(progress, total, message, context):
        progress_calls.append((progress, message, context.tool_name))

    async def logging_callback(params, context):
        logging_calls.append((params.level, context.tool_name))

    callbacks = Callbacks(
        on_progress=progress_callback,
        on_logging_message=logging_callback,
    )

    session = AsyncMock()

    async def mock_call_tool(tool_name, arguments, progress_callback=None):
        if progress_callback:
            await progress_callback(0.0, 1.0, "Starting...")
            await progress_callback(1.0, 1.0, "Complete!")

        return CallToolResult(
            content=[TextContent(type="text", text="Success")],
            isError=False,
        )

    session.call_tool.side_effect = mock_call_tool

    mcp_tool = MCPTool(
        name="test_tool",
        description="Test tool",
        inputSchema={
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "type": "object",
        },
    )

    lc_tool = convert_mcp_tool_to_langchain_tool(
        session,
        mcp_tool,
        callbacks=callbacks,
        server_name="test_server",
    )

    result = await lc_tool.ainvoke({"input": "test"})

    assert result == "Success"
    assert len(progress_calls) == 2
    assert progress_calls[0] == (0.0, "Starting...", "test_tool")
    assert progress_calls[1] == (1.0, "Complete!", "test_tool")


@pytest.mark.asyncio
async def test_callbacks_with_load_mcp_tools() -> None:
    """Test callbacks integration with load_mcp_tools."""
    progress_calls = []

    async def progress_callback(progress, total, message, context):
        progress_calls.append((context.tool_name, message))

    callbacks = Callbacks(on_progress=progress_callback)

    session = AsyncMock()
    session.list_tools.return_value = MagicMock(
        tools=[
            MCPTool(
                name="tool1",
                description="Tool 1",
                inputSchema={"type": "object", "properties": {}},
            )
        ],
        nextCursor=None,
    )

    async def mock_call_tool(tool_name, arguments, progress_callback=None):
        if progress_callback:
            await progress_callback(1.0, 1.0, f"{tool_name} done")
        return CallToolResult(
            content=[TextContent(type="text", text=f"{tool_name} result")],
            isError=False,
        )

    session.call_tool.side_effect = mock_call_tool

    tools = await load_mcp_tools(
        session,
        callbacks=callbacks,
        server_name="test_server",
    )

    assert len(tools) == 1

    result = await tools[0].ainvoke({})
    assert result == "tool1 result"
    assert len(progress_calls) == 1
    assert progress_calls[0] == ("tool1", "tool1 done")
