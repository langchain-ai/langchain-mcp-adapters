"""Tests for callback functionality."""

import asyncio

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp.types import LoggingMessageNotificationParams

from langchain_mcp_adapters.callbacks import (
    CallbackContext,
    Callbacks,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from tests.utils import run_streamable_http


async def test_to_mcp_format_with_callbacks() -> None:
    """Test converting to MCP format with callbacks."""
    logging_calls = []
    progress_calls = []

    async def logging_callback(
        params: LoggingMessageNotificationParams, context: CallbackContext
    ):
        logging_calls.append((params, context))

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append((progress, total, message, context))

    callbacks = Callbacks(
        on_logging_message=logging_callback,
        on_progress=progress_callback,
    )
    context = CallbackContext(server_name="test_server", tool_name="test_tool")

    mcp_callbacks = callbacks.to_mcp_format(context=context)

    assert mcp_callbacks.logging_callback is not None
    assert mcp_callbacks.progress_callback is not None

    # Test logging callback
    params = LoggingMessageNotificationParams(
        level="info", data={"message": "test log"}
    )
    await mcp_callbacks.logging_callback(params)
    assert len(logging_calls) == 1
    assert logging_calls[0][0] == params
    assert logging_calls[0][1].server_name == "test_server"

    # Test progress callback
    await mcp_callbacks.progress_callback(0.75, 1.0, "Almost done...")
    assert len(progress_calls) == 1
    assert progress_calls[0] == (0.75, 1.0, "Almost done...", context)


def _create_progress_server():
    """Create a server with a tool that reports progress."""
    server = FastMCP(port=8184)

    @server.tool()
    async def process_data(data: str, ctx: Context[ServerSession, None]) -> str:
        """Process data with progress reporting"""
        # Report progress at different stages
        await ctx.report_progress(progress=0.0, total=1.0)
        await asyncio.sleep(0.01)  # Small delay to ensure callback is processed

        await ctx.report_progress(progress=0.5, total=1.0)
        await asyncio.sleep(0.01)

        await ctx.report_progress(progress=1.0, total=1.0)
        await asyncio.sleep(0.01)

        return f"Processed: {data}"

    return server


async def test_progress_callback_execution(socket_enabled) -> None:
    """Test progress callback execution with real server."""
    progress_calls = []

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append((progress, total, message, context.server_name))

    callbacks = Callbacks(on_progress=progress_callback)

    with run_streamable_http(_create_progress_server, 8184):
        client = MultiServerMCPClient(
            {
                "progress_test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=callbacks,
        )

        tools = await client.get_tools(server_name="progress_test")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "process_data"

        result = await tool.ainvoke(
            {"args": {"data": "test"}, "id": "1", "type": "tool_call"}
        )
        assert "Processed: test" in result.content

        # Verify progress callbacks were called
        await asyncio.sleep(0.05)  # Give time for callbacks to complete
        assert len(progress_calls) >= 3, f"Expected at least 3 progress calls, got {len(progress_calls)}"
        # Verify we got progress updates at different stages
        progress_values = [call[0] for call in progress_calls]
        assert 0.0 in progress_values
        assert 1.0 in progress_values


def _create_logging_server():
    """Create a server with a tool that generates logs."""
    server = FastMCP(port=8185)

    @server.tool()
    async def analyze_data(data: str, ctx: Context[ServerSession, None]) -> str:
        """Analyze data with logging"""
        # Log at different levels
        await ctx.info(f"Starting analysis of: {data}")
        await asyncio.sleep(0.01)

        await ctx.debug("Processing data...")
        await asyncio.sleep(0.01)

        await ctx.info("Analysis complete")
        await asyncio.sleep(0.01)

        return f"Analyzed: {data}"

    return server


async def test_logging_callback_execution(socket_enabled) -> None:
    """Test logging callback execution with real server."""
    logging_calls = []

    async def logging_callback(
        params: LoggingMessageNotificationParams, context: CallbackContext
    ):
        logging_calls.append((params.level, params.data, context.server_name))

    callbacks = Callbacks(on_logging_message=logging_callback)

    with run_streamable_http(_create_logging_server, 8185):
        client = MultiServerMCPClient(
            {
                "logging_test": {
                    "url": "http://localhost:8185/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=callbacks,
        )

        tools = await client.get_tools(server_name="logging_test")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "analyze_data"

        result = await tool.ainvoke(
            {"args": {"data": "test"}, "id": "1", "type": "tool_call"}
        )
        assert "Analyzed: test" in result.content

        # Verify logging callbacks were called
        await asyncio.sleep(0.05)  # Give time for callbacks to complete
        assert len(logging_calls) >= 3, f"Expected at least 3 logging calls, got {len(logging_calls)}"
        # Verify we got different log levels
        log_levels = [call[0] for call in logging_calls]
        assert "info" in log_levels
        assert "debug" in log_levels


def _create_callback_server():
    """Create a server with a tool for testing callbacks."""
    server = FastMCP(port=8186)

    @server.tool()
    async def execute_task(task: str, ctx: Context[ServerSession, None]) -> str:
        """Execute a task with progress and logging"""
        await ctx.info(f"Starting task: {task}")
        await ctx.report_progress(progress=0.0, total=1.0)
        await asyncio.sleep(0.01)

        await ctx.debug("Executing task...")
        await ctx.report_progress(progress=0.5, total=1.0)
        await asyncio.sleep(0.01)

        await ctx.info(f"Completed task: {task}")
        await ctx.report_progress(progress=1.0, total=1.0)
        await asyncio.sleep(0.01)

        return f"Executed: {task}"

    return server


async def test_callbacks_with_mcp_tool_execution(socket_enabled) -> None:
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

    with run_streamable_http(_create_callback_server, 8186):
        client = MultiServerMCPClient(
            {
                "callback_test": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=callbacks,
        )

        tools = await client.get_tools(server_name="callback_test")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "execute_task"

        result = await tool.ainvoke(
            {"args": {"task": "test"}, "id": "1", "type": "tool_call"}
        )
        assert "Executed: test" in result.content

        # Verify both progress and logging callbacks were called
        await asyncio.sleep(0.05)  # Give time for callbacks to complete
        raise ValueError(progress_calls)
        assert len(progress_calls) >= 3, f"Expected at least 3 progress calls, got {len(progress_calls)}"
        assert len(logging_calls) >= 2, f"Expected at least 2 logging calls, got {len(logging_calls)}"

        # Verify progress values
        progress_values = [call[0] for call in progress_calls]
        assert 0.0 in progress_values
        assert 1.0 in progress_values

        # Verify log levels
        log_levels = [call[0] for call in logging_calls]
        assert "info" in log_levels


def _create_multi_tool_server():
    """Create a server with multiple tools for testing load_mcp_tools."""
    server = FastMCP(port=8187)

    @server.tool()
    async def tool1(input_data: str, ctx: Context[ServerSession, None]) -> str:
        """First test tool"""
        await ctx.report_progress(progress=0.5, total=1.0)
        await asyncio.sleep(0.01)
        await ctx.report_progress(progress=1.0, total=1.0)
        await asyncio.sleep(0.01)
        return f"tool1 result: {input_data}"

    @server.tool()
    async def tool2(input_data: str, ctx: Context[ServerSession, None]) -> str:
        """Second test tool"""
        await ctx.report_progress(progress=1.0, total=1.0)
        await asyncio.sleep(0.01)
        return f"tool2 result: {input_data}"

    return server


async def test_callbacks_with_load_mcp_tools(socket_enabled) -> None:
    """Test callbacks integration with load_mcp_tools."""
    progress_calls = []

    async def progress_callback(progress, total, message, context):
        progress_calls.append((context.tool_name, message))

    callbacks = Callbacks(on_progress=progress_callback)

    with run_streamable_http(_create_multi_tool_server, 8187):
        client = MultiServerMCPClient(
            {
                "multi_tool_test": {
                    "url": "http://localhost:8187/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=callbacks,
        )

        tools = await client.get_tools(server_name="multi_tool_test")
        assert len(tools) == 2

        # Test first tool
        result1 = await tools[0].ainvoke(
            {"args": {"input_data": "test1"}, "id": "1", "type": "tool_call"}
        )
        assert "tool1 result: test1" in result1.content

        # Test second tool
        result2 = await tools[1].ainvoke(
            {"args": {"input_data": "test2"}, "id": "2", "type": "tool_call"}
        )
        assert "tool2 result: test2" in result2.content

        # Verify progress callbacks were called for both tools
        await asyncio.sleep(0.05)  # Give time for callbacks to complete
        assert len(progress_calls) >= 3, f"Expected at least 3 progress calls, got {len(progress_calls)}"

        # Verify both tools reported progress
        tool_names = [call[0] for call in progress_calls]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
