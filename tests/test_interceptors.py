"""Test interceptor functionality for MCP tool calls."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import (
    AfterToolCallHook,
    BeforeToolCallHook,
    InterceptorConfig,
    MessageEvent,
    OnMessageCallback,
    OnProgressCallback,
    ProgressEvent,
    ToolCallArgs,
    ToolCallResult,
)
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import Tool as MCPTool
from pydantic import BaseModel


class MockToolSchema(BaseModel):
    """Mock schema for testing."""
    input_value: str


@pytest.fixture
def mock_mcp_tool() -> MCPTool:
    """Create a mock MCP tool for testing."""
    return MCPTool(
        name="test_tool",
        description="A test tool",
        inputSchema=MockToolSchema,
    )


@pytest.fixture
def mock_session():
    """Create a mock session."""
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="test result")],
        isError=False,
    )
    return session


@pytest.mark.asyncio
async def test_before_tool_call_hook(mock_mcp_tool, mock_session):
    """Test that before-tool-call hook modifies arguments."""

    def before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
        # Modify the arguments
        tool_call["arguments"]["injected_param"] = "injected_value"
        tool_call["headers"] = {"Authorization": "Bearer token"}
        return tool_call

    interceptors: InterceptorConfig = {
        "before_tool_call": before_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    # Call the tool
    await tool.ainvoke({"input_value": "test"})

    # Verify the session was called with modified arguments
    mock_session.call_tool.assert_called_once_with(
        "test_tool",
        {"input_value": "test", "injected_param": "injected_value"}
    )


@pytest.mark.asyncio
async def test_after_tool_call_hook_dict_format(mock_mcp_tool, mock_session):
    """Test after-tool-call hook with dict return format."""

    def after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> ToolCallResult:
        return {
            "content": f"Modified: {result.content[0].text}",
            "artifacts": ["test_artifact"],
        }

    interceptors: InterceptorConfig = {
        "after_tool_call": after_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    # Call the tool
    result = await tool.ainvoke({"input_value": "test"})

    # Verify the result was modified
    assert result == ("Modified: test result", ["test_artifact"])


@pytest.mark.asyncio
async def test_after_tool_call_hook_tuple_format(mock_mcp_tool, mock_session):
    """Test after-tool-call hook with tuple return format."""

    def after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> tuple[str, list]:
        return f"Tuple: {result.content[0].text}", ["tuple_artifact"]

    interceptors: InterceptorConfig = {
        "after_tool_call": after_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    # Call the tool
    result = await tool.ainvoke({"input_value": "test"})

    # Verify the result was modified
    assert result == ("Tuple: test result", ["tuple_artifact"])


@pytest.mark.asyncio
async def test_after_tool_call_hook_string_format(mock_mcp_tool, mock_session):
    """Test after-tool-call hook with string return format."""

    def after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> str:
        return f"String: {result.content[0].text}"

    interceptors: InterceptorConfig = {
        "after_tool_call": after_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    # Call the tool
    result = await tool.ainvoke({"input_value": "test"})

    # Verify the result was modified
    assert result == ("String: test result", None)


@pytest.mark.asyncio
async def test_async_before_hook(mock_mcp_tool, mock_session):
    """Test async before-tool-call hook."""

    async def async_before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
        # Simulate async operation
        await asyncio.sleep(0.01)
        tool_call["arguments"]["async_param"] = "async_value"
        return tool_call

    interceptors: InterceptorConfig = {
        "before_tool_call": async_before_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    # Call the tool
    await tool.ainvoke({"input_value": "test"})

    # Verify the session was called with modified arguments
    mock_session.call_tool.assert_called_once_with(
        "test_tool",
        {"input_value": "test", "async_param": "async_value"}
    )


@pytest.mark.asyncio
async def test_async_after_hook(mock_mcp_tool, mock_session):
    """Test async after-tool-call hook."""

    async def async_after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> str:
        # Simulate async operation
        await asyncio.sleep(0.01)
        return f"Async: {result.content[0].text}"

    interceptors: InterceptorConfig = {
        "after_tool_call": async_after_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    # Call the tool
    result = await tool.ainvoke({"input_value": "test"})

    # Verify the result was modified
    assert result == ("Async: test result", None)


@pytest.mark.asyncio
async def test_no_interceptors(mock_mcp_tool, mock_session):
    """Test tool works normally without interceptors."""

    tool = convert_mcp_tool_to_langchain_tool(mock_session, mock_mcp_tool)

    # Call the tool
    result = await tool.ainvoke({"input_value": "test"})

    # Verify the default behavior
    assert result == ("test result", None)
    mock_session.call_tool.assert_called_once_with("test_tool", {"input_value": "test"})


def test_callback_type_annotations():
    """Test that callback types are properly defined."""

    def message_callback(event: MessageEvent) -> None:
        assert "level" in event
        assert "message" in event

    def progress_callback(event: ProgressEvent) -> None:
        assert isinstance(event, dict)

    async def async_message_callback(event: MessageEvent) -> None:
        assert "level" in event
        assert "message" in event

    # Test that the callbacks match the expected protocol
    callbacks: InterceptorConfig = {
        "on_message": message_callback,
        "on_progress": progress_callback,
    }

    assert "on_message" in callbacks
    assert "on_progress" in callbacks


def test_interceptor_config_types():
    """Test that InterceptorConfig accepts all expected types."""

    def before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
        return tool_call

    def after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> str:
        return "test"

    def message_callback(event: MessageEvent) -> None:
        pass

    def progress_callback(event: ProgressEvent) -> None:
        pass

    def tools_changed_callback(tools: list[str]) -> None:
        pass

    # Test creating a full interceptor config
    config: InterceptorConfig = {
        "before_tool_call": before_hook,
        "after_tool_call": after_hook,
        "on_message": message_callback,
        "on_progress": progress_callback,
        "on_tools_list_changed": tools_changed_callback,
        "on_prompts_list_changed": tools_changed_callback,
        "on_resources_list_changed": tools_changed_callback,
    }

    assert len(config) == 7


@pytest.mark.asyncio
async def test_complex_interceptor_chain(mock_mcp_tool, mock_session):
    """Test complex interceptor chain with both hooks."""

    def before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
        tool_call["arguments"]["step1"] = "before_applied"
        tool_call["headers"] = {"X-Custom": "test"}
        return tool_call

    def after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> ToolCallResult:
        return {
            "content": f"Chain result: {result.content[0].text} + {tool_call['arguments']['step1']}",
            "artifacts": [{"chain": True}],
        }

    interceptors: InterceptorConfig = {
        "before_tool_call": before_hook,
        "after_tool_call": after_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    result = await tool.ainvoke({"input_value": "test"})

    # Verify both hooks were applied
    mock_session.call_tool.assert_called_once_with(
        "test_tool",
        {"input_value": "test", "step1": "before_applied"}
    )
    assert result == ("Chain result: test result + before_applied", [{"chain": True}])


@pytest.mark.asyncio
async def test_error_handling_in_hooks(mock_mcp_tool, mock_session):
    """Test error handling when hooks raise exceptions."""

    def failing_before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
        raise ValueError("Before hook failed")

    interceptors: InterceptorConfig = {
        "before_tool_call": failing_before_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    # The hook failure should propagate
    with pytest.raises(ValueError, match="Before hook failed"):
        await tool.ainvoke({"input_value": "test"})


@pytest.mark.asyncio
async def test_tool_message_format_in_after_hook(mock_mcp_tool, mock_session):
    """Test after-hook returning ToolMessage format."""
    from langchain_core.messages import ToolMessage

    def tool_message_after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> ToolMessage:
        return ToolMessage(
            content=f"ToolMessage: {result.content[0].text}",
            tool_call_id="test_id",
        )

    interceptors: InterceptorConfig = {
        "after_tool_call": tool_message_after_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    result = await tool.ainvoke({"input_value": "test"})

    # Should extract content from ToolMessage
    assert result == ("ToolMessage: test result", None)


@pytest.mark.asyncio
async def test_hook_with_connection_creation(mock_mcp_tool):
    """Test hooks work when creating connection on-the-fly."""
    from unittest.mock import patch, AsyncMock

    def before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
        tool_call["arguments"]["connection_created"] = True
        return tool_call

    interceptors: InterceptorConfig = {
        "before_tool_call": before_hook,
    }

    connection = {
        "transport": "stdio",
        "command": "test",
        "args": ["test"],
    }

    # Mock the create_session function
    with patch('langchain_mcp_adapters.tools.create_session') as mock_create_session:
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = CallToolResult(
            content=[TextContent(type="text", text="connection result")],
            isError=False,
        )
        mock_create_session.return_value.__aenter__.return_value = mock_session

        tool = convert_mcp_tool_to_langchain_tool(
            None, mock_mcp_tool, connection=connection, interceptors=interceptors
        )

        result = await tool.ainvoke({"input_value": "test"})

        # Verify the hook was applied and connection was used
        mock_session.call_tool.assert_called_once_with(
            "test_tool",
            {"input_value": "test", "connection_created": True}
        )
        assert result == ("connection result", None)


@pytest.mark.asyncio
async def test_custom_result_format_handling(mock_mcp_tool, mock_session):
    """Test handling of custom result formats from after-hook."""

    def custom_after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> dict:
        # Return a custom object that should be stringified
        return {"custom": "format", "data": [1, 2, 3]}

    interceptors: InterceptorConfig = {
        "after_tool_call": custom_after_hook,
    }

    tool = convert_mcp_tool_to_langchain_tool(
        mock_session, mock_mcp_tool, interceptors=interceptors
    )

    result = await tool.ainvoke({"input_value": "test"})

    # Should convert custom format to string
    assert result == ("{'custom': 'format', 'data': [1, 2, 3]}", None)


class TestMultiServerMCPClientWithInterceptors:
    """Test MultiServerMCPClient with interceptor functionality."""

    def test_client_init_with_interceptors(self):
        """Test that client accepts interceptors in initialization."""

        def before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
            return tool_call

        interceptors: InterceptorConfig = {
            "before_tool_call": before_hook,
        }

        client = MultiServerMCPClient(
            connections={},
            interceptors=interceptors,
        )

        assert client.interceptors == interceptors

    def test_client_init_without_interceptors(self):
        """Test that client works without interceptors."""

        client = MultiServerMCPClient(connections={})

        assert client.interceptors == {}

    @pytest.mark.asyncio
    async def test_get_tools_passes_interceptors(self):
        """Test that get_tools passes interceptors to load_mcp_tools."""
        from unittest.mock import patch, AsyncMock

        def test_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
            return tool_call

        interceptors: InterceptorConfig = {
            "before_tool_call": test_hook,
        }

        connections = {
            "test": {
                "transport": "stdio",
                "command": "test",
                "args": ["test"],
            }
        }

        client = MultiServerMCPClient(
            connections=connections,
            interceptors=interceptors,
        )

        # Mock load_mcp_tools to verify interceptors are passed
        with patch('langchain_mcp_adapters.client.load_mcp_tools') as mock_load_tools:
            mock_load_tools.return_value = []

            await client.get_tools()

            # Verify load_mcp_tools was called with interceptors
            mock_load_tools.assert_called_once_with(
                None, connection=connections["test"], interceptors=interceptors
            )

    @pytest.mark.asyncio
    async def test_get_tools_specific_server_passes_interceptors(self):
        """Test that get_tools with server_name passes interceptors."""
        from unittest.mock import patch

        def test_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
            return tool_call

        interceptors: InterceptorConfig = {
            "before_tool_call": test_hook,
        }

        connections = {
            "server1": {
                "transport": "stdio",
                "command": "test1",
                "args": ["test1"],
            },
            "server2": {
                "transport": "stdio",
                "command": "test2",
                "args": ["test2"],
            },
        }

        client = MultiServerMCPClient(
            connections=connections,
            interceptors=interceptors,
        )

        # Mock load_mcp_tools
        with patch('langchain_mcp_adapters.client.load_mcp_tools') as mock_load_tools:
            mock_load_tools.return_value = []

            await client.get_tools(server_name="server1")

            # Verify load_mcp_tools was called with correct server and interceptors
            mock_load_tools.assert_called_once_with(
                None, connection=connections["server1"], interceptors=interceptors
            )


class TestCallbackFunctionality:
    """Test notification callback functionality."""

    def test_message_callback_structure(self):
        """Test message callback receives correct structure."""
        received_events = []

        def message_callback(event: MessageEvent) -> None:
            received_events.append(event)

        # Simulate callback usage
        test_event: MessageEvent = {
            "level": "info",
            "message": "Test message",
            "source": "test_server",
        }

        message_callback(test_event)

        assert len(received_events) == 1
        assert received_events[0]["level"] == "info"
        assert received_events[0]["message"] == "Test message"
        assert received_events[0]["source"] == "test_server"

    def test_progress_callback_structure(self):
        """Test progress callback receives correct structure."""
        received_events = []

        def progress_callback(event: ProgressEvent) -> None:
            received_events.append(event)

        # Simulate callback usage
        test_event: ProgressEvent = {
            "progress": 75.5,
            "source": "tool_execution",
            "total": 100,
            "completed": 75,
        }

        progress_callback(test_event)

        assert len(received_events) == 1
        assert received_events[0]["progress"] == 75.5
        assert received_events[0]["source"] == "tool_execution"
        assert received_events[0]["total"] == 100
        assert received_events[0]["completed"] == 75

    @pytest.mark.asyncio
    async def test_async_callbacks(self):
        """Test that async callbacks work correctly."""
        received_events = []

        async def async_message_callback(event: MessageEvent) -> None:
            await asyncio.sleep(0.01)  # Simulate async work
            received_events.append(event)

        # Test that we can call the async callback
        test_event: MessageEvent = {
            "level": "error",
            "message": "Async test",
        }

        await async_message_callback(test_event)

        assert len(received_events) == 1
        assert received_events[0]["level"] == "error"
        assert received_events[0]["message"] == "Async test"


class TestInterceptorIntegration:
    """Test integration scenarios with interceptors."""

    @pytest.mark.asyncio
    async def test_full_interceptor_workflow(self, mock_mcp_tool, mock_session):
        """Test complete workflow with all interceptor types."""
        workflow_log = []

        def before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
            workflow_log.append(f"before: {tool_call['name']}")
            tool_call["arguments"]["workflow"] = "started"
            return tool_call

        def after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> ToolCallResult:
            workflow_log.append(f"after: {tool_call['name']}")
            return {
                "content": f"Workflow: {result.content[0].text}",
                "artifacts": [{"workflow": "completed"}],
            }

        def message_callback(event: MessageEvent) -> None:
            workflow_log.append(f"message: {event['message']}")

        def progress_callback(event: ProgressEvent) -> None:
            workflow_log.append(f"progress: {event.get('progress', 0)}")

        interceptors: InterceptorConfig = {
            "before_tool_call": before_hook,
            "after_tool_call": after_hook,
            "on_message": message_callback,
            "on_progress": progress_callback,
        }

        tool = convert_mcp_tool_to_langchain_tool(
            mock_session, mock_mcp_tool, interceptors=interceptors
        )

        result = await tool.ainvoke({"input_value": "test"})

        # Verify workflow executed correctly
        assert "before: test_tool" in workflow_log
        assert "after: test_tool" in workflow_log
        assert result == ("Workflow: test result", [{"workflow": "completed"}])

        # Verify session called with modified arguments
        mock_session.call_tool.assert_called_once_with(
            "test_tool",
            {"input_value": "test", "workflow": "started"}
        )

    def test_interceptor_config_partial(self):
        """Test that partial interceptor configs work."""
        # Only before hook
        config1: InterceptorConfig = {
            "before_tool_call": lambda tc: tc,
        }
        assert "before_tool_call" in config1
        assert len(config1) == 1

        # Only callbacks
        config2: InterceptorConfig = {
            "on_message": lambda event: None,
            "on_progress": lambda event: None,
        }
        assert "on_message" in config2
        assert "on_progress" in config2
        assert len(config2) == 2

        # Mixed configuration
        config3: InterceptorConfig = {
            "after_tool_call": lambda tc, result: "test",
            "on_tools_list_changed": lambda tools: None,
        }
        assert "after_tool_call" in config3
        assert "on_tools_list_changed" in config3
        assert len(config3) == 2


class TestErrorScenarios:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_before_hook_exception_propagation(self, mock_mcp_tool, mock_session):
        """Test that exceptions in before hooks are properly propagated."""

        def failing_before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
            raise RuntimeError("Before hook critical error")

        interceptors: InterceptorConfig = {
            "before_tool_call": failing_before_hook,
        }

        tool = convert_mcp_tool_to_langchain_tool(
            mock_session, mock_mcp_tool, interceptors=interceptors
        )

        with pytest.raises(RuntimeError, match="Before hook critical error"):
            await tool.ainvoke({"input_value": "test"})

        # Session should not be called if before hook fails
        mock_session.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_after_hook_exception_propagation(self, mock_mcp_tool, mock_session):
        """Test that exceptions in after hooks are properly propagated."""

        def failing_after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> str:
            raise RuntimeError("After hook critical error")

        interceptors: InterceptorConfig = {
            "after_tool_call": failing_after_hook,
        }

        tool = convert_mcp_tool_to_langchain_tool(
            mock_session, mock_mcp_tool, interceptors=interceptors
        )

        with pytest.raises(RuntimeError, match="After hook critical error"):
            await tool.ainvoke({"input_value": "test"})

        # Session should be called but after hook fails
        mock_session.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_hook_exception_propagation(self, mock_mcp_tool, mock_session):
        """Test that exceptions in async hooks are properly propagated."""

        async def failing_async_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
            await asyncio.sleep(0.01)
            raise ValueError("Async hook failed")

        interceptors: InterceptorConfig = {
            "before_tool_call": failing_async_hook,
        }

        tool = convert_mcp_tool_to_langchain_tool(
            mock_session, mock_mcp_tool, interceptors=interceptors
        )

        with pytest.raises(ValueError, match="Async hook failed"):
            await tool.ainvoke({"input_value": "test"})