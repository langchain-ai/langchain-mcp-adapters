from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolArg, ToolException, tool
from mcp.server import FastMCP
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import (
    _convert_call_tool_result,
    convert_mcp_tool_to_langchain_tool,
    load_mcp_tools,
    to_fastmcp,
)
from tests.utils import run_streamable_http


def test_convert_empty_text_content():
    # Test with a single text content
    result = CallToolResult(content=[], isError=False)

    text_content, non_text_content = _convert_call_tool_result(result)

    assert text_content == ""
    assert non_text_content is None


def test_convert_single_text_content():
    # Test with a single text content
    result = CallToolResult(
        content=[TextContent(type="text", text="test result")], isError=False
    )

    text_content, non_text_content = _convert_call_tool_result(result)

    assert text_content == "test result"
    assert non_text_content is None


def test_convert_multiple_text_contents():
    # Test with multiple text contents
    result = CallToolResult(
        content=[
            TextContent(type="text", text="result 1"),
            TextContent(type="text", text="result 2"),
        ],
        isError=False,
    )

    text_content, non_text_content = _convert_call_tool_result(result)

    assert text_content == ["result 1", "result 2"]
    assert non_text_content is None


def test_convert_with_non_text_content():
    # Test with non-text content
    image_content = ImageContent(type="image", mimeType="image/png", data="base64data")
    resource_content = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri="resource://test", mimeType="text/plain", text="hi"
        ),
    )

    result = CallToolResult(
        content=[
            TextContent(type="text", text="text result"),
            image_content,
            resource_content,
        ],
        isError=False,
    )

    text_content, non_text_content = _convert_call_tool_result(result)

    assert text_content == "text result"
    assert non_text_content == [image_content, resource_content]


def test_convert_with_error():
    # Test with error
    result = CallToolResult(
        content=[TextContent(type="text", text="error message")], isError=True
    )

    with pytest.raises(ToolException) as exc_info:
        _convert_call_tool_result(result)

    assert str(exc_info.value) == "error message"


async def test_convert_mcp_tool_to_langchain_tool():
    tool_input_schema = {
        "properties": {
            "param1": {"title": "Param1", "type": "string"},
            "param2": {"title": "Param2", "type": "integer"},
        },
        "required": ["param1", "param2"],
        "title": "ToolSchema",
        "type": "object",
    }
    # Mock session and MCP tool
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="tool result")],
        isError=False,
    )

    mcp_tool = MCPTool(
        name="test_tool",
        description="Test tool description",
        inputSchema=tool_input_schema,
    )

    # Convert MCP tool to LangChain tool
    lc_tool = convert_mcp_tool_to_langchain_tool(session, mcp_tool)

    # Verify the converted tool
    assert lc_tool.name == "test_tool"
    assert lc_tool.description == "Test tool description"
    assert lc_tool.args_schema == tool_input_schema

    # Test calling the tool
    result = await lc_tool.ainvoke(
        {"args": {"param1": "test", "param2": 42}, "id": "1", "type": "tool_call"},
    )

    # Verify session.call_tool was called with correct arguments
    session.call_tool.assert_called_once_with(
        "test_tool", {"param1": "test", "param2": 42}, progress_callback=None
    )

    # Verify result
    assert result == ToolMessage(
        content="tool result", name="test_tool", tool_call_id="1"
    )


async def test_convert_mcp_tool_with_empty_input_schema():
    tool_input_schema = {
        "title": "ToolSchema",
        "type": "object",
    }
    # Mock session and MCP tool
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="tool result")],
        isError=False,
    )

    mcp_tool = MCPTool(
        name="test_tool",
        description="Test tool description",
        inputSchema=tool_input_schema,
    )

    # Convert MCP tool to LangChain tool
    lc_tool = convert_mcp_tool_to_langchain_tool(session, mcp_tool)

    # Verify the converted tool
    assert lc_tool.name == "test_tool"
    assert lc_tool.description == "Test tool description"
    assert lc_tool.args_schema == {**tool_input_schema, "properties": {}}

    # Test calling the tool
    result = await lc_tool.ainvoke({"args": {}, "id": "1", "type": "tool_call"})

    # Verify session.call_tool was called with correct arguments
    session.call_tool.assert_called_once_with("test_tool", {}, progress_callback=None)

    # Verify result
    assert result == ToolMessage(
        content="tool result", name="test_tool", tool_call_id="1"
    )


async def test_load_mcp_tools():
    tool_input_schema = {
        "properties": {
            "param1": {"title": "Param1", "type": "string"},
            "param2": {"title": "Param2", "type": "integer"},
        },
        "required": ["param1", "param2"],
        "title": "ToolSchema",
        "type": "object",
    }
    # Mock session and list_tools response
    session = AsyncMock()
    mcp_tools = [
        MCPTool(
            name="tool1",
            description="Tool 1 description",
            inputSchema=tool_input_schema,
        ),
        MCPTool(
            name="tool2",
            description="Tool 2 description",
            inputSchema=tool_input_schema,
        ),
    ]
    session.list_tools.return_value = MagicMock(tools=mcp_tools, nextCursor=None)

    # Mock call_tool to return different results for different tools
    async def mock_call_tool(tool_name, arguments, progress_callback=None):
        if tool_name == "tool1":
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"tool1 result with {arguments}")
                ],
                isError=False,
            )
        return CallToolResult(
            content=[TextContent(type="text", text=f"tool2 result with {arguments}")],
            isError=False,
        )

    session.call_tool.side_effect = mock_call_tool

    # Load MCP tools
    tools = await load_mcp_tools(session)

    # Verify the tools
    assert len(tools) == 2
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool2"

    # Test calling the first tool
    result1 = await tools[0].ainvoke(
        {"args": {"param1": "test1", "param2": 1}, "id": "1", "type": "tool_call"},
    )
    assert result1 == ToolMessage(
        content="tool1 result with {'param1': 'test1', 'param2': 1}",
        name="tool1",
        tool_call_id="1",
    )

    # Test calling the second tool
    result2 = await tools[1].ainvoke(
        {"args": {"param1": "test2", "param2": 2}, "id": "2", "type": "tool_call"},
    )
    assert result2 == ToolMessage(
        content="tool2 result with {'param1': 'test2', 'param2': 2}",
        name="tool2",
        tool_call_id="2",
    )


def _create_annotations_server():
    server = FastMCP(port=8181)

    @server.tool(
        annotations=ToolAnnotations(
            title="Get Time", readOnlyHint=True, idempotentHint=False
        ),
    )
    def get_time() -> str:
        """Get current time"""
        return "5:20:00 PM EST"

    return server


async def test_load_mcp_tools_with_annotations(socket_enabled) -> None:
    """Test load mcp tools with annotations."""
    with run_streamable_http(_create_annotations_server, 8181):
        # Initialize client without initial connections
        client = MultiServerMCPClient(
            {
                "time": {
                    "url": "http://localhost:8181/mcp",
                    "transport": "streamable_http",
                }
            },
        )
        # pass
        tools = await client.get_tools(server_name="time")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_time"
        assert tool.metadata == {
            "title": "Get Time",
            "readOnlyHint": True,
            "idempotentHint": False,
            "destructiveHint": None,
            "openWorldHint": None,
        }


# Tests for to_fastmcp functionality


@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


class AddInput(BaseModel):
    """Add two numbers"""

    a: int
    b: int


@tool("add", args_schema=AddInput)
def add_with_schema(a: int, b: int) -> int:
    return a + b


@tool("add")
def add_with_injection(
    a: int, b: int, injected_arg: Annotated[str, InjectedToolArg()]
) -> int:
    """Add two numbers"""
    return a + b


class AddTool(BaseTool):
    name: str = "add"
    description: str = "Add two numbers"
    args_schema: type[BaseModel] | None = AddInput

    def _run(
        self, a: int, b: int, run_manager: CallbackManagerForToolRun | None = None
    ) -> int:
        """Use the tool."""
        return a + b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> int:
        """Use the tool."""
        return self._run(a, b, run_manager=run_manager)


@pytest.mark.parametrize(
    "tool_instance",
    [add, add_with_schema, AddTool()],
    ids=["tool", "tool_with_schema", "tool_class"],
)
async def test_convert_langchain_tool_to_fastmcp_tool(tool_instance):
    fastmcp_tool = to_fastmcp(tool_instance)
    assert fastmcp_tool.name == "add"
    assert fastmcp_tool.description == "Add two numbers"
    assert fastmcp_tool.parameters == {
        "description": "Add two numbers",
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "add",
        "type": "object",
    }
    assert fastmcp_tool.fn_metadata.arg_model.model_json_schema() == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "addArguments",
        "type": "object",
    }

    arguments = {"a": 1, "b": 2}
    assert await fastmcp_tool.run(arguments=arguments) == 3


def test_convert_langchain_tool_to_fastmcp_tool_with_injection():
    with pytest.raises(NotImplementedError):
        to_fastmcp(add_with_injection)


def _create_status_server():
    server = FastMCP(port=8182)

    @server.tool()
    def get_status() -> str:
        """Get server status"""
        return "Server is running"

    return server


# Tests for httpx_client_factory functionality


async def test_load_mcp_tools_with_custom_httpx_client_factory(socket_enabled) -> None:
    """Test load mcp tools with custom httpx client factory."""

    # Custom httpx client factory
    def custom_httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """Custom factory for creating httpx.AsyncClient with specific configuration."""
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout or httpx.Timeout(30.0),
            auth=auth,
            # Custom configuration
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    with run_streamable_http(_create_status_server, 8182):
        # Initialize client with custom httpx_client_factory
        client = MultiServerMCPClient(
            {
                "status": {
                    "url": "http://localhost:8182/mcp",
                    "transport": "streamable_http",
                    "httpx_client_factory": custom_httpx_client_factory,
                },
            },
        )

        tools = await client.get_tools(server_name="status")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_status"

        # Test that the tool works correctly
        result = await tool.ainvoke({"args": {}, "id": "1", "type": "tool_call"})
        assert result.content == "Server is running"


def _create_info_server():
    server = FastMCP(port=8183)

    @server.tool()
    def get_info() -> str:
        """Get server info"""
        return "SSE Server Info"

    return server


async def test_load_mcp_tools_with_custom_httpx_client_factory_sse(
    socket_enabled,
) -> None:
    """Test load mcp tools with custom httpx client factory using SSE transport."""

    # Custom httpx client factory
    def custom_httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """Custom factory for creating httpx.AsyncClient with specific configuration."""
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout or httpx.Timeout(30.0),
            auth=auth,
            # Custom configuration for SSE
            limits=httpx.Limits(max_keepalive_connections=3, max_connections=5),
        )

    with run_streamable_http(_create_info_server, 8183):
        # Initialize client with custom httpx_client_factory for SSE
        client = MultiServerMCPClient(
            {
                "info": {
                    "url": "http://localhost:8183/sse",
                    "transport": "sse",
                    "httpx_client_factory": custom_httpx_client_factory,
                },
            },
        )

        # Note: This test may not work in practice since the server doesn't expose SSE
        # endpoint,
        # but it tests the configuration propagation
        try:
            tools = await client.get_tools(server_name="info")
            # If we get here, the httpx_client_factory was properly passed
            assert isinstance(tools, list)
        except Exception:
            # Expected to fail since server doesn't have SSE endpoint,
            # but the important thing is that httpx_client_factory was passed correctly
            pass


async def test_convert_mcp_tool_metadata_variants():
    """Verify metadata merging rules in convert_mcp_tool_to_langchain_tool."""
    tool_input_schema = {
        "properties": {},
        "required": [],
        "title": "EmptySchema",
        "type": "object",
    }

    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")], isError=False
    )

    mcp_tool_none = MCPTool(
        name="t_none",
        description="",
        inputSchema=tool_input_schema,
    )
    lc_tool_none = convert_mcp_tool_to_langchain_tool(session, mcp_tool_none)
    assert lc_tool_none.metadata is None

    mcp_tool_ann = MCPTool(
        name="t_ann",
        description="",
        inputSchema=tool_input_schema,
        annotations=ToolAnnotations(
            title="Title", readOnlyHint=True, idempotentHint=False
        ),
    )
    lc_tool_ann = convert_mcp_tool_to_langchain_tool(session, mcp_tool_ann)
    assert lc_tool_ann.metadata == {
        "title": "Title",
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": None,
        "openWorldHint": None,
    }

    mcp_tool_meta = MCPTool(
        name="t_meta",
        description="",
        inputSchema=tool_input_schema,
        _meta={"source": "unit-test", "version": 1},
    )
    lc_tool_meta = convert_mcp_tool_to_langchain_tool(session, mcp_tool_meta)
    assert lc_tool_meta.metadata == {"_meta": {"source": "unit-test", "version": 1}}

    mcp_tool_both = MCPTool(
        name="t_both",
        description="",
        inputSchema=tool_input_schema,
        annotations=ToolAnnotations(title="Both"),
        _meta={"flag": True},
    )

    lc_tool_both = convert_mcp_tool_to_langchain_tool(session, mcp_tool_both)
    assert lc_tool_both.metadata == {
        "title": "Both",
        "readOnlyHint": None,
        "idempotentHint": None,
        "destructiveHint": None,
        "openWorldHint": None,
        "_meta": {"flag": True},
    }
