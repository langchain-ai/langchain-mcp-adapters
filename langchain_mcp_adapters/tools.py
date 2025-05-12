from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.tools import tool as create_langchain_tool
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from mcp.types import (
    Tool as MCPTool,
)
from pydantic import BaseModel, create_model

NonTextContent = ImageContent | EmbeddedResource


def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content: str | list[str] = [content.text for content in text_contents]
    if not text_contents:
        tool_content = ""
    elif len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise ToolException(tool_content)

    return tool_content, non_text_contents or None


def convert_mcp_tool_to_langchain_tool(
    session: ClientSession,
    tool: MCPTool,
) -> BaseTool:
    """Convert an MCP tool to a LangChain tool.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert

    Returns:
        a LangChain tool
    """

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        call_tool_result = await session.call_tool(tool.name, arguments)
        return _convert_call_tool_result(call_tool_result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
    )


async def load_mcp_tools(session: ClientSession) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain tools."""
    tools = await session.list_tools()
    return [convert_mcp_tool_to_langchain_tool(session, tool) for tool in tools.tools]


def add_langchain_tool(server: FastMCP, tool: BaseTool | Callable[[Any], Any]) -> None:
    """Add LangChain tool to an MCP server."""
    if not isinstance(tool, BaseTool):
        tool = create_langchain_tool(tool)

    if not issubclass(tool.args_schema, BaseModel):
        raise ValueError(
            "Tool args_schema must be a subclass of pydantic.BaseModel. "
            "Tools with dict args schema are not supported."
        )

    parameters = tool.tool_call_schema.model_json_schema()
    field_definitions = {
        field: (field_info.annotation, field_info)
        for field, field_info in tool.tool_call_schema.model_fields.items()
    }
    arg_model = create_model(
        f"{tool.name}Arguments",
        **field_definitions,
        __base__=ArgModelBase,
    )
    fn_metadata = FuncMetadata(arg_model=arg_model)

    async def fn(**arguments: dict[str, Any]) -> Any:
        return await tool.ainvoke(arguments)

    fastmcp_tool = FastMCPTool(
        fn=fn,
        name=tool.name,
        description=tool.description,
        parameters=parameters,
        fn_metadata=fn_metadata,
        is_async=True,
    )
    if tool.name in server._tool_manager._tools:
        raise ValueError(f"Tool {tool.name} already exists on the server")

    server._tool_manager._tools[tool.name] = fastmcp_tool


def add_langchain_tools(server: FastMCP, tools: list[BaseTool | Callable[[Any], Any]]) -> None:
    """Add multiple LangChain tools to an MCP server."""
    for tool in tools:
        add_langchain_tool(server, tool)
