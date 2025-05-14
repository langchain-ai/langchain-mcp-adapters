from typing import Any, cast

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from mcp.types import (
    Tool as MCPTool,
)

from langchain_mcp_adapters.sessions import Connection, create_session

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
    session: ClientSession | None,
    tool: MCPTool,
    *,
    connection: Connection | None = None,
) -> BaseTool:
    """Convert an MCP tool to a LangChain tool.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert
        connection: Optional connection config to use to create a new session if a `session` is not provided

    Returns:
        a LangChain tool
    """
    if session is None and connection is None:
        raise ValueError("Either a session or a connection config must be provided")

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        if session is None:
            # If a session is not provided, we will create one on the fly
            async with create_session(connection) as tool_session:
                await tool_session.initialize()
                call_tool_result = await cast(ClientSession, tool_session).call_tool(
                    tool.name, arguments
                )
        else:
            call_tool_result = await session.call_tool(tool.name, arguments)
        return _convert_call_tool_result(call_tool_result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
    )


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain tools.

    Args:
        session: MCP client session
        connection: Optional connection config to use to create a new session if a `session` is not provided

    Returns:
        a list of LangChain tools
    """
    if session is None and connection is None:
        raise ValueError("Either a session or a connection config must be provided")

    if session is None:
        # If a session is not provided, we will create one on the fly
        async with create_session(connection) as tool_session:
            await tool_session.initialize()
            tools = await tool_session.list_tools()
    else:
        tools = await session.list_tools()

    return [
        convert_mcp_tool_to_langchain_tool(session, tool, connection=connection)
        for tool in tools.tools
    ]
