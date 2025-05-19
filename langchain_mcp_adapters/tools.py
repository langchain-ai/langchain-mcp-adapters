from typing import Any, TypedDict, cast, overload

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


class Annotations(TypedDict, total=False):
    """Annotations for a tool.

    https://modelcontextprotocol.io/docs/concepts/tools#tool-definition-structure
    """

    title: str
    """Human-readable title for the tool"""
    readOnlyHint: bool
    """"If true, the tool does not modify its environment"""
    destructiveHint: bool
    """If true, the tool may perform destructive updates"""
    idempotentHint: bool
    """If true, repeated calls with same args have no additional effect"""
    openWorldHint: bool
    """If true, tool interacts with external entities"""


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
        include_annotations: Whether to include annotations in the tool description

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


@overload
async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    include_annotations: bool = False,
) -> list[BaseTool]: ...


@overload
async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    include_annotations: bool = True,
) -> list[tuple[BaseTool, Annotations]]: ...


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    include_annotations: bool = False,
) -> list[BaseTool] | list[tuple[BaseTool, Annotations]]:
    """Load all available MCP tools and convert them to LangChain tools.

    Args:
        session: MCP client session
        connection: Optional connection config to use to create a new session if a `session` is not provided
        include_annotations: Whether to include annotations in the tool description

    Returns:
        List of LangChain tools if include_annotations is False, otherwise
        a list of tuples containing LangChain tools and their annotations
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

    converted_tools = [
        convert_mcp_tool_to_langchain_tool(session, tool, connection=connection)
        for tool in tools.tools
    ]

    if include_annotations:
        return [(tool, tool.annotations) for tool in converted_tools]
    return converted_tools
