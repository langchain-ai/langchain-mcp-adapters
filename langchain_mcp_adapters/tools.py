from mcp.types import (
    Tool as MCPTool,
    CallToolResult,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from mcp import ClientSession
from langchain_core.tools import StructuredTool, ToolException


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

    tool_content = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise ToolException(tool_content)

    return tool_content, non_text_contents or None


def convert_mcp_tool_to_structured_tool(
    session: ClientSession,
    tool: MCPTool,
) -> StructuredTool:
    """Convert an MCP tool to a LangChain StructuredTool.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert

    Returns:
        a LangChain StructuredTool
    """

    async def call_tool(**arguments):
        call_tool_result = await session.call_tool(tool.name, arguments)
        return _convert_call_tool_result(call_tool_result)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
    )


async def load_mcp_tools(session: ClientSession) -> list[StructuredTool]:
    """Load all available MCP tools and convert them to StructuredTools."""
    tools = await session.list_tools()
    return [convert_mcp_tool_to_structured_tool(session, tool) for tool in tools.tools]
