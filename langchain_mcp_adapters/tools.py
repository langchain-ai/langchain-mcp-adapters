from typing import Any

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
from pydantic.networks import AnyUrl

NonTextContent = ImageContent | EmbeddedResource

def _convert_mcp_artifact_types_to_langchain(non_text_content):
    if isinstance(non_text_content, ImageContent):
        return {
            "type": "image_url",
            "image_url": {"url": non_text_content.data},
        }
    elif isinstance(non_text_content, EmbeddedResource):
        if hasattr(non_text_content, 'resource'):
            artifact_resource = non_text_content.resource
            if hasattr(artifact_resource, 'blob'):
                artifact_name = artifact_resource.blob
            else:
                raise Exception("No blob found in the artifact")
            if hasattr(artifact_resource, 'uri'):
                artifact_uri = artifact_resource.uri
                if isinstance(artifact_uri, AnyUrl):
                    if artifact_uri.scheme:
                        file_data = artifact_uri.scheme + ":" + artifact_uri.path
                    else:
                        file_data = artifact_uri.path
                else:
                    raise Exception("No uri found in the artifact")
            else:
                raise Exception("No uri found in the artifact")
            return {
                "type": "file",
                "file": {"filename": artifact_name, "file_data": file_data},
            }
    else:
        raise NotImplementedError("Artifact type not supported")
    
def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(_convert_mcp_artifact_types_to_langchain(content))

    tool_content: str | list[str] = [content.text for content in text_contents]
    if len(text_contents) == 1:
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
