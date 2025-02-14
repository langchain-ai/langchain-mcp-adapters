from enum import Enum
from typing import Any

from mcp.types import (
    Tool as MCPTool,
    CallToolResult,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from mcp import ClientSession
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, create_model


NonTextContent = ImageContent | EmbeddedResource


def _create_pydantic_model(
    name: str, json_schema_dict: dict[str, Any], refs: dict[str, dict] | None = None
) -> BaseModel:
    fields = {}
    # Store schema definitions for reference resolution
    refs = refs or json_schema_dict.get("definitions", {})

    for prop_name, prop_schema in json_schema_dict["properties"].items():
        field_type = _get_field_type(prop_name, prop_schema, refs)
        required = prop_name in json_schema_dict.get("required", [])
        default = ... if required else None
        fields[prop_name] = (field_type, default)

    return create_model(name, **fields)


def _get_field_type(
    prop_name: str, prop_schema: dict[str, Any], refs: dict[str, dict]
) -> Any:
    # Handle $ref references
    if "$ref" in prop_schema:
        ref_path = prop_schema["$ref"]
        if ref_path.startswith("#/definitions/"):
            ref_name = ref_path.split("/")[-1]
            if ref_name in refs:
                return _create_pydantic_model(ref_name, refs[ref_name], refs)
        return Any  # Fallback if ref not found

    if "type" not in prop_schema:
        return Any

    if prop_schema["type"] == "array":
        if "items" in prop_schema:
            item_type = _get_field_type(prop_name, prop_schema["items"], refs)
            return list[item_type]
        return list

    if prop_schema["type"] == "object":
        if "properties" in prop_schema:
            return _create_pydantic_model(prop_name, prop_schema, refs)
        return dict

    if prop_schema["type"] == "string" and "enum" in prop_schema:
        return Enum(prop_name, {str(v): v for v in prop_schema["enum"]})

    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }.get(prop_schema["type"], str)


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
        args_schema=_create_pydantic_model(tool.name, tool.inputSchema),
        coroutine=call_tool,
        response_format="content_and_artifact",
    )


async def load_mcp_tools(session: ClientSession) -> list[StructuredTool]:
    """Load all available MCP tools and convert them to StructuredTools."""
    tools = await session.list_tools()
    return [convert_mcp_tool_to_structured_tool(session, tool) for tool in tools.tools]
