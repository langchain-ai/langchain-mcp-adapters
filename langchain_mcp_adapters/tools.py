"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from typing import Annotated, Any, Literal, Union, cast, get_args

from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
)
from langchain_core.tools.base import get_all_basemodel_annotations
from mcp import ClientSession
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, create_model

from langchain_mcp_adapters.sessions import Connection, create_session

NonTextContent = ImageContent | EmbeddedResource
MAX_ITERATIONS = 1000


def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    """Convert MCP CallToolResult to LangChain tool result format.

    Args:
        call_tool_result: The result from calling an MCP tool.

    Returns:
        A tuple containing the text content and any non-text content.

    Raises:
        ToolException: If the tool call resulted in an error.
    """
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


async def _list_all_tools(session: ClientSession) -> list[MCPTool]:
    """List all available tools from an MCP session with pagination support.

    Args:
        session: The MCP client session.

    Returns:
        A list of all available MCP tools.

    Raises:
        RuntimeError: If maximum iterations exceeded while listing tools.
    """
    current_cursor: str | None = None
    all_tools: list[MCPTool] = []

    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            msg = "Reached max of 1000 iterations while listing tools."
            raise RuntimeError(msg)

        list_tools_page_result = await session.list_tools(cursor=current_cursor)

        if list_tools_page_result.tools:
            all_tools.extend(list_tools_page_result.tools)

        # Pagination spec: https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination
        # compatible with None or ""
        if not list_tools_page_result.nextCursor:
            break

        current_cursor = list_tools_page_result.nextCursor
    return all_tools


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
        connection: Optional connection config to use to create a new session
                    if a `session` is not provided

    Returns:
        a LangChain tool

    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        if session is None:
            # If a session is not provided, we will create one on the fly
            async with create_session(connection) as tool_session:
                await tool_session.initialize()
                call_tool_result = await cast("ClientSession", tool_session).call_tool(
                    tool.name,
                    arguments,
                )
        else:
            call_tool_result = await session.call_tool(tool.name, arguments)
        return _convert_call_tool_result(call_tool_result)

    # base types being mapped from JSON
    type_map = {
        "null": None,
        "integer": int,
        "float": float,
        "string": str,
        "bool": bool,
        "object": dict,
        "bytes": bytes,
    }

    def _parse_model_fields(args: dict, injected_state: str | None = None) -> dict:
        """Parse a JSON field into a Pydantic Field, taking into account injected state.

        :param args: the function parameter schema
        :type args: dict
        :param injected_state: the name of the key used for the InjectedState
        :type injected_state: str
        :return: returns a dict of fields with their pydantic type
        and default value if any
        :rtype: dict
        """
        model_fields = {}

        def _parse_field(props: dict) -> tuple[type, Any]:
            if "anyOf" in props:
                return Union[tuple(_parse_field(p) for p in props["anyOf"])]  # noqa: UP007
            if "enum" in props:
                return Literal[tuple(props["enum"])]
            if props["type"] == "array":
                items = props["items"]
                return list[_parse_field(items)]
            return type_map.get(props["type"], dict)

        for field, props in args["properties"].items():
            if field == injected_state:
                field_type = Annotated[dict, InjectedState]
            else:
                field_type = _parse_field(props)
            if "default" in props:
                default = props["default"]
                model_fields[field] = (field_type, default)
            else:
                model_fields[field] = (field_type, ...)
        return model_fields

    args = tool.inputSchema
    # check for the `injected_state`` annotation on the MCP tool.
    # The injected_state value is the name of the function parameter used
    # as the injected state
    injected_state = tool.annotations.model_extra.get("injected_state")
    if injected_state:
        # import langgraph InjectedState only if we need it
        from langgraph.prebuilt import InjectedState
    model_fields = _parse_model_fields(args, injected_state)

    # recreate a dynamic model based on the parsed JSON schema,
    # which will be properly parsed with annotations for the InjectedState
    args_schema = create_model(tool.name, **model_fields)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=args_schema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=tool.annotations.model_dump() if tool.annotations else None,
    )


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain tools.

    Args:
        session: The MCP client session. If None, connection must be provided.
        connection: Connection config to create a new session if session is None.

    Returns:
        List of LangChain tools. Tool annotations are returned as part
        of the tool metadata object.

    Raises:
        ValueError: If neither session nor connection is provided.
    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    if session is None:
        # If a session is not provided, we will create one on the fly
        async with create_session(connection) as tool_session:
            await tool_session.initialize()
            tools = await _list_all_tools(tool_session)
    else:
        tools = await _list_all_tools(session)

    return [
        convert_mcp_tool_to_langchain_tool(session, tool, connection=connection)
        for tool in tools
    ]


def _get_injected_args(tool: BaseTool) -> list[str]:
    """Get the list of injected argument names from a LangChain tool.

    Args:
        tool: The LangChain tool to inspect.

    Returns:
        A list of injected argument names.
    """

    def _is_injected_arg_type(type_: type) -> bool:
        return any(
            isinstance(arg, InjectedToolArg)
            or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
            for arg in get_args(type_)[1:]
        )

    return [
        field
        for field, field_info in get_all_basemodel_annotations(tool.args_schema).items()
        if _is_injected_arg_type(field_info)
    ]


def to_fastmcp(tool: BaseTool) -> FastMCPTool:
    """Convert a LangChain tool to a FastMCP tool.

    Args:
        tool: The LangChain tool to convert.

    Returns:
        A FastMCP tool equivalent of the LangChain tool.

    Raises:
        TypeError: If the tool's args_schema is not a BaseModel subclass.
        NotImplementedError: If the tool has injected arguments.
    """
    if not issubclass(tool.args_schema, BaseModel):
        msg = (
            "Tool args_schema must be a subclass of pydantic.BaseModel. "
            "Tools with dict args schema are not supported."
        )
        raise TypeError(msg)

    parameters = tool.tool_call_schema.model_json_schema()
    field_definitions = {
        field: (field_info.annotation, field_info)
        for field, field_info in tool.tool_call_schema.model_fields.items()
    }
    arg_model = create_model(
        f"{tool.name}Arguments", **field_definitions, __base__=ArgModelBase
    )
    fn_metadata = FuncMetadata(arg_model=arg_model)

    # We'll use an Any type for the function return type.
    # We're providing the parameters separately
    async def fn(**arguments: dict[str, Any]) -> Any:  # noqa: ANN401
        return await tool.ainvoke(arguments)

    injected_args = _get_injected_args(tool)
    if len(injected_args) > 0:
        msg = "LangChain tools with injected arguments are not supported"
        raise NotImplementedError(msg)

    return FastMCPTool(
        fn=fn,
        name=tool.name,
        description=tool.description,
        parameters=parameters,
        fn_metadata=fn_metadata,
        is_async=True,
    )
