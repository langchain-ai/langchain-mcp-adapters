"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from typing import Any, TypeAlias, cast, get_args

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
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, create_model

from langchain_mcp_adapters.sessions import Connection, create_session

NonTextContent: TypeAlias = ImageContent | EmbeddedResource | TextResourceContents
MAX_ITERATIONS = 1000




def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], dict[str, Any] | None]:
    """Convert MCP CallToolResult to LangChain tool result format.

    The returned tuple follows LangChain's "content_and_artifact" format, where
    the artifact is a dictionary that can include both non-text contents and
    machine-readable structured content from MCP.

    Args:
        call_tool_result: The result from calling an MCP tool.

    Returns:
        A tuple of (content, artifact). "content" is a string or list of strings
        from text content blocks. "artifact" is a dict that may include:
          - "nonText": list of non-text content blocks (e.g., images, resources)
          - "structuredContent": structuredContent (machine-readable) when provided
            by MCP
        If there is no non-text nor structured content, artifact will be None.

    Raises:
        ToolException: If the tool call resulted in an error.
    """
    text_contents, non_text_contents = _separate_content_types(call_tool_result.content)
    tool_content = _format_text_content(text_contents)

    if call_tool_result.isError:
        raise ToolException(tool_content)

    artifact = _build_artifact(non_text_contents, call_tool_result)

    return tool_content, (artifact if artifact else None)


def _separate_content_types(
    content: list,
) -> tuple[list[TextContent], list[NonTextContent]]:
    """Separate content into text and non-text types."""
    text_contents: list[TextContent] = []
    non_text_contents: list[NonTextContent] = []

    for item in content:
        if isinstance(item, TextContent):
            text_contents.append(item)
        else:
            non_text_contents.append(item)

    return text_contents, non_text_contents


def _format_text_content(text_contents: list[TextContent]) -> str | list[str]:
    """Format text content into string or list of strings."""
    tool_content: str | list[str] = [content.text for content in text_contents]
    if not text_contents:
        return ""
    if len(text_contents) == 1:
        return tool_content[0]
    return tool_content


def _build_artifact(
    non_text_contents: list[NonTextContent],
    call_tool_result: CallToolResult,
) -> dict[str, Any]:
    """Build artifact dictionary with non-text and structured content."""
    artifact: dict[str, Any] = {}
    if non_text_contents:
        artifact["nonText"] = non_text_contents

    structured = getattr(call_tool_result, "structuredContent", None)
    if structured is not None:
        artifact["structuredContent"] = structured

    return artifact


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
    ) -> tuple[str | list[str], dict[str, Any] | None]:
        if session is None:
            # If a session is not provided, we will create one on the fly
            if connection is None:
                msg = "Connection must be provided when session is None"
                raise ValueError(msg)
            async with create_session(connection) as tool_session:
                await tool_session.initialize()
                call_tool_result = await cast("ClientSession", tool_session).call_tool(
                    tool.name,
                    arguments,
                )
        else:
            call_tool_result = await session.call_tool(tool.name, arguments)
        return _convert_call_tool_result(call_tool_result)

    meta = tool.meta if hasattr(tool, "meta") else None
    base = tool.annotations.model_dump() if tool.annotations is not None else {}
    meta = {"_meta": meta} if meta is not None else {}
    metadata = {**base, **meta} or None

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=metadata,
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
        if connection is None:
            msg = "Either a session or a connection config must be provided"
            raise ValueError(msg)
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
    if not (
        isinstance(tool.args_schema, type) and
        issubclass(tool.args_schema, BaseModel)
    ):
        msg = (
            "Tool args_schema must be a subclass of pydantic.BaseModel. "
            "Tools with dict args schema are not supported."
        )
        raise TypeError(msg)

    # We already checked that args_schema is a BaseModel subclass
    args_schema = cast("type[BaseModel]", tool.args_schema)
    parameters = args_schema.model_json_schema()
    field_definitions = {
        field: (field_info.annotation, field_info)
        for field, field_info in args_schema.model_fields.items()
    }
    arg_model = create_model(  # type: ignore[call-overload]
        f"{tool.name}Arguments",
        __base__=ArgModelBase,
        **field_definitions
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
        context_kwarg=None,
        annotations=None,
    )
