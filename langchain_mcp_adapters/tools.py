from typing import Any, cast, get_args, get_type_hints

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool, ToolException
from langchain_core.tools.base import get_all_basemodel_annotations
from mcp import ClientSession
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

from langchain_mcp_adapters.sessions import Connection, create_session

FAST_MCP_CONTEXT_KWARG = "__context"

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
        connection: Optional connection config to use to create a new session
                    if a `session` is not provided

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
        metadata=tool.annotations.model_dump() if tool.annotations else None,
    )


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain tools.

    Returns:
        list of LangChain tools. Tool annotations are returned as part
        of the tool metadata object.
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
    return converted_tools


def _get_injected_args(tool: BaseTool) -> list[str]:
    def _is_injected_arg_type(type_: type) -> bool:
        return any(
            isinstance(arg, InjectedToolArg)
            or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
            for arg in get_args(type_)[1:]
        )

    def _get_runnable_config_param() -> str | None:
        type_hints = get_type_hints(tool._run)
        if not type_hints:
            return None
        for name, type_ in type_hints.items():
            if type_ is RunnableConfig:
                return name
        return None

    injected_args = [
        field
        for field, field_info in get_all_basemodel_annotations(tool.args_schema).items()
        if _is_injected_arg_type(field_info)
    ]
    config_arg = _get_runnable_config_param()
    if config_arg:
        injected_args.append(config_arg)

    return injected_args


def convert_langchain_tool_to_fastmcp_tool(tool: BaseTool) -> FastMCPTool:
    """Add LangChain tool to an MCP server."""

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
        context = arguments.pop(FAST_MCP_CONTEXT_KWARG, {})
        combined_arguments = {**arguments, **context}
        return await tool.ainvoke(combined_arguments)

    injected_args = _get_injected_args(tool)
    has_injected_args = len(injected_args) > 0
    fastmcp_tool = FastMCPTool(
        fn=fn,
        name=tool.name,
        description=tool.description,
        parameters=parameters,
        fn_metadata=fn_metadata,
        is_async=True,
        context_kwarg=FAST_MCP_CONTEXT_KWARG if has_injected_args else None,
    )
    return fastmcp_tool
