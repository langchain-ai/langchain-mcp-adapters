"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from collections.abc import Awaitable, Callable
from typing import Any, get_args

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
    AudioContent,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, create_model

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks, _MCPCallbacks
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    MCPToolCallResult,
    ToolCallInterceptor,
)
from langchain_mcp_adapters.sessions import Connection, create_session

NonTextContent = ImageContent | AudioContent | ResourceLink | EmbeddedResource
MAX_ITERATIONS = 1000


def _convert_call_tool_result(
    call_tool_result: MCPToolCallResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    """Convert MCP MCPToolCallResult to LangChain tool result format.

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


def _build_interceptor_chain(
    base_handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    tool_interceptors: list[ToolCallInterceptor] | None,
) -> Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]:
    """Build composed handler chain with interceptors in onion pattern.

    Args:
        base_handler: Innermost handler executing the actual tool call.
        tool_interceptors: Optional list of interceptors to wrap the handler.

    Returns:
        Composed handler with all interceptors applied. First interceptor
        in list becomes outermost layer.
    """
    handler = base_handler

    if tool_interceptors:
        for interceptor in reversed(tool_interceptors):
            current_handler = handler

            async def wrapped_handler(
                req: MCPToolCallRequest,
                _interceptor: ToolCallInterceptor = interceptor,
                _handler: Callable[
                    [MCPToolCallRequest], Awaitable[MCPToolCallResult]
                ] = current_handler,
            ) -> MCPToolCallResult:
                return await _interceptor(req, _handler)

            handler = wrapped_handler

    return handler


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
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    server_name: str | None = None,
) -> BaseTool:
    """Convert an MCP tool to a LangChain tool.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert
        connection: Optional connection config to use to create a new session
                    if a `session` is not provided
        callbacks: Optional callbacks for handling notifications and events
        tool_interceptors: Optional list of interceptors for tool call processing
        server_name: Name of the server this tool belongs to

    Returns:
        a LangChain tool

    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    async def call_tool(
        runtime: Any = None,
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        """Execute tool call with interceptor chain and return formatted result.

        Args:
            runtime: LangGraph tool runtime if available, otherwise None.
            **arguments: Tool arguments as keyword args.

        Returns:
            Tuple of (text_content, non_text_content).
        """
        mcp_callbacks = (
            callbacks.to_mcp_format(
                context=CallbackContext(server_name=server_name, tool_name=tool.name)
            )
            if callbacks is not None
            else _MCPCallbacks()
        )

        # Create the innermost handler that actually executes the tool call
        async def execute_tool(request: MCPToolCallRequest) -> MCPToolCallResult:
            """Execute the actual MCP tool call with optional session creation.

            Args:
                request: Tool call request with name, args, headers, and context.

            Returns:
                MCPToolCallResult from MCP SDK.

            Raises:
                ValueError: If neither session nor connection provided.
                RuntimeError: If tool call returns None.
            """
            tool_name = request.name
            tool_args = request.args
            effective_connection = connection

            # If headers were modified, create a new connection with updated headers
            modified_headers = request.headers
            if modified_headers is not None and connection is not None:
                # Create a new connection config with updated headers
                updated_connection = dict(connection)
                if connection["transport"] in ("sse", "streamable_http"):
                    existing_headers = connection.get("headers", {})
                    updated_connection["headers"] = {
                        **existing_headers,
                        **modified_headers,
                    }
                    effective_connection = updated_connection

            captured_exception = None

            if session is None:
                # If a session is not provided, we will create one on the fly
                if effective_connection is None:
                    msg = "Either session or connection must be provided"
                    raise ValueError(msg)

                async with create_session(
                    effective_connection, mcp_callbacks=mcp_callbacks
                ) as tool_session:
                    await tool_session.initialize()
                    try:
                        call_tool_result = await tool_session.call_tool(
                            tool_name,
                            tool_args,
                            progress_callback=mcp_callbacks.progress_callback,
                        )
                    except Exception as e:  # noqa: BLE001
                        # Capture exception to re-raise outside context manager
                        captured_exception = e

                # Re-raise the exception outside the context manager
                # This is necessary because the context manager may suppress exceptions
                # This change was introduced to work-around an issue in MCP SDK
                # that may suppress exceptions when the client disconnects.
                # If this is causing an issue, with your use case, please file an issue
                # on the langchain-mcp-adapters GitHub repo.
                if captured_exception is not None:
                    raise captured_exception
            else:
                call_tool_result = await session.call_tool(
                    tool_name,
                    tool_args,
                    progress_callback=mcp_callbacks.progress_callback,
                )

            return call_tool_result

        # Build and execute the interceptor chain
        handler = _build_interceptor_chain(execute_tool, tool_interceptors)
        request = MCPToolCallRequest(
            name=tool.name,
            args=arguments,
            server_name=server_name or "unknown",
            headers=None,
            runtime=runtime,
        )
        call_tool_result = await handler(request)

        return _convert_call_tool_result(call_tool_result)

    meta = getattr(tool, "meta", None)
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
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    server_name: str | None = None,
) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools).

    Args:
        session: The MCP client session. If `None`, connection must be provided.
        connection: Connection config to create a new session if session is `None`.
        callbacks: Optional `Callbacks` for handling notifications and events.
        tool_interceptors: Optional list of interceptors for tool call processing.
        server_name: Name of the server these tools belong to.

    Returns:
        List of LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools).
            Tool annotations are returned as part of the tool metadata object.

    Raises:
        ValueError: If neither session nor connection is provided.
    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    mcp_callbacks = (
        callbacks.to_mcp_format(context=CallbackContext(server_name=server_name))
        if callbacks is not None
        else _MCPCallbacks()
    )

    if session is None:
        # If a session is not provided, we will create one on the fly
        if connection is None:
            msg = "Either session or connection must be provided"
            raise ValueError(msg)
        async with create_session(
            connection, mcp_callbacks=mcp_callbacks
        ) as tool_session:
            await tool_session.initialize()
            tools = await _list_all_tools(tool_session)
    else:
        tools = await _list_all_tools(session)

    return [
        convert_mcp_tool_to_langchain_tool(
            session,
            tool,
            connection=connection,
            callbacks=callbacks,
            tool_interceptors=tool_interceptors,
            server_name=server_name,
        )
        for tool in tools
    ]


def _get_injected_args(tool: BaseTool) -> list[str]:
    """Extract field names with InjectedToolArg annotation from tool schema.

    Args:
        tool: LangChain tool to inspect.

    Returns:
        List of field names marked as injected arguments.
    """

    def _is_injected_arg_type(type_: type) -> bool:
        """Check if type annotation contains InjectedToolArg."""
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
    """Convert LangChain tool to FastMCP tool.

    Args:
        tool: LangChain tool to convert.

    Returns:
        FastMCP tool equivalent.

    Raises:
        TypeError: If args_schema is not BaseModel subclass.
        NotImplementedError: If tool has injected arguments.
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
