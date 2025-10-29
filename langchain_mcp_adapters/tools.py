"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from collections.abc import Awaitable, Callable
from typing import Any, cast, get_args

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
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, create_model

from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks, _MCPCallbacks
from langchain_mcp_adapters.interceptors import (
    Interceptors,
    ToolCallInterceptor,
    ToolCallRequest,
    ToolInterceptorContext,
)
from langchain_mcp_adapters.sessions import Connection, create_session

try:
    from langgraph.config import get_config
    from langgraph.runtime import get_runtime
except ImportError:

    def get_config() -> dict:
        """no-op config getter."""
        return {}

    def get_runtime() -> None:
        """no-op runtime getter."""
        return


NonTextContent = ImageContent | AudioContent | ResourceLink | EmbeddedResource
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


def _build_interceptor_chain(
    base_handler: Callable[[ToolCallRequest], Awaitable[CallToolResult]],
    interceptors: Interceptors | None,
    context: ToolInterceptorContext,
) -> Callable[[ToolCallRequest], Awaitable[CallToolResult]]:
    """Build composed handler chain with interceptors in onion pattern.

    Args:
        base_handler: Innermost handler executing the actual tool call.
        interceptors: Optional interceptors to wrap the handler.
        context: Context passed to each interceptor.

    Returns:
        Composed handler with all interceptors applied. First interceptor
        in list becomes outermost layer.
    """
    handler = base_handler

    if interceptors and interceptors.tools:
        for interceptor in reversed(interceptors.tools):
            current_handler = handler

            async def wrapped_handler(
                req: ToolCallRequest,
                _interceptor: ToolCallInterceptor = interceptor,
                _handler: Callable[[ToolCallRequest], Awaitable[CallToolResult]] = current_handler,
            ) -> CallToolResult:
                return await _interceptor(req, context, _handler)

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
    interceptors: Interceptors | None = None,
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
        interceptors: Optional interceptors for tool call processing
        server_name: Name of the server this tool belongs to

    Returns:
        a LangChain tool

    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        """Execute tool call with interceptor chain and return formatted result.

        Args:
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

        # try to get config and runtime if we're in a langgraph context
        try:
            config = get_config()
            runtime = get_runtime()
        except Exception:  # noqa: BLE001
            config = {}
            runtime = None

        interceptor_context = ToolInterceptorContext(
            server_name=server_name or "unknown",
            tool_name=tool.name,
            config=config,
            runtime=runtime,
        )

        # Create the innermost handler that actually executes the tool call
        async def execute_tool(request: ToolCallRequest) -> CallToolResult:
            """Execute the actual MCP tool call with optional session creation.

            Args:
                request: Tool call request with name, args, and headers.

            Returns:
                CallToolResult from MCP SDK.

            Raises:
                ValueError: If neither session nor connection provided.
                RuntimeError: If tool call returns None.
            """
            tool_name = request.get("name", tool.name)
            tool_args = request.get("args", {})
            effective_connection = connection

            # If headers were modified, create a new connection with updated headers
            modified_headers = request.get("headers")
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

            # Execute the tool call
            call_tool_result = None

            if session is None:
                # If a session is not provided, we will create one on the fly
                if effective_connection is None:
                    msg = "Either session or connection must be provided"
                    raise ValueError(msg)

                async with create_session(
                    effective_connection, mcp_callbacks=mcp_callbacks
                ) as tool_session:
                    await tool_session.initialize()
                    call_tool_result = await cast(
                        "ClientSession", tool_session
                    ).call_tool(
                        tool_name,
                        tool_args,
                        progress_callback=mcp_callbacks.progress_callback,
                    )
            else:
                call_tool_result = await session.call_tool(
                    tool_name,
                    tool_args,
                    progress_callback=mcp_callbacks.progress_callback,
                )

            if call_tool_result is None:
                msg = (
                    "Tool call failed: no result returned from the underlying MCP SDK. "
                    "This may indicate that an exception was handled or suppressed "
                    "by the MCP SDK (e.g., client disconnection, network issue, "
                    "or other execution error)."
                )
                raise RuntimeError(msg)

            return call_tool_result

        # Build and execute the interceptor chain
        handler = _build_interceptor_chain(
            execute_tool, interceptors, interceptor_context
        )
        request = ToolCallRequest(name=tool.name, args=arguments)
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
    interceptors: Interceptors | None = None,
    server_name: str | None = None,
) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain tools.

    Args:
        session: The MCP client session. If None, connection must be provided.
        connection: Connection config to create a new session if session is None.
        callbacks: Optional callbacks for handling notifications and events.
        interceptors: Optional interceptors for tool call processing.
        server_name: Name of the server these tools belong to.

    Returns:
        List of LangChain tools. Tool annotations are returned as part
        of the tool metadata object.

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
            interceptors=interceptors,
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
