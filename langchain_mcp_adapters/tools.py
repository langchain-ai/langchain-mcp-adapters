"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from collections.abc import Awaitable, Callable
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
    MCPToolCallRequest,
    MCPToolCallResult,
    ToolCallInterceptor,
)
from langchain_mcp_adapters.sessions import Connection, create_session

NonTextContent: TypeAlias = (
    ImageContent | AudioContent | ResourceLink | EmbeddedResource
)

try:
    from langgraph.runtime import get_runtime  # type: ignore[import-not-found]
except ImportError:

    def get_runtime() -> None:
        """no-op runtime getter."""
        return


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
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], dict[str, Any] | None]:
        """Execute tool call with interceptor chain and return formatted result.

        Args:
            **arguments: Tool arguments as keyword args.

        Returns:
            Tuple of (text_content, artifact).
        """
        mcp_callbacks = (
            callbacks.to_mcp_format(
                context=CallbackContext(
                    server_name=server_name or "unknown", tool_name=tool.name
                )
            )
            if callbacks is not None
            else _MCPCallbacks()
        )

        # try to get runtime if we're in a langgraph context
        try:
            runtime = get_runtime()
        except Exception:  # noqa: BLE001
            runtime = None

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
                updated_connection: dict[str, Any] = dict(connection)  # type: ignore[arg-type]
                if connection["transport"] in ("sse", "streamable_http"):
                    existing_headers_raw = connection.get("headers", {})
                    existing_headers: dict[str, Any] = (
                        existing_headers_raw
                        if isinstance(existing_headers_raw, dict)
                        else {}
                    )
                    headers_dict: dict[str, Any] = modified_headers
                    updated_connection["headers"] = {
                        **existing_headers,
                        **headers_dict,
                    }
                    effective_connection = cast("Connection", updated_connection)

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
        callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name or "unknown")
        )
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

    def _is_injected_arg_type(annotation: Any) -> bool:
        """Check if type annotation contains InjectedToolArg."""
        try:
            args = get_args(annotation)
            if len(args) > 1:
                return any(
                    isinstance(arg, InjectedToolArg)
                    or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
                    for arg in args[1:]
                )
        except (TypeError, AttributeError):
            pass
        return False

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
    if not (
        isinstance(tool.args_schema, type) and issubclass(tool.args_schema, BaseModel)
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
        f"{tool.name}Arguments", __base__=ArgModelBase, **field_definitions
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
        title=tool.name,
        description=tool.description,
        parameters=parameters,
        fn_metadata=fn_metadata,
        is_async=True,
        context_kwarg=None,
        annotations=None,
    )
