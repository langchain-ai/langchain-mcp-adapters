"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict, get_args

from langchain_core.messages import ToolMessage
from langchain_core.messages.content import (
    FileContentBlock,
    ImageContentBlock,
    TextContentBlock,
    create_file_block,
    create_image_block,
    create_text_block,
)
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
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
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

try:
    # langgraph installed
    from langgraph.types import Command, interrupt

    LANGGRAPH_PRESENT = True
except ImportError:
    LANGGRAPH_PRESENT = False
    interrupt = None  # type: ignore[assignment]

# Type alias for LangChain content blocks used in ToolMessage
ToolMessageContentBlock = TextContentBlock | ImageContentBlock | FileContentBlock

# Conditional type based on langgraph availability
if LANGGRAPH_PRESENT:
    ConvertedToolResult = list[ToolMessageContentBlock] | ToolMessage | Command
else:
    ConvertedToolResult = list[ToolMessageContentBlock] | ToolMessage

MAX_ITERATIONS = 1000


class MCPToolArtifact(TypedDict):
    """Artifact returned from MCP tool calls.

    This TypedDict wraps the structured content from MCP tool calls,
    allowing for future extension if MCP adds more fields to tool results.

    Attributes:
        structured_content: The structured content returned by the MCP tool,
            corresponding to the structuredContent field in CallToolResult.
    """

    structured_content: dict[str, Any]


def _convert_mcp_content_to_lc_block(  # noqa: PLR0911
    content: ContentBlock,
) -> ToolMessageContentBlock:
    """Convert any MCP content block to a LangChain content block.

    Args:
        content: MCP content object (TextContent, ImageContent, AudioContent,
            ResourceLink, or EmbeddedResource).

    Returns:
        LangChain content block dict.

    Raises:
        NotImplementedError: If AudioContent is passed.
        ValueError: If an unknown content type is passed.
    """
    if isinstance(content, TextContent):
        return create_text_block(text=content.text)

    if isinstance(content, ImageContent):
        return create_image_block(base64=content.data, mime_type=content.mimeType)

    if isinstance(content, AudioContent):
        msg = (
            "AudioContent conversion to LangChain content blocks is not yet "
            f"supported. Received audio with mime type: {content.mimeType}"
        )
        raise NotImplementedError(msg)

    if isinstance(content, ResourceLink):
        mime_type = content.mimeType or None
        if mime_type and mime_type.startswith("image/"):
            return create_image_block(url=str(content.uri), mime_type=mime_type)
        return create_file_block(url=str(content.uri), mime_type=mime_type)

    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return create_text_block(text=resource.text)
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mimeType or None
            if mime_type and mime_type.startswith("image/"):
                return create_image_block(base64=resource.blob, mime_type=mime_type)
            return create_file_block(base64=resource.blob, mime_type=mime_type)
        msg = f"Unknown embedded resource type: {type(resource).__name__}"
        raise ValueError(msg)

    msg = f"Unknown MCP content type: {type(content).__name__}"
    raise ValueError(msg)


def _convert_call_tool_result(
    call_tool_result: MCPToolCallResult,
) -> tuple[ConvertedToolResult, MCPToolArtifact | None]:
    """Convert MCP MCPToolCallResult to LangChain tool result format.

    Converts MCP content blocks to LangChain content blocks:
    - TextContent -> {"type": "text", "text": ...}
    - ImageContent -> {"type": "image", "base64": ..., "mime_type": ...}
    - ResourceLink (image/*) -> {"type": "image", "url": ..., "mime_type": ...}
    - ResourceLink (other) -> {"type": "file", "url": ..., "mime_type": ...}
    - EmbeddedResource (text) -> {"type": "text", "text": ...}
    - EmbeddedResource (blob) -> {"type": "image", ...} or {"type": "file", ...}
    - AudioContent -> raises NotImplementedError

    Args:
        call_tool_result: The result from calling an MCP tool. Can be either
            a CallToolResult (MCP format), a ToolMessage (LangChain format),
            or a Command (LangGraph format, if langgraph is installed).

    Returns:
        A tuple containing:
        - The content: either a string (single text), list of content blocks,
          ToolMessage, or Command
        - The artifact: MCPToolArtifact with structured_content if present,
          otherwise None

    Raises:
        ToolException: If the tool call resulted in an error.
        NotImplementedError: If AudioContent is encountered.
    """
    # If the interceptor returned a ToolMessage directly, return it as the content
    # with None as the artifact to match the content_and_artifact format
    if isinstance(call_tool_result, ToolMessage):
        return call_tool_result, None

    # If the interceptor returned a Command (LangGraph), return it directly
    if LANGGRAPH_PRESENT and isinstance(call_tool_result, Command):
        return call_tool_result, None

    # Convert all MCP content blocks to LangChain content blocks
    tool_content: list[ToolMessageContentBlock] = [
        _convert_mcp_content_to_lc_block(content)
        for content in call_tool_result.content
    ]

    if call_tool_result.isError:
        # Join text from all blocks
        error_parts = []
        for item in tool_content:
            if isinstance(item, str):
                error_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                error_parts.append(item.get("text", ""))
        error_msg = "\n".join(error_parts) if error_parts else str(tool_content)
        raise ToolException(error_msg)

    # Extract structured content and wrap in MCPToolArtifact
    artifact: MCPToolArtifact | None = None
    if call_tool_result.structuredContent is not None:
        artifact = MCPToolArtifact(
            structured_content=call_tool_result.structuredContent
        )

    return tool_content, artifact


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


class _ElicitationCoordinator:
    """Coordinates elicitation between MCP callbacks and LangGraph interrupts.

    The fundamental constraint is that MCP elicitation callbacks must return
    a valid response - they cannot raise exceptions. This means we need async
    coordination between:
    1. The callback (running inside MCP SDK) - must block until response ready
    2. The tool execution (where LangGraph context exists) - must call interrupt()

    Flow:
    1. Tool execution starts, callback is registered
    2. MCP server calls elicitation callback
    3. Callback stores request and signals request_ready
    4. Tool execution sees request, calls interrupt() - graph pauses
    5. User resumes with response
    6. Tool execution stores response and signals response_ready
    7. Callback wakes up and returns the response to MCP SDK
    8. Tool execution completes
    """

    def __init__(self) -> None:
        """Initialize the coordinator with fresh events."""
        import asyncio  # noqa: PLC0415

        self.pending_request: Any = None
        self.response: Any = None
        self.request_ready = asyncio.Event()
        self.response_ready = asyncio.Event()

    def create_callback(self) -> Callbacks:
        """Create an elicitation callback for MCP.

        Returns:
            Callbacks instance with the elicitation handler configured.
        """
        from mcp.shared.context import RequestContext as MCPRequestContext  # noqa: PLC0415
        from mcp.types import ElicitRequestParams, ElicitResult  # noqa: PLC0415

        from langchain_mcp_adapters.elicitation import (  # noqa: PLC0415
            ElicitationRequest,
        )

        async def elicitation_handler(
            mcp_ctx: MCPRequestContext,  # noqa: ARG001
            params: ElicitRequestParams,
            callback_context: CallbackContext,
        ) -> ElicitResult:
            # Store the request and signal that we need user input
            self.pending_request = ElicitationRequest(
                message=params.message,
                requested_schema=params.requestedSchema,
                server_name=callback_context.server_name,
                tool_name=callback_context.tool_name or "unknown",
            )
            self.request_ready.set()

            # Wait for the response (tool execution will provide it after interrupt)
            await self.response_ready.wait()

            # Return the response to MCP SDK
            return self.response

        return Callbacks(on_elicitation=elicitation_handler)


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

    When LangGraph is installed, elicitation is automatically supported. If an MCP
    server requests user input via elicitation during tool execution, a LangGraph
    interrupt will be triggered. The graph execution will pause, allowing the user
    to provide a response via `Command(resume=ElicitationResponse(...))`.

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
        runtime: Any = None,  # noqa: ANN401
        **arguments: dict[str, Any],
    ) -> tuple[ConvertedToolResult, MCPToolArtifact | None]:
        """Execute tool call with interceptor chain and return formatted result.

        Args:
            runtime: LangGraph tool runtime if available, otherwise None.
            **arguments: Tool arguments as keyword args.

        Returns:
            A tuple of (content, artifact) where:
            - content: string, list of strings/content blocks, ToolMessage, or Command
            - artifact: MCPToolArtifact with structured_content if present, else None
        """
        from mcp.types import ElicitResult  # noqa: PLC0415

        from langchain_mcp_adapters.elicitation import (  # noqa: PLC0415
            ElicitationResponse,
        )

        context = CallbackContext(server_name=server_name or "unknown", tool_name=tool.name)

        # Determine if we should handle elicitation (LangGraph present, no user callback)
        use_elicitation = (
            LANGGRAPH_PRESENT
            and (callbacks is None or callbacks.on_elicitation is None)
        )

        async def execute_with_callbacks(
            mcp_callbacks: _MCPCallbacks,
            request: MCPToolCallRequest,
        ) -> MCPToolCallResult:
            """Execute the tool call with the given callbacks."""
            tool_name = request.name
            tool_args = request.args
            effective_connection = connection

            # If headers were modified, create a new connection with updated headers
            modified_headers = request.headers
            if modified_headers is not None and connection is not None:
                updated_connection = dict(connection)
                if connection["transport"] in (
                    "sse",
                    "http",
                    "streamable_http",
                    "streamable-http",
                ):
                    existing_headers = connection.get("headers", {})
                    updated_connection["headers"] = {
                        **existing_headers,
                        **modified_headers,
                    }
                    effective_connection = updated_connection

            captured_exception = None

            if session is None:
                if effective_connection is None:
                    msg = "Either session or connection must be provided"
                    raise ValueError(msg)

                call_tool_result: MCPToolCallResult | None = None
                async with create_session(
                    effective_connection,
                    mcp_callbacks=mcp_callbacks,
                ) as tool_session:
                    await tool_session.initialize()
                    try:
                        call_tool_result = await tool_session.call_tool(
                            tool_name,
                            tool_args,
                            progress_callback=mcp_callbacks.progress_callback,
                        )
                    except Exception as e:  # noqa: BLE001
                        captured_exception = e

                # Re-raise outside context manager (MCP SDK workaround)
                if captured_exception is not None:
                    raise captured_exception

                # This should never happen if no exception was raised
                assert call_tool_result is not None
                return call_tool_result
            else:
                # Persistent sessions don't support elicitation callbacks
                return await session.call_tool(
                    tool_name,
                    tool_args,
                    progress_callback=mcp_callbacks.progress_callback,
                )

        request = MCPToolCallRequest(
            name=tool.name,
            args=arguments,
            server_name=server_name or "unknown",
            headers=None,
            runtime=runtime,
        )

        if use_elicitation:
            import asyncio  # noqa: PLC0415

            coordinator = _ElicitationCoordinator()
            elicitation_callbacks = coordinator.create_callback()

            # Merge with user callbacks (preserving logging/progress)
            if callbacks is not None:
                effective_callbacks = Callbacks(
                    on_logging_message=callbacks.on_logging_message,
                    on_progress=callbacks.on_progress,
                    on_elicitation=elicitation_callbacks.on_elicitation,
                )
            else:
                effective_callbacks = elicitation_callbacks

            mcp_callbacks = effective_callbacks.to_mcp_format(context=context)

            async def execute_tool(req: MCPToolCallRequest) -> MCPToolCallResult:
                return await execute_with_callbacks(mcp_callbacks, req)

            handler = _build_interceptor_chain(execute_tool, tool_interceptors)

            # Run tool execution, watching for elicitation requests
            tool_task = asyncio.create_task(handler(request))
            request_task = asyncio.create_task(coordinator.request_ready.wait())

            done, pending = await asyncio.wait(
                [tool_task, request_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if request_task in done and tool_task in pending:
                # Elicitation was requested - call interrupt to pause graph
                # The callback is now waiting on response_ready
                #
                # NOTE: interrupt() will either:
                # 1. Raise GraphInterrupt (first execution) - graph pauses
                # 2. Return the resume value (re-execution after resume)
                #
                # On first execution, we need to cancel the pending tool_task
                # before interrupt raises, otherwise it becomes orphaned.
                try:
                    response = interrupt(coordinator.pending_request)
                except BaseException:
                    # interrupt() raised (first execution) - cancel orphaned task
                    tool_task.cancel()
                    # Signal the callback to unblock (it will be cancelled anyway)
                    coordinator.response = ElicitResult(action="cancel")
                    coordinator.response_ready.set()
                    try:
                        await tool_task
                    except asyncio.CancelledError:
                        pass
                    raise

                # If we get here, interrupt() returned the resume value (re-execution)
                # Convert response to ElicitResult
                if isinstance(response, ElicitationResponse):
                    elicit_result = response.to_result()
                elif isinstance(response, ElicitResult):
                    elicit_result = response
                elif isinstance(response, dict):
                    elicit_result = ElicitResult(
                        action=response.get("action", "cancel"),
                        content=response.get("content"),
                    )
                else:
                    elicit_result = ElicitResult(action="cancel")

                # Provide response to the waiting callback
                coordinator.response = elicit_result
                coordinator.response_ready.set()

                # Now wait for tool to complete
                call_tool_result = await tool_task
            else:
                # Tool completed without elicitation
                request_task.cancel()
                try:
                    await request_task
                except asyncio.CancelledError:
                    pass
                call_tool_result = tool_task.result()
        else:
            # No elicitation handling - just execute with user callbacks
            mcp_callbacks = (
                callbacks.to_mcp_format(context=context)
                if callbacks is not None
                else _MCPCallbacks()
            )

            async def execute_tool(req: MCPToolCallRequest) -> MCPToolCallResult:
                return await execute_with_callbacks(mcp_callbacks, req)

            handler = _build_interceptor_chain(execute_tool, tool_interceptors)
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

    When LangGraph is installed, elicitation is automatically supported. If an MCP
    server requests user input via elicitation during tool execution, a LangGraph
    interrupt will be triggered.

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
