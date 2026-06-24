"""Tests for the experimental use_task_execution flag."""

import asyncio
import contextlib
import multiprocessing
import socket
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn
from mcp import types
from mcp.server.experimental.task_context import ServerTaskContext
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import (
    CallToolResult,
    CreateTaskResult,
    GetTaskResult,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_INPUT_REQUIRED,
    TASK_STATUS_WORKING,
    Task,
    TextContent,
)
from mcp.types import Tool as MCPTool

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import (
    _call_tool_as_task,
    convert_mcp_tool_to_langchain_tool,
    load_mcp_tools,
)
from tests.utils import IsLangChainID

_TASK_SERVER_PORT = 8185


# ---------------------------------------------------------------------------
# Low-level task server — module-level for multiprocessing picklability
# ---------------------------------------------------------------------------


def _run_task_server(server_port: int) -> None:
    """Entry point for the task server subprocess.

    Uses the low-level mcp Server (not FastMCP) so the call_tool handler can
    detect task metadata and return CreateTaskResult for background execution.
    """
    from mcp.server.fastmcp.server import StreamableHTTPASGIApp
    from starlette.applications import Starlette
    from starlette.routing import Route

    server = Server("langchain-mcp-task-test")
    server.experimental.enable_tasks()

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="slow_add",
                description="Add two numbers with an async delay",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="echo",
                description="Return the message unchanged",
                inputSchema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> types.CallToolResult | types.CreateTaskResult:
        ctx = server.request_context
        experimental = ctx.experimental
        task_support = experimental._task_support

        async def _execute() -> types.CallToolResult:
            if name == "slow_add":
                await asyncio.sleep(0.05)
                total = arguments["a"] + arguments["b"]
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(total))],
                    isError=False,
                )
            if name == "echo":
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=arguments["message"])],
                    isError=False,
                )
            msg = f"Unknown tool: {name}"
            raise ValueError(msg)

        if experimental.is_task and task_support is not None:
            task = await task_support.store.create_task(experimental.task_metadata)

            async def background() -> None:
                task_ctx = ServerTaskContext(
                    task=task,
                    store=task_support.store,
                    session=ctx.session,
                    queue=task_support.queue,
                    handler=task_support.handler,
                )
                try:
                    result = await _execute()
                    await task_ctx.complete(result)
                except Exception as exc:  # noqa: BLE001
                    await task_ctx.fail(str(exc))

            task_support.task_group.start_soon(background)
            return types.CreateTaskResult(task=task)

        return await _execute()

    session_manager = StreamableHTTPSessionManager(app=server, stateless=False)
    asgi_app = StreamableHTTPASGIApp(session_manager)
    starlette_app = Starlette(
        routes=[Route("/mcp", endpoint=asgi_app)],
        lifespan=lambda _app: session_manager.run(),
    )

    uvicorn_server = uvicorn.Server(
        config=uvicorn.Config(
            app=starlette_app,
            host="127.0.0.1",
            port=server_port,
            log_level="error",
        )
    )
    uvicorn_server.run()


@contextlib.contextmanager
def _run_task_server_ctx(server_port: int):
    """Context manager that starts the task server in a subprocess."""
    proc = multiprocessing.Process(
        target=_run_task_server,
        kwargs={"server_port": server_port},
        daemon=True,
    )
    proc.start()

    for _ in range(40):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", server_port))
                break
        except ConnectionRefusedError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("Task server failed to start")

    try:
        yield
    finally:
        proc.kill()
        proc.join(timeout=2)

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "title": "Schema",
}


def _make_task(task_id: str = "task-123", status=TASK_STATUS_COMPLETED) -> Task:
    now = datetime.now(timezone.utc)
    return Task(taskId=task_id, status=status, createdAt=now, lastUpdatedAt=now, ttl=None)


def _make_session(
    task_id: str = "task-123",
    poll_statuses=None,
    final_result: CallToolResult | None = None,
) -> AsyncMock:
    """Build a mock session whose session.experimental API returns task fixtures."""
    if poll_statuses is None:
        poll_statuses = [TASK_STATUS_COMPLETED]
    if final_result is None:
        final_result = CallToolResult(
            content=[TextContent(type="text", text="task result")],
            isError=False,
        )

    now = datetime.now(timezone.utc)

    async def mock_poll_task(tid):
        for status in poll_statuses:
            yield GetTaskResult(
                taskId=tid,
                status=status,
                statusMessage=None,
                createdAt=now,
                lastUpdatedAt=now,
                ttl=None,
            )

    session = AsyncMock()
    session.experimental.call_tool_as_task.return_value = CreateTaskResult(
        task=_make_task(task_id)
    )
    session.experimental.poll_task = mock_poll_task
    session.experimental.cancel_task = AsyncMock()
    session.experimental.get_task_result.return_value = final_result
    return session


# ---------------------------------------------------------------------------
# Unit tests for _call_tool_as_task
# ---------------------------------------------------------------------------


async def test_call_tool_as_task_success():
    """Happy path: task completes and the CallToolResult is returned."""
    expected = CallToolResult(
        content=[TextContent(type="text", text="done")],
        isError=False,
    )
    session = _make_session(
        task_id="t1",
        poll_statuses=[TASK_STATUS_WORKING, TASK_STATUS_COMPLETED],
        final_result=expected,
    )

    result = await _call_tool_as_task(session, "my_tool", {"x": 1})

    session.experimental.call_tool_as_task.assert_called_once_with("my_tool", {"x": 1})
    session.experimental.get_task_result.assert_called_once_with("t1", CallToolResult)
    assert result is expected


async def test_call_tool_as_task_input_required_cancels_and_raises():
    """Task reaching input_required must be cancelled and raise RuntimeError."""
    session = _make_session(
        task_id="t2",
        poll_statuses=[TASK_STATUS_WORKING, TASK_STATUS_INPUT_REQUIRED],
    )

    with pytest.raises(RuntimeError, match="input_required"):
        await _call_tool_as_task(session, "my_tool", {})

    session.experimental.cancel_task.assert_called_once_with("t2")
    session.experimental.get_task_result.assert_not_called()


# ---------------------------------------------------------------------------
# Integration with convert_mcp_tool_to_langchain_tool
# ---------------------------------------------------------------------------


async def test_convert_mcp_tool_routes_through_tasks_when_flag_set():
    """use_task_execution=True must call experimental API, not call_tool."""
    expected = CallToolResult(
        content=[TextContent(type="text", text="42")],
        isError=False,
    )
    session = _make_session(task_id="t3", final_result=expected)

    lc_tool = convert_mcp_tool_to_langchain_tool(
        session,
        MCPTool(name="calc", description="calc", inputSchema=_SIMPLE_SCHEMA),
        use_task_execution=True,
    )
    result = await lc_tool.ainvoke(
        {"args": {"n": 7}, "id": "tc1", "type": "tool_call"}
    )

    session.experimental.call_tool_as_task.assert_called_once_with("calc", {"n": 7})
    session.call_tool.assert_not_called()
    assert result.name == "calc"
    assert result.tool_call_id == "tc1"
    assert result.content == [{"type": "text", "text": "42", "id": IsLangChainID}]


async def test_convert_mcp_tool_uses_regular_call_tool_by_default():
    """Without the flag, the regular call_tool path must be used."""
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="99")],
        isError=False,
    )

    lc_tool = convert_mcp_tool_to_langchain_tool(
        session,
        MCPTool(name="calc", description="calc", inputSchema=_SIMPLE_SCHEMA),
    )
    result = await lc_tool.ainvoke(
        {"args": {"n": 1}, "id": "tc2", "type": "tool_call"}
    )

    session.call_tool.assert_called_once()
    session.experimental.call_tool_as_task.assert_not_called()
    assert result.name == "calc"
    assert result.tool_call_id == "tc2"
    assert result.content == [{"type": "text", "text": "99", "id": IsLangChainID}]


# ---------------------------------------------------------------------------
# Propagation through load_mcp_tools
# ---------------------------------------------------------------------------


async def test_load_mcp_tools_propagates_use_task_execution():
    """load_mcp_tools must pass use_task_execution down to each converted tool."""
    mcp_tool = MCPTool(
        name="adder",
        description="Adds things",
        inputSchema=_SIMPLE_SCHEMA,
    )
    expected = CallToolResult(
        content=[TextContent(type="text", text="sum")],
        isError=False,
    )
    session = _make_session(task_id="t4", final_result=expected)
    session.list_tools.return_value = MagicMock(tools=[mcp_tool], nextCursor=None)

    tools = await load_mcp_tools(session, use_task_execution=True)
    assert len(tools) == 1

    result = await tools[0].ainvoke({"args": {"n": 5}, "id": "tc3", "type": "tool_call"})

    session.experimental.call_tool_as_task.assert_called_once()
    session.call_tool.assert_not_called()
    assert result.name == "adder"
    assert result.tool_call_id == "tc3"


# ---------------------------------------------------------------------------
# Integration tests against a real FastMCP server with tasks enabled
# ---------------------------------------------------------------------------


async def test_task_execution_returns_correct_result(socket_enabled) -> None:
    """Tool called via task API returns the expected result end-to-end."""
    with _run_task_server_ctx(_TASK_SERVER_PORT):
        client = MultiServerMCPClient(
            {"test": {"url": f"http://localhost:{_TASK_SERVER_PORT}/mcp", "transport": "http"}},
            use_task_execution=True,
        )
        tools = await client.get_tools()
        tool_map = {t.name: t for t in tools}

        result = await tool_map["slow_add"].ainvoke(
            {"args": {"a": 3, "b": 4}, "id": "i1", "type": "tool_call"}
        )

        assert result.name == "slow_add"
        assert result.tool_call_id == "i1"
        assert result.content == [{"type": "text", "text": "7", "id": IsLangChainID}]


async def test_task_execution_concurrent_calls(socket_enabled) -> None:
    """Multiple tools can be called concurrently via the task API."""
    with _run_task_server_ctx(_TASK_SERVER_PORT):
        client = MultiServerMCPClient(
            {"test": {"url": f"http://localhost:{_TASK_SERVER_PORT}/mcp", "transport": "http"}},
            use_task_execution=True,
        )
        tools = await client.get_tools()
        tool_map = {t.name: t for t in tools}

        echo_result, add_result = await asyncio.gather(
            tool_map["echo"].ainvoke(
                {"args": {"message": "hello"}, "id": "i2", "type": "tool_call"}
            ),
            tool_map["slow_add"].ainvoke(
                {"args": {"a": 10, "b": 20}, "id": "i3", "type": "tool_call"}
            ),
        )

        assert echo_result.content[0]["text"] == "hello"
        assert add_result.content[0]["text"] == "30"


async def test_task_execution_transparent_to_agents(socket_enabled) -> None:
    """use_task_execution=True and False produce identical content for agents."""
    with _run_task_server_ctx(_TASK_SERVER_PORT):
        client_task = MultiServerMCPClient(
            {"test": {"url": f"http://localhost:{_TASK_SERVER_PORT}/mcp", "transport": "http"}},
            use_task_execution=True,
        )
        client_direct = MultiServerMCPClient(
            {"test": {"url": f"http://localhost:{_TASK_SERVER_PORT}/mcp", "transport": "http"}},
        )

        tools_task = {t.name: t for t in await client_task.get_tools()}
        tools_direct = {t.name: t for t in await client_direct.get_tools()}

        result_task = await tools_task["echo"].ainvoke(
            {"args": {"message": "same"}, "id": "i4", "type": "tool_call"}
        )
        result_direct = await tools_direct["echo"].ainvoke(
            {"args": {"message": "same"}, "id": "i5", "type": "tool_call"}
        )

        assert result_task.content[0]["text"] == result_direct.content[0]["text"]
