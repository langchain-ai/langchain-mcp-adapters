"""Tests for experimental MCP Tasks support."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
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

from langchain_mcp_adapters.tools import (
    _call_tool_as_task,
    convert_mcp_tool_to_langchain_tool,
    load_mcp_tools,
)
from tests.utils import IsLangChainID


def _make_task(task_id: str = "task-123", status=TASK_STATUS_COMPLETED) -> Task:
    now = datetime.now(timezone.utc)
    return Task(
        taskId=task_id,
        status=status,
        createdAt=now,
        lastUpdatedAt=now,
        ttl=None,
    )


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

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "title": "Schema",
}


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
    await lc_tool.ainvoke({"args": {"n": 1}, "id": "tc2", "type": "tool_call"})

    session.call_tool.assert_called_once()
    session.experimental.call_tool_as_task.assert_not_called()


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

    await tools[0].ainvoke({"args": {"n": 5}, "id": "tc3", "type": "tool_call"})

    session.experimental.call_tool_as_task.assert_called_once()
    session.call_tool.assert_not_called()
