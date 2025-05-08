import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from langchain_mcp_adapters.client import MultiServerMCPClient


@pytest.mark.asyncio
async def test_multi_server_mcp_client(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test that the MultiServerMCPClient can connect to multiple servers and load tools."""

    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")

    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": [weather_server_path],
                "transport": "stdio",
            },
            "time": {
                "url": f"ws://127.0.0.1:{websocket_server_port}/ws",
                "transport": "websocket",
            },
        }
    ) as client:
        # Check that we have tools from both servers
        all_tools = client.get_tools()

        # Should have 3 tools (add, multiply, get_weather)
        assert len(all_tools) == 4

        # Check that tools are BaseTool instances
        for tool in all_tools:
            assert isinstance(tool, BaseTool)

        # Verify tool names
        tool_names = {tool.name for tool in all_tools}
        assert tool_names == {"add", "multiply", "get_weather", "get_time"}

        # Check math server tools
        math_tools = client.server_name_to_tools["math"]
        assert len(math_tools) == 2
        math_tool_names = {tool.name for tool in math_tools}
        assert math_tool_names == {"add", "multiply"}

        # Check weather server tools
        weather_tools = client.server_name_to_tools["weather"]
        assert len(weather_tools) == 1
        assert weather_tools[0].name == "get_weather"

        # Check time server tools
        time_tools = client.server_name_to_tools["time"]
        assert len(time_tools) == 1
        assert time_tools[0].name == "get_time"

        # Test that we can call a math tool
        add_tool = next(tool for tool in all_tools if tool.name == "add")
        result = await add_tool.ainvoke({"a": 2, "b": 3})
        assert result == "5"

        # Test that we can call a weather tool
        weather_tool = next(tool for tool in all_tools if tool.name == "get_weather")
        result = await weather_tool.ainvoke({"location": "London"})
        assert result == "It's always sunny in London"

        # Test the multiply tool
        multiply_tool = next(tool for tool in all_tools if tool.name == "multiply")
        result = await multiply_tool.ainvoke({"a": 4, "b": 5})
        assert result == "20"

        # Test that we can call a time tool
        time_tool = next(tool for tool in all_tools if tool.name == "get_time")
        result = await time_tool.ainvoke({"args": ""})
        assert result == "5:20:00 PM EST"


@pytest.mark.asyncio
async def test_multi_server_connect_methods(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test the different connect methods for MultiServerMCPClient."""

    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")

    # Initialize client without initial connections
    client = MultiServerMCPClient()
    async with client:
        # Connect to math server using connect_to_server
        await client.connect_to_server(
            "math", transport="stdio", command="python", args=[math_server_path]
        )

        # Connect to weather server using connect_to_server_via_stdio
        await client.connect_to_server_via_stdio(
            "weather", command="python", args=[weather_server_path]
        )

        await client.connect_to_server_via_websocket(
            "time", url=f"ws://127.0.0.1:{websocket_server_port}/ws"
        )

        # Check that we have tools from both servers
        all_tools = client.get_tools()
        assert len(all_tools) == 4

        # Verify tool names
        tool_names = {tool.name for tool in all_tools}
        assert tool_names == {"add", "multiply", "get_weather", "get_time"}


@pytest.mark.asyncio
async def test_get_prompt():
    """Test retrieving prompts from MCP servers."""

    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [math_server_path],
                "transport": "stdio",
            }
        }
    ) as client:
        # Test getting a prompt from the math server
        messages = await client.get_prompt(
            "math", "configure_assistant", {"skills": "math, addition, multiplication"}
        )

        # Check that we got an AIMessage back
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert "You are a helpful assistant" in messages[0].content
        assert "math, addition, multiplication" in messages[0].content


@pytest.mark.asyncio
async def test_multi_server_explicit_session_management(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test explicit session management with connect_all_sessions and close_all_sessions."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": [weather_server_path],
                "transport": "stdio",
            },
            "time": {
                "url": f"ws://127.0.0.1:{websocket_server_port}/ws",
                "transport": "websocket",
            },
        }
    )

    # 1. Connect all sessions
    await client.connect_all_sessions()
    assert client._is_active_via_connect_all
    assert client.exit_stack is not None
    assert len(client.sessions) == 3
    assert len(client.server_name_to_tools) == 3
    
    all_tools_initial = client.get_tools()
    assert len(all_tools_initial) == 4
    tool_names_initial = {tool.name for tool in all_tools_initial}
    assert tool_names_initial == {"add", "multiply", "get_weather", "get_time"}

    # Test a tool
    add_tool = next(tool for tool in all_tools_initial if tool.name == "add")
    result = await add_tool.ainvoke({"a": 10, "b": 5})
    assert result == "15"

    # 2. Test context manager behavior AFTER explicit connect
    async with client as client_cm:
        assert client_cm._is_active_via_connect_all  # Should still be true
        assert client_cm.exit_stack is not None # Original exit_stack
        
        # Tools should still be available
        all_tools_cm = client_cm.get_tools()
        assert len(all_tools_cm) == 4
        
        # Ensure it's the same tools/sessions
        assert len(client_cm.sessions) == 3
        
        # Test a tool from within context manager
        weather_tool_cm = next(tool for tool in all_tools_cm if tool.name == "get_weather")
        result_cm = await weather_tool_cm.ainvoke({"location": "TestCity"})
        assert result_cm == "It's always sunny in TestCity"

    # 3. Check that sessions are still active after context manager exit
    assert client._is_active_via_connect_all
    assert client.exit_stack is not None # Original exit_stack should persist
    assert len(client.sessions) == 3 # Sessions should not have been cleared by __aexit__
    assert len(client.server_name_to_tools) == 3

    # Test a tool again to ensure session is live
    multiply_tool_after_cm = next(tool for tool in client.get_tools() if tool.name == "multiply")
    result_after_cm = await multiply_tool_after_cm.ainvoke({"a": 3, "b": 3})
    assert result_after_cm == "9"

    # 4. Close all sessions
    await client.close_all_sessions()
    assert not client._is_active_via_connect_all
    assert client.exit_stack is None
    assert len(client.sessions) == 0
    assert len(client.server_name_to_tools) == 0

    # 5. Verify tools are gone
    all_tools_after_close = client.get_tools()
    assert len(all_tools_after_close) == 0

    # 6. Test idempotency of close_all_sessions (should do nothing)
    await client.close_all_sessions() 
    assert not client._is_active_via_connect_all

    # 7. Test that __aenter__ and __aexit__ work independently if connect_all_sessions wasn't called
    client_short_lived = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [math_server_path],
                "transport": "stdio",
            }
        }
    )
    assert not client_short_lived._is_active_via_connect_all
    async with client_short_lived as cl_sl:
        assert not cl_sl._is_active_via_connect_all # Should remain false
        assert cl_sl.exit_stack is not None # Local exit_stack
        assert len(cl_sl.sessions) == 1
        math_tools_sl = cl_sl.get_tools()
        assert len(math_tools_sl) == 2
        add_tool_sl = next(tool for tool in math_tools_sl if tool.name == "add")
        res_sl = await add_tool_sl.ainvoke({"a": 1, "b": 2})
        assert res_sl == "3"
    
    assert cl_sl.exit_stack is None # Local exit_stack should be gone
    assert len(cl_sl.sessions) == 0
    assert len(cl_sl.server_name_to_tools) == 0


@pytest.mark.asyncio
async def test_error_on_connect_all_sessions_if_already_active(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test that connect_all_sessions raises RuntimeError if called when already active."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    client = MultiServerMCPClient(
        {"math": {"command": "python", "args": [math_server_path], "transport": "stdio"}}
    )
    await client.connect_all_sessions()
    assert client._is_active_via_connect_all
    with pytest.raises(RuntimeError, match="Sessions are already active"):
        await client.connect_all_sessions()
    # Cleanup
    await client.close_all_sessions()

@pytest.mark.asyncio
async def test_error_on_connect_all_sessions_if_context_active_without_flag(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test that connect_all_sessions raises RuntimeError if __aenter__ created a stack."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    client = MultiServerMCPClient(
        {"math": {"command": "python", "args": [math_server_path], "transport": "stdio"}}
    )
    async with client: # This will create an exit_stack
      assert client.exit_stack is not None
      assert not client._is_active_via_connect_all
      with pytest.raises(RuntimeError, match="An exit stack is already present"):
          await client.connect_all_sessions()
    assert client.exit_stack is None # Ensure __aexit__ cleaned up
