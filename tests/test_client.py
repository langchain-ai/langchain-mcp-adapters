import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from mcp.types import Prompt

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


async def test_multi_server_mcp_client(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test that MultiServerMCPClient can connect to multiple servers and load tools."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "command": "python3",
                "args": [weather_server_path],
                "transport": "stdio",
            },
            "time": {
                "url": f"ws://127.0.0.1:{websocket_server_port}/ws",
                "transport": "websocket",
            },
        },
    )
    # Check that we have tools from both servers
    all_tools = await client.get_tools()

    # Should have 3 tools (add, multiply, get_weather)
    assert len(all_tools) == 4

    # Check that tools are BaseTool instances
    for tool in all_tools:
        assert isinstance(tool, BaseTool)

    # Verify tool names
    tool_names = {tool.name for tool in all_tools}
    assert tool_names == {"add", "multiply", "get_weather", "get_time"}

    # Check math server tools
    math_tools = await client.get_tools(server_name="math")
    assert len(math_tools) == 2
    math_tool_names = {tool.name for tool in math_tools}
    assert math_tool_names == {"add", "multiply"}

    # Check weather server tools
    weather_tools = await client.get_tools(server_name="weather")
    assert len(weather_tools) == 1
    assert weather_tools[0].name == "get_weather"

    # Check time server tools
    time_tools = await client.get_tools(server_name="time")
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


async def test_multi_server_connect_methods(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test the different connect methods for MultiServerMCPClient."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    # Initialize client without initial connections
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "time": {
                "url": f"ws://127.0.0.1:{websocket_server_port}/ws",
                "transport": "websocket",
            },
        },
    )
    tool_names = set()
    async with client.session("math") as session:
        tools = await load_mcp_tools(session)
        assert len(tools) == 2
        result = await tools[0].ainvoke({"a": 2, "b": 3})
        assert result == "5"

        for tool in tools:
            tool_names.add(tool.name)

    async with client.session("time") as session:
        tools = await load_mcp_tools(session)
        assert len(tools) == 1
        result = await tools[0].ainvoke({"args": ""})
        assert result == "5:20:00 PM EST"

        for tool in tools:
            tool_names.add(tool.name)

    assert tool_names == {"add", "multiply", "get_time"}


async def test_get_prompt():
    """Test retrieving prompts from MCP servers."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            }
        },
    )
    # Test getting a prompt from the math server
    messages = await client.get_prompt(
        "math",
        "configure_assistant",
        arguments={"skills": "math, addition, multiplication"},
    )

    # Check that we got an AIMessage back
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert "You are a helpful assistant" in messages[0].content
    assert "math, addition, multiplication" in messages[0].content


async def test_get_prompts():
    """Test retrieving prompts from MCP servers."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            }
        },
    )
    # Test getting prompts from the math server
    prompts = await client.get_prompts(
        "math",
    )

    # Check that we got multiple Prompts back
    assert len(prompts) == 3
    assert all(isinstance(prompt, Prompt) for prompt in prompts)

    # Check the first prompt (configure_assistant)
    configure_prompt = [p for p in prompts if p.name == "configure_assistant"][0]
    assert configure_prompt.description == ""
    assert configure_prompt.title is None
    assert len(configure_prompt.arguments) == 1
    assert configure_prompt.arguments[0].name == "skills"
    assert configure_prompt.arguments[0].required is True

    # Check the second prompt (math_problem_solver)
    solver_prompt = [p for p in prompts if p.name == "math_problem_solver"][0]
    assert solver_prompt.description == ""
    assert len(solver_prompt.arguments) == 1
    assert solver_prompt.arguments[0].name == "problem_type"
    assert solver_prompt.arguments[0].required is True

    # Check the third prompt (calculation_guide)
    guide_prompt = [p for p in prompts if p.name == "calculation_guide"][0]
    assert guide_prompt.description == ""
    # This prompt has no arguments
    assert guide_prompt.arguments == []


async def test_get_prompts_invalid_server():
    """Test that get_prompts raises an error for invalid server name."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            }
        },
    )

    # Test getting prompts from a non-existent server
    with pytest.raises(ValueError) as exc_info:
        await client.get_prompts("nonexistent_server")

    error_msg = str(exc_info.value)
    assert "Couldn't find a server with name 'nonexistent_server'" in error_msg
    assert "expected one of '['math']'" in error_msg


async def test_get_prompts_multiple_servers(
    socket_enabled,
    websocket_server,
    websocket_server_port: int,
):
    """Test retrieving prompts from multiple servers."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "command": "python3",
                "args": [weather_server_path],
                "transport": "stdio",
            },
        },
    )

    # Test getting prompts from math server
    math_prompts = await client.get_prompts("math")
    assert len(math_prompts) == 3

    # Test getting prompts from weather server (may have different prompts)
    weather_prompts = await client.get_prompts("weather")
    # Weather server may or may not have prompts, just verify it doesn't crash
    assert isinstance(weather_prompts, list)
