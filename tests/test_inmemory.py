import importlib.util
import os
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from langchain_mcp_adapters.client import MultiServerMCPClient


def _load_module(module_name: str, server_path: str) -> any:
    module_spec = importlib.util.spec_from_file_location(module_name, server_path)
    assert module_spec is not None

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


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
    # import weather_server
    weather_server_module = _load_module("weather_server", weather_server_path)

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "server": weather_server_module.mcp,
                "transport": "in_memory",
            },
        },
    )
    # Check that we have tools from both servers
    all_tools = await client.get_tools()

    # Should have 3 tools (add, multiply, get_weather)
    assert len(all_tools) == 3

    # Check that tools are BaseTool instances
    for tool in all_tools:
        assert isinstance(tool, BaseTool)

    # Verify tool names
    tool_names = {tool.name for tool in all_tools}
    assert tool_names == {"add", "multiply", "get_weather"}

    # Check math server tools
    math_tools = await client.get_tools(server_name="math")
    assert len(math_tools) == 2
    math_tool_names = {tool.name for tool in math_tools}
    assert math_tool_names == {"add", "multiply"}

    # Check weather server tools
    weather_tools = await client.get_tools(server_name="weather")
    assert len(weather_tools) == 1
    assert weather_tools[0].name == "get_weather"

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


async def test_get_prompt():
    """Test retrieving prompts from MCP servers."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    # import weather_server
    math_server_module = _load_module("math_server", math_server_path)

    client = MultiServerMCPClient(
        {
            "math": {
                "server": math_server_module.mcp,
                "transport": "in_memory",
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
