import datetime
import multiprocessing
import os
import socket
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import uvicorn
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from mcp.server.websocket import websocket_server
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

from langchain_mcp_adapters.client import MultiServerMCPClient
from tests.servers.time_server import mcp as time_mcp


def make_server_app() -> Starlette:
    server = time_mcp._mcp_server

    async def handle_ws(websocket):
        async with websocket_server(websocket.scope, websocket.receive, websocket.send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    app = Starlette(
        routes=[
            WebSocketRoute("/ws", endpoint=handle_ws),
        ]
    )

    return app


def run_server(server_port: int) -> None:
    app = make_server_app()
    server = uvicorn.Server(
        config=uvicorn.Config(app=app, host="127.0.0.1", port=server_port, log_level="error")
    )
    server.run()

    # Give server time to start
    while not server.started:
        time.sleep(0.5)


@pytest.fixture()
def server(server_port: int) -> Generator[None, None, None]:
    proc = multiprocessing.Process(
        target=run_server, kwargs={"server_port": server_port}, daemon=True
    )
    proc.start()

    # Wait for server to be running
    max_attempts = 20
    attempt = 0

    while attempt < max_attempts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", server_port))
                break
        except ConnectionRefusedError:
            time.sleep(0.1)
            attempt += 1
    else:
        raise RuntimeError(f"Server failed to start after {max_attempts} attempts")

    yield

    # Signal the server to stop
    proc.kill()
    proc.join(timeout=2)
    if proc.is_alive():
        raise RuntimeError("Server process is still alive after attempting to terminate it")


@pytest.mark.asyncio
async def test_multi_server_mcp_client(server, server_port: int):
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
            "time": {"transport": "websocket", "url": f"ws://127.0.0.1:{server_port}/ws"},
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
        parsed_time = datetime.datetime.strptime(result, "%H:%M:%S.%f").time()
        current_time = datetime.datetime.now().time()
        time_diff = abs(
            (current_time.hour * 3600 + current_time.minute * 60 + current_time.second)
            - (parsed_time.hour * 3600 + parsed_time.minute * 60 + parsed_time.second)
        )
        assert time_diff < 5


@pytest.mark.asyncio
async def test_multi_server_connect_methods(server, server_port: int):
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

        await client.connect_to_server_via_websocket("time", url=f"ws://127.0.0.1:{server_port}/ws")

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
