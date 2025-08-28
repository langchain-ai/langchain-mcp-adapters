import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


@pytest.mark.asyncio
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

    # Initialize client without initial connections
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
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


@pytest.mark.asyncio
async def test_get_prompt():
    """Test retrieving prompts from MCP servers."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
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


@dataclass
class EnvInheritanceGiven:
    """Configuration for environment variable inheritance test setup."""
    client_env_config: Optional[Dict[str, str]]
    env_vars: Dict[str, str]


@dataclass
class EnvInheritanceExpected:
    """Configuration for environment variable inheritance test expectations."""
    env_vars: Dict[str, str]
    not_env_vars: List[str]


async def _test_env_inheritance_case(given: EnvInheritanceGiven, expected: EnvInheritanceExpected):
    """Helper function to test environment variable inheritance scenarios.
    
    Args:
        given: Configuration for what to set up before running the test
        expected: Configuration for what to expect after running the test
    """
    import os
    import tempfile
    import json
    
    # Set up test environment variables in parent process
    # Note: Variable names are unique per test to avoid conflicts
    for var_name, var_value in given.env_vars.items():
        os.environ[var_name] = var_value
    
    test_server_path = None
    try:
        # Create temporary file that persists (delete=False) so subprocess can read it
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_file.write("""
from mcp.server.fastmcp import FastMCP
import os
import json

mcp = FastMCP("EnvTest")

@mcp.tool()
def get_all_env_vars() -> str:
    '''Return all environment variables as JSON'''
    return json.dumps(dict(os.environ))

if __name__ == "__main__":
    mcp.run(transport="stdio")
""")
        temp_file.close()
        test_server_path = temp_file.name
        
        client_config = {
            "envtest": {
                "command": "python",
                "args": [test_server_path],
                "transport": "stdio",
            }
        }
        
        if given.client_env_config is not None:
            client_config["envtest"]["env"] = given.client_env_config
        
        client = MultiServerMCPClient(client_config)
        
        # Test that the server can start and load tools
        tools = await client.get_tools()
        assert len(tools) == 1  # get_all_env_vars tool
        
        # Get environment variables from the server using the get_all_env_vars tool
        env_tool = tools[0]
        result = await env_tool.ainvoke({})
        env_vars = json.loads(result)
        
        # Make env var assertions
        for var_name, expected_value in expected.env_vars.items():
            assert var_name in env_vars, f"Expected {var_name} to be in environment"
            assert env_vars[var_name] == expected_value, f"Expected {var_name}='{expected_value}', got '{env_vars[var_name]}'"
        
        for var_name in expected.not_env_vars:
            assert var_name not in env_vars, f"Expected {var_name} to NOT be in environment"
        
    finally:
        # Clean up the temporary file
        if test_server_path:
            try:
                Path(test_server_path).unlink(missing_ok=True)
            except Exception:
                # If we can't delete it, that's okay - it's a temp file
                pass
        
        # Clean up test environment variables
        for var_name in given.env_vars:
            if var_name in os.environ:
                del os.environ[var_name]


@pytest.mark.asyncio
async def test_stdio_environment_inheritance_none():
    """Test that stdio sessions inherit all environment variables when env is None."""
    test_var1 = "TEST_ENV_INHERITANCE_NONE1"
    test_var2 = "TEST_ENV_INHERITANCE_NONE2"
    
    await _test_env_inheritance_case(
        given=EnvInheritanceGiven(
            client_env_config=None,  # Default behavior - should inherit all
            # MCP library creates subprocess with empty env, OS provides minimal default environment
            # (PATH typically present on Unix-like systems) in addition to below custom env vars
            env_vars={
                test_var1: "test_value_123",
                test_var2: "test_value_456",
            },
        ),
        expected=EnvInheritanceExpected(
            env_vars={
                test_var1: "test_value_123",
                test_var2: "test_value_456",
                "PATH": os.environ.get("PATH", ""),  # Should inherit PATH (OS provided on Unix-like systems)
            },
            not_env_vars=[],
        ),
    )


@pytest.mark.asyncio
async def test_stdio_environment_inheritance_empty_dict():
    """Test that stdio sessions get no environment variables when env is empty dict."""
    test_var1 = "TEST_ENV_INHERITANCE_EMPTY1"
    test_var2 = "TEST_ENV_INHERITANCE_EMPTY2"
    
    await _test_env_inheritance_case(
        given=EnvInheritanceGiven(
            client_env_config={},  # Explicit empty dict
            # MCP library creates subprocess with empty env, OS provides minimal default environment
            # (PATH typically present on Unix-like systems) in addition to below custom env vars
            env_vars={
                test_var1: "test_value_789",
                test_var2: "test_value_012",
            },
        ),
        expected=EnvInheritanceExpected(
            env_vars={
                "PATH": os.environ.get("PATH", ""),  # Should inherit PATH (OS provided on Unix-like systems)
            },
            not_env_vars=[
                test_var1,
                test_var2,
            ],
        ),
    )


@pytest.mark.asyncio
async def test_stdio_environment_inheritance_override():
    """Test that stdio sessions inherit parent env vars and can override specific ones."""
    test_var1 = "TEST_ENV_INHERITANCE_OVERRIDE1"
    test_var2 = "TEST_ENV_INHERITANCE_OVERRIDE2"
    test_var3 = "TEST_ENV_INHERITANCE_OVERRIDE3"
    
    await _test_env_inheritance_case(
        given=EnvInheritanceGiven(
            client_env_config={
                test_var1: "override_value1",  # Override this one
                test_var2: "override_value2",  # Override this one
                "NEW_VAR": "new_value",        # Add new one
                # test_var3 not specified - should inherit parent value
            },
            # MCP library creates subprocess with empty env, OS provides minimal default environment
            # (PATH typically present on Unix-like systems) in addition to below custom env vars
            env_vars={
                test_var1: "parent_value1",
                test_var2: "parent_value2", 
                test_var3: "parent_value3",
            },
        ),
        expected=EnvInheritanceExpected(
            env_vars={
                test_var1: "override_value1",  # Should be overridden
                test_var2: "override_value2",  # Should be overridden
                test_var3: "parent_value3",    # Should inherit parent value
                "NEW_VAR": "new_value",        # Should be added
                "PATH": os.environ.get("PATH", ""),  # Should inherit PATH (OS provided on Unix-like systems)
            },
            not_env_vars=[],
        ),
    )
