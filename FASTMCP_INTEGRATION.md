# FastMCP Integration Guide

This guide explains how to use the enhanced FastMCP integration features in `langchain-mcp-adapters` to seamlessly connect FastMCP servers with LangGraph agents.

## Overview

The `langchain-mcp-adapters` library now provides enhanced support for FastMCP servers, making it easy to:

- Connect to FastMCP servers (both local and remote)
- Automatically convert FastMCP tools to LangChain tools
- Use FastMCP tools in LangGraph agents
- Manage multiple FastMCP servers

## Installation

```bash
pip install langchain-mcp-adapters
```

For FastMCP 2.0 support (optional):
```bash
pip install fastmcp
```

## Quick Start

### 1. Simple FastMCP Server Integration

```python
from langchain_mcp_adapters import FastMCPClient
from langgraph.prebuilt import create_react_agent

# Connect to a FastMCP server
client = FastMCPClient(server_script="./my_fastmcp_server.py")

# Get tools from the server
tools = await client.get_tools()

# Create LangGraph agent with the tools
agent = create_react_agent("openai:gpt-4", tools)

# Use the agent
response = await agent.ainvoke({"messages": "Help me with calculations"})
```

### 2. Quick Tool Loading

```python
from langchain_mcp_adapters import quick_load_fastmcp_tools
from langgraph.prebuilt import create_react_agent

# Quick load tools from FastMCP server
tools = await quick_load_fastmcp_tools(server_script="./math_server.py")

# Create agent
agent = create_react_agent("openai:gpt-4", tools)
```

### 3. Synchronous Usage

```python
from langchain_mcp_adapters import quick_load_fastmcp_tools_sync

# Load tools synchronously
tools = quick_load_fastmcp_tools_sync(server_script="./math_server.py")

# Use with LangGraph (note: agent execution is still async)
agent = create_react_agent("openai:gpt-4", tools)
```

## Advanced Usage

### Multiple FastMCP Servers

```python
from langchain_mcp_adapters import FastMCPMultiClient

# Configure multiple servers
servers = {
    "math": {
        "server_script": "./math_server.py"
    },
    "weather": {
        "server_url": "http://localhost:8000/mcp/"
    },
    "database": {
        "server_script": "./db_server.py",
        "server_command": "python3",
        "server_args": ["./db_server.py", "--config", "prod.json"]
    }
}

# Create multi-client
multi_client = FastMCPMultiClient(servers)

# Get all tools from all servers
all_tools = await multi_client.get_tools_flat()

# Get tools from specific server
math_tools = await multi_client.get_tools_from_server("math")

# Create agent with all tools
agent = create_react_agent("openai:gpt-4", all_tools)
```

### HTTP Server Integration

```python
from langchain_mcp_adapters import FastMCPClient

# Connect to HTTP FastMCP server
client = FastMCPClient(server_url="http://localhost:8000/mcp/")

# Get tools
tools = await client.get_tools()

# Use with LangGraph
agent = create_react_agent("openai:gpt-4", tools)
```

### Custom Server Configuration

```python
from langchain_mcp_adapters import FastMCPClient

# Custom server configuration
client = FastMCPClient(
    server_command="python3",
    server_args=["./custom_server.py", "--port", "9000", "--debug"],
    # Additional connection parameters
    env={"CUSTOM_VAR": "value"}
)

tools = await client.get_tools()
```

### LangGraph StateGraph Integration

```python
from langchain_mcp_adapters import quick_load_fastmcp_tools
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

# Load FastMCP tools
tools = await quick_load_fastmcp_tools(server_script="./math_server.py")

# Initialize model
model = init_chat_model("openai:gpt-4")

# Define the call_model function
def call_model(state: MessagesState):
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": response}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")

# Compile and use
graph = builder.compile()
response = await graph.ainvoke({"messages": "Calculate 15 * 8"})
```

## Creating FastMCP Servers

### Example FastMCP Server

```python
# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math Server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Running the Server

```bash
# For stdio transport (default)
python math_server.py

# For HTTP transport
python -c "
from math_server import mcp
mcp.run(transport='streamable-http', host='127.0.0.1', port=8000)
"
```

## API Reference

### FastMCPClient

```python
class FastMCPClient:
    def __init__(
        self,
        server_script: Optional[str] = None,
        server_url: Optional[str] = None,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
        **kwargs
    )
    
    async def get_tools(self) -> List[BaseTool]
    async def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]
    async def list_tool_names(self) -> List[str]
    
    # Synchronous versions
    def get_tools_sync(self) -> List[BaseTool]
    def get_tool_by_name_sync(self, tool_name: str) -> Optional[BaseTool]
    def list_tool_names_sync(self) -> List[str]
```

### FastMCPMultiClient

```python
class FastMCPMultiClient:
    def __init__(self, servers: Dict[str, Dict[str, Any]])
    
    async def get_all_tools(self) -> Dict[str, List[BaseTool]]
    async def get_tools_flat(self) -> List[BaseTool]
    async def get_tools_from_server(self, server_name: str) -> List[BaseTool]
    
    # Synchronous versions
    def get_all_tools_sync(self) -> Dict[str, List[BaseTool]]
    def get_tools_flat_sync(self) -> List[BaseTool]
    def get_tools_from_server_sync(self, server_name: str) -> List[BaseTool]
```

### Convenience Functions

```python
# Create client
def create_fastmcp_client(
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> FastMCPClient

# Quick tool loading
async def quick_load_fastmcp_tools(
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> List[BaseTool]

def quick_load_fastmcp_tools_sync(
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> List[BaseTool]
```

## Error Handling

```python
from langchain_mcp_adapters import FastMCPClient

try:
    client = FastMCPClient(server_script="./my_server.py")
    tools = await client.get_tools()
except Exception as e:
    print(f"Error connecting to FastMCP server: {e}")
    # Handle error or use fallback
    tools = []

if tools:
    agent = create_react_agent("openai:gpt-4", tools)
else:
    print("No tools available")
```

## Best Practices

1. **Use async/await**: FastMCP integration is designed for async usage
2. **Handle errors gracefully**: Always wrap server connections in try-catch blocks
3. **Cache tools when possible**: Avoid repeatedly loading tools from the same server
4. **Use multi-client for multiple servers**: More efficient than individual clients
5. **Test server connectivity**: Verify servers are running before creating agents

## Troubleshooting

### Common Issues

1. **Server not found**: Ensure the server script path is correct
2. **Connection timeout**: Check if the server is running and accessible
3. **Import errors**: Make sure all dependencies are installed
4. **Tool conversion errors**: Verify FastMCP tools have proper schemas

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed connection and tool loading information
client = FastMCPClient(server_script="./my_server.py")
tools = await client.get_tools()
```

## Examples

See the `examples/fastmcp_integration.py` file for complete working examples demonstrating various integration patterns.

## Compatibility

- **FastMCP Servers**: Works with any MCP-compatible server, including FastMCP
- **LangChain**: Compatible with LangChain tools and agents
- **LangGraph**: Full support for LangGraph agents and StateGraphs
- **Python**: Requires Python 3.10+

## Migration from Standard MCP

If you're currently using the standard MCP integration, you can easily migrate:

```python
# Old way
from langchain_mcp_adapters import MultiServerMCPClient

client = MultiServerMCPClient({
    "math": {
        "command": "python",
        "args": ["./math_server.py"],
        "transport": "stdio"
    }
})
tools = await client.get_tools()

# New FastMCP way
from langchain_mcp_adapters import FastMCPClient

client = FastMCPClient(server_script="./math_server.py")
tools = await client.get_tools()
```

The new FastMCP integration provides a simpler API while maintaining full compatibility with existing MCP servers.