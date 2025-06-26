# FastMCP 2.0 Integration Guide

This guide explains how to use the FastMCP 2.0 integration features in `langchain-mcp-adapters` to seamlessly connect FastMCP 2.0 servers with LangGraph agents.

**Important**: This guide covers integration with the independent FastMCP 2.0 SDK (jlowin/fastmcp), not the `fastmcp` module in the official MCP Python SDK.

## Overview

The `langchain-mcp-adapters` library provides comprehensive support for FastMCP 2.0 servers, making it easy to:

- **Connect to FastMCP 2.0 servers** via direct instances, stdio, or HTTP
- **Automatically convert tools** from FastMCP 2.0 to LangChain format
- **Integrate with LangGraph** agents seamlessly
- **Manage multiple servers** with a single client
- **Handle errors gracefully** with robust error handling
- **Support both async and sync** usage patterns

## Installation

```bash
# Install the base library
pip install langchain-mcp-adapters

# Install FastMCP 2.0 SDK
pip install fastmcp
```

## Quick Start

### 1. Simple FastMCP 2.0 Integration

```python
from langchain_mcp_adapters import quick_load_fastmcp2_tools
from langgraph.prebuilt import create_react_agent

# Load tools from FastMCP 2.0 server
tools = await quick_load_fastmcp2_tools(
    server_script="./my_fastmcp2_server.py"
)

# Create LangGraph agent with FastMCP 2.0 tools
agent = create_react_agent("openai:gpt-4", tools)

# Use the agent
response = await agent.ainvoke({"messages": "Help me with calculations"})
```

### 2. Direct Server Instance Integration

```python
from fastmcp import FastMCP
from langchain_mcp_adapters import FastMCP2Client

# Create FastMCP 2.0 server
mcp = FastMCP("My Server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

# Connect directly to server instance
client = FastMCP2Client(server_instance=mcp)
tools = await client.get_tools()

# Use with LangGraph
agent = create_react_agent("openai:gpt-4", tools)
```

### 3. HTTP Server Integration

```python
from langchain_mcp_adapters import FastMCP2Client

# Connect to remote FastMCP 2.0 server
client = FastMCP2Client(server_url="http://localhost:8000/")
tools = await client.get_tools()

# Use with LangGraph
agent = create_react_agent("openai:gpt-4", tools)
```

## Advanced Usage

### Multiple FastMCP 2.0 Servers

```python
from langchain_mcp_adapters import FastMCP2MultiClient

# Configure multiple servers
servers = {
    "math": {"server_script": "./math_server.py"},
    "weather": {"server_url": "http://localhost:8001/"},
    "database": {"server_script": "./db_server.py"}
}

# Create multi-client
multi_client = FastMCP2MultiClient(servers)

# Get all tools from all servers
all_tools = await multi_client.get_tools_flat()

# Get tools from specific server
math_tools = await multi_client.get_tools_from_server("math")

# Use with LangGraph
agent = create_react_agent("openai:gpt-4", all_tools)
```

### Synchronous Usage

```python
from langchain_mcp_adapters import (
    quick_load_fastmcp2_tools_sync,
    FastMCP2Client
)

# Synchronous tool loading
tools = quick_load_fastmcp2_tools_sync(
    server_script="./my_server.py"
)

# Synchronous client operations
client = FastMCP2Client(server_script="./my_server.py")
tools = client.get_tools_sync()
tool_names = client.list_tool_names_sync()
specific_tool = client.get_tool_by_name_sync("add")
```

### Error Handling

```python
from langchain_mcp_adapters import quick_load_fastmcp2_tools

async def load_tools_with_fallback():
    try:
        # Try to load FastMCP 2.0 tools
        tools = await quick_load_fastmcp2_tools(
            server_script="./my_server.py"
        )
        print(f"Loaded {len(tools)} FastMCP 2.0 tools")
        return tools
    except Exception as e:
        print(f"Failed to load FastMCP 2.0 tools: {e}")
        # Fallback to empty tools or built-in tools
        return []

# Use in your application
tools = await load_tools_with_fallback()
agent = create_react_agent("openai:gpt-4", tools)
```

## Creating FastMCP 2.0 Servers

### Basic Server

```python
from fastmcp import FastMCP

# Create server
mcp = FastMCP("My Math Server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run()
```

### Advanced Server with Complex Tools

```python
from fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("Advanced Server")

@mcp.tool()
def process_data(data: List[Dict[str, Any]], operation: str) -> Dict[str, Any]:
    """Process a list of data records.
    
    Args:
        data: List of data records
        operation: Operation to perform ('sum', 'average', 'count')
    
    Returns:
        Processed result
    """
    if operation == "sum":
        total = sum(record.get("value", 0) for record in data)
        return {"operation": "sum", "result": total, "count": len(data)}
    elif operation == "average":
        values = [record.get("value", 0) for record in data]
        avg = sum(values) / len(values) if values else 0
        return {"operation": "average", "result": avg, "count": len(data)}
    elif operation == "count":
        return {"operation": "count", "result": len(data)}
    else:
        raise ValueError(f"Unknown operation: {operation}")

@mcp.tool()
def search_records(
    records: List[Dict[str, Any]], 
    field: str, 
    value: Any
) -> List[Dict[str, Any]]:
    """Search records by field value."""
    return [record for record in records if record.get(field) == value]

if __name__ == "__main__":
    mcp.run()
```

## API Reference

### FastMCP2Client

```python
class FastMCP2Client:
    def __init__(
        self,
        server_instance: Optional[Any] = None,
        server_script: Optional[str] = None,
        server_url: Optional[str] = None,
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

### FastMCP2MultiClient

```python
class FastMCP2MultiClient:
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
# Async functions
async def quick_load_fastmcp2_tools(
    server_instance: Optional[Any] = None,
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> List[BaseTool]

# Sync functions
def quick_load_fastmcp2_tools_sync(
    server_instance: Optional[Any] = None,
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> List[BaseTool]

def create_fastmcp2_client(
    server_instance: Optional[Any] = None,
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> FastMCP2Client
```

## Best Practices

### 1. Error Handling

Always implement proper error handling when connecting to FastMCP 2.0 servers:

```python
async def robust_tool_loading():
    try:
        tools = await quick_load_fastmcp2_tools(
            server_script="./my_server.py"
        )
        return tools
    except ImportError:
        print("FastMCP 2.0 not available")
        return []
    except Exception as e:
        print(f"Server connection failed: {e}")
        return []
```

### 2. Resource Management

Use context managers or proper cleanup for long-running applications:

```python
async def use_client_properly():
    client = FastMCP2Client(server_script="./my_server.py")
    try:
        tools = await client.get_tools()
        # Use tools...
    finally:
        # Cleanup if needed
        pass
```

### 3. Performance Optimization

Cache tools when possible to avoid repeated server connections:

```python
class ToolCache:
    def __init__(self):
        self._cache = {}
    
    async def get_tools(self, server_config):
        cache_key = str(server_config)
        if cache_key not in self._cache:
            self._cache[cache_key] = await quick_load_fastmcp2_tools(**server_config)
        return self._cache[cache_key]
```

### 4. Configuration Management

Use configuration files for server settings:

```python
import json

# config.json
{
    "servers": {
        "math": {"server_script": "./math_server.py"},
        "weather": {"server_url": "http://localhost:8001/"}
    }
}

# Load configuration
with open("config.json") as f:
    config = json.load(f)

multi_client = FastMCP2MultiClient(config["servers"])
```

## Troubleshooting

### Common Issues

1. **FastMCP 2.0 not found**
   ```
   ImportError: No module named 'fastmcp'
   ```
   Solution: Install FastMCP 2.0 with `pip install fastmcp`

2. **Server connection failed**
   ```
   ConnectionError: Failed to connect to server
   ```
   Solution: Check server script path or URL, ensure server is running

3. **Tool conversion errors**
   ```
   ToolException: Error converting tool
   ```
   Solution: Check tool schema compatibility, ensure proper type annotations

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain_mcp_adapters")

# Your code here...
```

## Examples

See the `examples/` directory for complete working examples:

- `fastmcp2_math_server.py` - Example FastMCP 2.0 server
- `fastmcp2_integration_example.py` - Complete integration examples
- `langgraph_fastmcp2_example.py` - LangGraph usage examples

## Contributing

Contributions are welcome! Please see the main README for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.