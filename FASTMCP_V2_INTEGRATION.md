# FastMCP 2.0 SDK Integration Guide

This guide explains how to use the **FastMCP 2.0 SDK** integration features in `langchain-mcp-adapters` to seamlessly connect FastMCP 2.0 servers with LangGraph agents.

## 🎯 What is FastMCP 2.0?

FastMCP 2.0 is the **independent FastMCP SDK** (https://github.com/jlowin/fastmcp) - a powerful, Pythonic way to build MCP servers and clients with advanced features like:

- 🚀 **High-level API**: Simplified server and client creation
- 🔧 **Direct Integration**: Work directly with server instances
- 🌐 **Advanced Transports**: HTTP, SSE, and in-memory connections
- 🏗️ **Composition**: Proxy servers and server composition
- 📦 **Deployment**: Production-ready deployment options

> **Note**: This is different from the `fastmcp` module in the official MCP Python SDK. FastMCP 2.0 is a separate, more advanced SDK.

## Installation

```bash
# Install the base adapter
pip install langchain-mcp-adapters

# Install FastMCP 2.0 SDK
pip install fastmcp
```

## Quick Start

### 1. Direct Server Instance Integration

The most powerful feature of FastMCP 2.0 integration is working directly with server instances:

```python
from fastmcp import FastMCP
from langchain_mcp_adapters import extract_tools_from_fastmcp_v2_server
from langgraph.prebuilt import create_react_agent

# Create FastMCP 2.0 server
server = FastMCP("My Server")

@server.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@server.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Extract tools directly from server instance
tools = extract_tools_from_fastmcp_v2_server(server)

# Create LangGraph agent
agent = create_react_agent("openai:gpt-4", tools)
```

### 2. Quick Tool Loading

```python
from langchain_mcp_adapters import quick_load_fastmcp_v2_tools

# Load tools from FastMCP 2.0 server script
tools = await quick_load_fastmcp_v2_tools(
    server_script="./my_fastmcp_server.py"
)

# Create agent
agent = create_react_agent("openai:gpt-4", tools)
```

### 3. Client-Based Integration

```python
from langchain_mcp_adapters import FastMCPv2Client

# Create client for FastMCP 2.0 server
client = FastMCPv2Client(server_instance=my_server)

# Get tools
tools = await client.get_tools()

# Create agent
agent = create_react_agent("openai:gpt-4", tools)
```

## Advanced Usage

### Multiple Server Management

```python
from langchain_mcp_adapters import FastMCPv2MultiClient

# Configure multiple FastMCP 2.0 servers
servers = {
    "math": {
        "server_instance": math_server  # Direct server instance
    },
    "text": {
        "server_url": "http://localhost:8000"  # Remote server
    },
    "data": {
        "server_script": "./data_server.py"  # Script-based server
    }
}

# Create multi-client
multi_client = FastMCPv2MultiClient(servers)

# Get all tools from all servers
all_tools = await multi_client.get_tools_flat()

# Create agent with all tools
agent = create_react_agent("openai:gpt-4", all_tools)
```

### Server Manager

```python
from langchain_mcp_adapters import FastMCPv2ServerManager
from fastmcp import FastMCP

# Create server manager
manager = FastMCPv2ServerManager()

# Create and add servers
math_server = FastMCP("Math")
@math_server.tool()
def add(a: float, b: float) -> float:
    return a + b

text_server = FastMCP("Text")
@text_server.tool()
def uppercase(text: str) -> str:
    return text.upper()

# Add to manager
manager.add_server("math", math_server)
manager.add_server("text", text_server)

# Get tools from all managed servers
all_tools = await manager.get_all_tools()

# Get tools from specific server
math_tools = await manager.get_tools_from_server("math")

# Create client for specific server
math_client = manager.create_client_for_server("math")
```

### Advanced FastMCP 2.0 Features

```python
from fastmcp import FastMCP
from langchain_mcp_adapters import FastMCPv2Adapter

# Create server with advanced features
server = FastMCP("Advanced Server")

# Tools
@server.tool()
async def async_calculation(x: float, y: float) -> float:
    """Async tool example."""
    await asyncio.sleep(0.1)  # Simulate async work
    return x * y

# Resources
@server.resource("data://constants")
def get_constants():
    """Get mathematical constants."""
    return {"pi": 3.14159, "e": 2.71828}

# Prompts
@server.prompt()
def help_prompt(topic: str) -> str:
    """Generate help prompt."""
    return f"Please explain {topic} in detail."

# Extract tools using adapter
adapter = FastMCPv2Adapter(fastmcp_server=server)
tools = await adapter.get_tools()
```

## API Reference

### FastMCPv2Adapter

```python
class FastMCPv2Adapter:
    def __init__(
        self,
        fastmcp_server: Optional[Any] = None,
        server_url: Optional[str] = None,
        server_script: Optional[str] = None
    )
    
    async def get_tools(self) -> List[BaseTool]
```

### FastMCPv2Client

```python
class FastMCPv2Client:
    def __init__(
        self,
        server_instance: Optional[Any] = None,
        server_url: Optional[str] = None,
        server_script: Optional[str] = None,
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

### FastMCPv2MultiClient

```python
class FastMCPv2MultiClient:
    def __init__(self, servers: Dict[str, Dict[str, Any]])
    
    async def get_all_tools(self) -> Dict[str, List[BaseTool]]
    async def get_tools_flat(self) -> List[BaseTool]
    async def get_tools_from_server(self, server_name: str) -> List[BaseTool]
```

### FastMCPv2ServerManager

```python
class FastMCPv2ServerManager:
    def add_server(self, name: str, server: Any) -> None
    def remove_server(self, name: str) -> None
    async def get_all_tools(self) -> Dict[str, List[BaseTool]]
    async def get_tools_from_server(self, server_name: str) -> List[BaseTool]
    def create_client_for_server(self, server_name: str) -> FastMCPv2Client
```

### Convenience Functions

```python
# Tool loading
async def load_fastmcp_v2_tools(
    fastmcp_server: Optional[Any] = None,
    server_url: Optional[str] = None,
    server_script: Optional[str] = None
) -> List[BaseTool]

def load_fastmcp_v2_tools_sync(...) -> List[BaseTool]

async def quick_load_fastmcp_v2_tools(...) -> List[BaseTool]
def quick_load_fastmcp_v2_tools_sync(...) -> List[BaseTool]

# Direct extraction
def extract_tools_from_fastmcp_v2_server(server: Any) -> List[BaseTool]

# Client creation
def create_fastmcp_v2_client(...) -> FastMCPv2Client
```

## Creating FastMCP 2.0 Servers

### Basic Server

```python
from fastmcp import FastMCP

# Create server
server = FastMCP("My Server")

@server.tool()
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

@server.tool()
async def async_task(data: str) -> str:
    """Async tool example."""
    await asyncio.sleep(0.1)
    return f"Processed: {data}"

# Run server (for script-based usage)
if __name__ == "__main__":
    server.run(transport="stdio")
```

### Server with Resources and Prompts

```python
from fastmcp import FastMCP

server = FastMCP("Advanced Server")

@server.tool()
def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    # Implementation here
    pass

@server.resource("config://settings")
def get_settings():
    """Get server settings."""
    return {"version": "1.0", "mode": "production"}

@server.prompt()
def help_prompt(topic: str) -> str:
    """Generate help for a topic."""
    return f"Help for {topic}: ..."

# Use with LangGraph
from langchain_mcp_adapters import extract_tools_from_fastmcp_v2_server

tools = extract_tools_from_fastmcp_v2_server(server)
```

## Integration Patterns

### 1. Development Pattern (Direct Integration)

```python
# Best for development and testing
from fastmcp import FastMCP
from langchain_mcp_adapters import extract_tools_from_fastmcp_v2_server

server = FastMCP("Dev Server")

@server.tool()
def my_tool(x: int) -> int:
    return x * 2

tools = extract_tools_from_fastmcp_v2_server(server)
agent = create_react_agent("openai:gpt-4", tools)
```

### 2. Production Pattern (Client-Based)

```python
# Best for production deployments
from langchain_mcp_adapters import FastMCPv2Client

client = FastMCPv2Client(server_url="https://my-fastmcp-server.com")
tools = await client.get_tools()
agent = create_react_agent("openai:gpt-4", tools)
```

### 3. Microservices Pattern (Multi-Client)

```python
# Best for microservices architecture
from langchain_mcp_adapters import FastMCPv2MultiClient

servers = {
    "auth": {"server_url": "https://auth-service.com"},
    "data": {"server_url": "https://data-service.com"},
    "ml": {"server_url": "https://ml-service.com"}
}

multi_client = FastMCPv2MultiClient(servers)
tools = await multi_client.get_tools_flat()
agent = create_react_agent("openai:gpt-4", tools)
```

## Error Handling

```python
from langchain_mcp_adapters import FastMCPv2Client

try:
    client = FastMCPv2Client(server_url="http://localhost:8000")
    tools = await client.get_tools()
except Exception as e:
    print(f"Error connecting to FastMCP 2.0 server: {e}")
    # Fallback to default tools
    tools = []

if tools:
    agent = create_react_agent("openai:gpt-4", tools)
else:
    print("No tools available, using base agent")
    agent = create_react_agent("openai:gpt-4", [])
```

## Best Practices

### 1. Use Direct Integration for Development

```python
# Development: Direct server integration
server = FastMCP("Dev Server")
# ... add tools ...
tools = extract_tools_from_fastmcp_v2_server(server)
```

### 2. Use Clients for Production

```python
# Production: Client-based integration
client = FastMCPv2Client(server_url=os.getenv("FASTMCP_SERVER_URL"))
tools = await client.get_tools()
```

### 3. Implement Proper Error Handling

```python
async def load_tools_safely():
    try:
        return await quick_load_fastmcp_v2_tools(server_url=server_url)
    except Exception as e:
        logger.error(f"Failed to load FastMCP 2.0 tools: {e}")
        return []  # Return empty list as fallback
```

### 4. Use Server Manager for Complex Setups

```python
# Complex setups: Use server manager
manager = FastMCPv2ServerManager()
manager.add_server("primary", primary_server)
manager.add_server("secondary", secondary_server)

all_tools = await manager.get_all_tools()
```

## Comparison: FastMCP 2.0 vs Standard MCP

| Feature | Standard MCP | FastMCP 2.0 |
|---------|-------------|-------------|
| **Server Creation** | Manual protocol handling | High-level decorators |
| **Direct Integration** | Not available | ✅ Server instances |
| **Tool Extraction** | Protocol-based | Direct function access |
| **Advanced Features** | Basic | Proxying, composition |
| **Development Experience** | Complex | Simplified |
| **Production Deployment** | Manual setup | Built-in options |

## Migration from Standard MCP

```python
# Old way (Standard MCP)
from langchain_mcp_adapters import MultiServerMCPClient

client = MultiServerMCPClient({
    "math": {
        "command": "python",
        "args": ["./math_server.py"],
        "transport": "stdio"
    }
})
tools = await client.get_tools()

# New way (FastMCP 2.0)
from langchain_mcp_adapters import quick_load_fastmcp_v2_tools

tools = await quick_load_fastmcp_v2_tools(
    server_script="./math_server.py"
)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure FastMCP 2.0 is installed
   ```bash
   pip install fastmcp
   ```

2. **Server Not Found**: Check server instance/URL/script path
   ```python
   # Verify server is accessible
   client = FastMCPv2Client(server_instance=server)
   tools = await client.get_tools()
   ```

3. **Tool Conversion Error**: Ensure tools have proper type hints
   ```python
   @server.tool()
   def my_tool(x: int, y: str) -> str:  # ✅ Good: type hints
       return f"{x}: {y}"
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed information about tool extraction
tools = extract_tools_from_fastmcp_v2_server(server)
```

## Examples

See the following example files:
- `examples/fastmcp_v2_math_server.py` - FastMCP 2.0 server example
- `examples/fastmcp_v2_integration_example.py` - Complete integration examples

## Next Steps

1. **Install FastMCP 2.0**: `pip install fastmcp`
2. **Create your first server**: Use the examples as templates
3. **Integrate with LangGraph**: Use the adapters to connect
4. **Explore advanced features**: Proxying, composition, deployment
5. **Build amazing AI applications**: Combine FastMCP 2.0 power with LangGraph flexibility

FastMCP 2.0 integration provides the most powerful and flexible way to connect MCP servers with LangGraph, offering both simplicity for development and robustness for production deployments.