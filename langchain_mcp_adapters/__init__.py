"""LangChain MCP Adapters - FastMCP Integration."""

from .client import MultiServerMCPClient
from .tools import load_mcp_tools, convert_mcp_tool_to_langchain_tool, to_fastmcp
from .fastmcp_client import (
    FastMCPClient,
    FastMCPMultiClient,
    create_fastmcp_client,
    quick_load_fastmcp_tools,
    quick_load_fastmcp_tools_sync,
)

try:
    from .fastmcp_adapter import (
        FastMCPAdapter,
        FastMCPServerAdapter,
        load_fastmcp_tools as load_fastmcp_tools_v2,
        load_fastmcp_tools_sync,
        create_fastmcp_tool_from_function,
    )
    FASTMCP_V2_AVAILABLE = True
except ImportError:
    FASTMCP_V2_AVAILABLE = False

__all__ = [
    # Original MCP adapters
    "MultiServerMCPClient",
    "load_mcp_tools",
    "convert_mcp_tool_to_langchain_tool",
    "to_fastmcp",
    
    # FastMCP client adapters (works with any MCP server including FastMCP)
    "FastMCPClient",
    "FastMCPMultiClient", 
    "create_fastmcp_client",
    "quick_load_fastmcp_tools",
    "quick_load_fastmcp_tools_sync",
]

# Add FastMCP v2 exports if available
if FASTMCP_V2_AVAILABLE:
    __all__.extend([
        "FastMCPAdapter",
        "FastMCPServerAdapter", 
        "load_fastmcp_tools_v2",
        "load_fastmcp_tools_sync",
        "create_fastmcp_tool_from_function",
    ])