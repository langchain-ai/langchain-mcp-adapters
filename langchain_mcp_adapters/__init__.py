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

# FastMCP 2.0 SDK integration (jlowin/fastmcp)
try:
    from .fastmcp2_client import (
        FastMCP2Client,
        FastMCP2MultiClient,
        create_fastmcp2_client,
        quick_load_fastmcp2_tools,
        quick_load_fastmcp2_tools_sync,
    )
    from .fastmcp_adapter import (
        FastMCP2Adapter,
        FastMCP2ServerAdapter,
        load_fastmcp2_tools,
        load_fastmcp2_tools_sync,
        create_fastmcp2_tool_from_function,
        # Backward compatibility aliases
        FastMCPAdapter,
        FastMCPServerAdapter,
        load_fastmcp_tools as load_fastmcp_tools_v2,
        load_fastmcp_tools_sync,
        create_fastmcp_tool_from_function,
    )
    FASTMCP2_AVAILABLE = True
except ImportError:
    FASTMCP2_AVAILABLE = False

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

# Add FastMCP 2.0 SDK exports if available
if FASTMCP2_AVAILABLE:
    __all__.extend([
        # FastMCP 2.0 SDK specific
        "FastMCP2Client",
        "FastMCP2MultiClient",
        "create_fastmcp2_client",
        "quick_load_fastmcp2_tools",
        "quick_load_fastmcp2_tools_sync",
        "FastMCP2Adapter",
        "FastMCP2ServerAdapter",
        "load_fastmcp2_tools",
        "load_fastmcp2_tools_sync",
        "create_fastmcp2_tool_from_function",
        
        # Backward compatibility aliases
        "FastMCPAdapter",
        "FastMCPServerAdapter", 
        "load_fastmcp_tools_v2",
        "load_fastmcp_tools_sync",
        "create_fastmcp_tool_from_function",
        
        # Availability flag
        "FASTMCP2_AVAILABLE",
    ])