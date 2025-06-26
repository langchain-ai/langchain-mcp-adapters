"""
FastMCP 2.0 Client Integration for LangChain and LangGraph.

This module provides a simplified client interface for connecting to FastMCP 2.0
servers and automatically converting their tools for use with LangGraph.

FastMCP 2.0 is the independent FastMCP SDK with advanced features like
proxying, composition, and deployment capabilities.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import concurrent.futures

from langchain_core.tools import BaseTool

try:
    # FastMCP 2.0 SDK imports
    import fastmcp
    from fastmcp import FastMCP
    from fastmcp.client import Client as FastMCPClient
    FASTMCP_V2_AVAILABLE = True
except ImportError:
    FASTMCP_V2_AVAILABLE = False
    fastmcp = None
    FastMCP = None
    FastMCPClient = None

from .fastmcp_v2_adapter import FastMCPv2Adapter, load_fastmcp_v2_tools


class FastMCPv2Client:
    """Simplified client for FastMCP 2.0 servers with LangGraph integration."""
    
    def __init__(
        self,
        server_instance: Optional[Any] = None,
        server_url: Optional[str] = None,
        server_script: Optional[str] = None,
        **kwargs
    ):
        """Initialize FastMCP 2.0 client.
        
        Args:
            server_instance: Direct FastMCP 2.0 server instance
            server_url: URL to FastMCP 2.0 server (for HTTP connections)
            server_script: Path to FastMCP 2.0 server script (for stdio)
            **kwargs: Additional connection parameters
        """
        if not FASTMCP_V2_AVAILABLE:
            raise ImportError(
                "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
            )
        
        if not any([server_instance, server_url, server_script]):
            raise ValueError("Either server_instance, server_url, or server_script must be provided")
        
        self.server_instance = server_instance
        self.server_url = server_url
        self.server_script = server_script
        self.kwargs = kwargs
        
        # Determine connection type
        if server_instance:
            self.connection_type = "direct"
        elif server_url:
            self.connection_type = "http"
        elif server_script:
            self.connection_type = "script"
    
    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from the FastMCP 2.0 server as LangChain tools.
        
        Returns:
            List of LangChain BaseTool instances ready for LangGraph
        """
        adapter = FastMCPv2Adapter(
            fastmcp_server=self.server_instance,
            server_url=self.server_url,
            server_script=self.server_script
        )
        return await adapter.get_tools()
    
    async def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            LangChain BaseTool instance or None if not found
        """
        tools = await self.get_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def list_tool_names(self) -> List[str]:
        """Get list of available tool names.
        
        Returns:
            List of tool names
        """
        tools = await self.get_tools()
        return [tool.name for tool in tools]
    
    def get_tools_sync(self) -> List[BaseTool]:
        """Synchronous version of get_tools.
        
        Returns:
            List of LangChain BaseTool instances
        """
        return self._run_async_safely(self.get_tools())
    
    def get_tool_by_name_sync(self, tool_name: str) -> Optional[BaseTool]:
        """Synchronous version of get_tool_by_name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            LangChain BaseTool instance or None if not found
        """
        return self._run_async_safely(self.get_tool_by_name(tool_name))
    
    def list_tool_names_sync(self) -> List[str]:
        """Synchronous version of list_tool_names.
        
        Returns:
            List of tool names
        """
        return self._run_async_safely(self.list_tool_names())
    
    def _run_async_safely(self, coro):
        """Run async function safely, handling event loop conflicts."""
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, use ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(coro)


class FastMCPv2MultiClient:
    """Client for managing multiple FastMCP 2.0 servers."""
    
    def __init__(self, servers: Dict[str, Dict[str, Any]]):
        """Initialize with multiple FastMCP 2.0 server configurations.
        
        Args:
            servers: Dictionary mapping server names to configuration dicts
                    Each config dict should contain parameters for FastMCPv2Client
        """
        if not FASTMCP_V2_AVAILABLE:
            raise ImportError(
                "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
            )
        
        self.servers = servers
        self.clients = {}
        
        # Initialize clients
        for name, config in servers.items():
            self.clients[name] = FastMCPv2Client(**config)
    
    async def get_all_tools(self) -> Dict[str, List[BaseTool]]:
        """Get tools from all servers.
        
        Returns:
            Dictionary mapping server names to their tools
        """
        all_tools = {}
        for name, client in self.clients.items():
            try:
                all_tools[name] = await client.get_tools()
            except Exception as e:
                print(f"Warning: Failed to get tools from server '{name}': {e}")
                all_tools[name] = []
        return all_tools
    
    async def get_tools_flat(self) -> List[BaseTool]:
        """Get all tools from all servers as a flat list.
        
        Returns:
            List of all tools from all servers
        """
        all_tools = await self.get_all_tools()
        flat_tools = []
        for tools in all_tools.values():
            flat_tools.extend(tools)
        return flat_tools
    
    async def get_tools_from_server(self, server_name: str) -> List[BaseTool]:
        """Get tools from a specific server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            List of tools from the specified server
        """
        if server_name not in self.clients:
            raise ValueError(f"Server '{server_name}' not found")
        
        return await self.clients[server_name].get_tools()
    
    def get_all_tools_sync(self) -> Dict[str, List[BaseTool]]:
        """Synchronous version of get_all_tools."""
        return asyncio.run(self.get_all_tools())
    
    def get_tools_flat_sync(self) -> List[BaseTool]:
        """Synchronous version of get_tools_flat."""
        return asyncio.run(self.get_tools_flat())
    
    def get_tools_from_server_sync(self, server_name: str) -> List[BaseTool]:
        """Synchronous version of get_tools_from_server."""
        return asyncio.run(self.get_tools_from_server(server_name))


class FastMCPv2ServerManager:
    """Manager for FastMCP 2.0 server instances."""
    
    def __init__(self):
        """Initialize the server manager."""
        if not FASTMCP_V2_AVAILABLE:
            raise ImportError(
                "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
            )
        
        self.servers: Dict[str, Any] = {}
    
    def add_server(self, name: str, server: Any) -> None:
        """Add a FastMCP 2.0 server instance.
        
        Args:
            name: Name for the server
            server: FastMCP 2.0 server instance
        """
        self.servers[name] = server
    
    def remove_server(self, name: str) -> None:
        """Remove a server.
        
        Args:
            name: Name of the server to remove
        """
        if name in self.servers:
            del self.servers[name]
    
    async def get_all_tools(self) -> Dict[str, List[BaseTool]]:
        """Get tools from all managed servers.
        
        Returns:
            Dictionary mapping server names to their tools
        """
        all_tools = {}
        for name, server in self.servers.items():
            try:
                adapter = FastMCPv2Adapter(fastmcp_server=server)
                all_tools[name] = await adapter.get_tools()
            except Exception as e:
                print(f"Warning: Failed to get tools from server '{name}': {e}")
                all_tools[name] = []
        return all_tools
    
    async def get_tools_from_server(self, server_name: str) -> List[BaseTool]:
        """Get tools from a specific managed server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            List of tools from the specified server
        """
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        adapter = FastMCPv2Adapter(fastmcp_server=self.servers[server_name])
        return await adapter.get_tools()
    
    def create_client_for_server(self, server_name: str) -> FastMCPv2Client:
        """Create a client for a specific managed server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            FastMCPv2Client instance
        """
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        return FastMCPv2Client(server_instance=self.servers[server_name])


# Convenience functions for FastMCP 2.0 integration
def create_fastmcp_v2_client(
    server_instance: Optional[Any] = None,
    server_url: Optional[str] = None,
    server_script: Optional[str] = None,
    **kwargs
) -> FastMCPv2Client:
    """Create a FastMCP 2.0 client.
    
    Args:
        server_instance: FastMCP 2.0 server instance
        server_url: URL to FastMCP 2.0 server
        server_script: Path to FastMCP 2.0 server script
        **kwargs: Additional connection parameters
        
    Returns:
        FastMCPv2Client instance
    """
    return FastMCPv2Client(
        server_instance=server_instance,
        server_url=server_url,
        server_script=server_script,
        **kwargs
    )


async def quick_load_fastmcp_v2_tools(
    server_instance: Optional[Any] = None,
    server_url: Optional[str] = None,
    server_script: Optional[str] = None,
    **kwargs
) -> List[BaseTool]:
    """Quickly load tools from a FastMCP 2.0 server.
    
    Args:
        server_instance: FastMCP 2.0 server instance
        server_url: URL to FastMCP 2.0 server
        server_script: Path to FastMCP 2.0 server script
        **kwargs: Additional connection parameters
        
    Returns:
        List of LangChain tools ready for LangGraph
    """
    return await load_fastmcp_v2_tools(
        fastmcp_server=server_instance,
        server_url=server_url,
        server_script=server_script
    )


def quick_load_fastmcp_v2_tools_sync(
    server_instance: Optional[Any] = None,
    server_url: Optional[str] = None,
    server_script: Optional[str] = None,
    **kwargs
) -> List[BaseTool]:
    """Synchronously load tools from a FastMCP 2.0 server.
    
    Args:
        server_instance: FastMCP 2.0 server instance
        server_url: URL to FastMCP 2.0 server
        server_script: Path to FastMCP 2.0 server script
        **kwargs: Additional connection parameters
        
    Returns:
        List of LangChain tools ready for LangGraph
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                quick_load_fastmcp_v2_tools(
                    server_instance=server_instance,
                    server_url=server_url,
                    server_script=server_script,
                    **kwargs
                )
            )
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(quick_load_fastmcp_v2_tools(
            server_instance=server_instance,
            server_url=server_url,
            server_script=server_script,
            **kwargs
        ))