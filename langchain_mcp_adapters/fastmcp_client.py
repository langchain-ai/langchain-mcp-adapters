"""
FastMCP Client adapter for seamless LangGraph integration.

This module provides a simplified client for connecting to FastMCP servers
and automatically converting their tools for use with LangGraph.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.tools import BaseTool
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp import StdioServerParameters

from .tools import load_mcp_tools, convert_mcp_tool_to_langchain_tool


class FastMCPClient:
    """Simplified client for FastMCP servers with LangGraph integration."""
    
    def __init__(
        self,
        server_script: Optional[str] = None,
        server_url: Optional[str] = None,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize FastMCP client.
        
        Args:
            server_script: Path to FastMCP server script (for stdio)
            server_url: URL to FastMCP server (for HTTP)
            server_command: Command to run FastMCP server
            server_args: Arguments for server command
            **kwargs: Additional connection parameters
        """
        self.server_script = server_script
        self.server_url = server_url
        self.server_command = server_command or "python"
        self.server_args = server_args or []
        self.kwargs = kwargs
        
        # Determine connection type
        if server_url:
            self.connection_type = "http"
        elif server_script:
            self.connection_type = "stdio"
            if not server_args:
                self.server_args = [server_script]
        else:
            raise ValueError("Either server_script or server_url must be provided")
    
    @asynccontextmanager
    async def _get_session(self) -> AsyncIterator[ClientSession]:
        """Get an MCP client session."""
        if self.connection_type == "stdio":
            # Connect via stdio
            server_params = StdioServerParameters(
                command=self.server_command,
                args=self.server_args,
                **self.kwargs
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
                    
        elif self.connection_type == "http":
            # Connect via HTTP
            async with streamablehttp_client(self.server_url, **self.kwargs) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")
    
    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from the FastMCP server as LangChain tools.
        
        Returns:
            List of LangChain BaseTool instances ready for LangGraph
        """
        async with self._get_session() as session:
            return await load_mcp_tools(session)
    
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
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.get_tools())
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.get_tools())
    
    def get_tool_by_name_sync(self, tool_name: str) -> Optional[BaseTool]:
        """Synchronous version of get_tool_by_name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            LangChain BaseTool instance or None if not found
        """
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.get_tool_by_name(tool_name))
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.get_tool_by_name(tool_name))
    
    def list_tool_names_sync(self) -> List[str]:
        """Synchronous version of list_tool_names.
        
        Returns:
            List of tool names
        """
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.list_tool_names())
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.list_tool_names())


class FastMCPMultiClient:
    """Client for managing multiple FastMCP servers."""
    
    def __init__(self, servers: Dict[str, Dict[str, Any]]):
        """Initialize with multiple server configurations.
        
        Args:
            servers: Dictionary mapping server names to configuration dicts
                    Each config dict should contain parameters for FastMCPClient
        """
        self.servers = servers
        self.clients = {}
        
        # Initialize clients
        for name, config in servers.items():
            self.clients[name] = FastMCPClient(**config)
    
    async def get_all_tools(self) -> Dict[str, List[BaseTool]]:
        """Get tools from all servers.
        
        Returns:
            Dictionary mapping server names to their tools
        """
        all_tools = {}
        for name, client in self.clients.items():
            all_tools[name] = await client.get_tools()
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


# Convenience functions
def create_fastmcp_client(
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> FastMCPClient:
    """Create a FastMCP client.
    
    Args:
        server_script: Path to FastMCP server script
        server_url: URL to FastMCP server
        **kwargs: Additional connection parameters
        
    Returns:
        FastMCPClient instance
    """
    return FastMCPClient(
        server_script=server_script,
        server_url=server_url,
        **kwargs
    )


async def quick_load_fastmcp_tools(
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> List[BaseTool]:
    """Quickly load tools from a FastMCP server.
    
    Args:
        server_script: Path to FastMCP server script
        server_url: URL to FastMCP server
        **kwargs: Additional connection parameters
        
    Returns:
        List of LangChain tools ready for LangGraph
    """
    client = create_fastmcp_client(
        server_script=server_script,
        server_url=server_url,
        **kwargs
    )
    return await client.get_tools()


def quick_load_fastmcp_tools_sync(
    server_script: Optional[str] = None,
    server_url: Optional[str] = None,
    **kwargs
) -> List[BaseTool]:
    """Synchronously load tools from a FastMCP server.
    
    Args:
        server_script: Path to FastMCP server script
        server_url: URL to FastMCP server
        **kwargs: Additional connection parameters
        
    Returns:
        List of LangChain tools ready for LangGraph
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're in an event loop, we can't use asyncio.run()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run, 
                quick_load_fastmcp_tools(
                    server_script=server_script,
                    server_url=server_url,
                    **kwargs
                )
            )
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(quick_load_fastmcp_tools(
            server_script=server_script,
            server_url=server_url,
            **kwargs
        ))