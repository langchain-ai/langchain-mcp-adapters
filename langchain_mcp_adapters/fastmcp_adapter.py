"""
FastMCP 2.0 adapter for LangChain and LangGraph integration.

This module provides adapters to seamlessly integrate FastMCP 2.0 servers
with LangChain tools and LangGraph agents.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, create_model

try:
    from fastmcp import FastMCP, Client as FastMCPClient
    from fastmcp.transports import FastMCPTransport
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    FastMCPClient = None
    FastMCPTransport = None


class FastMCPAdapter:
    """Adapter for integrating FastMCP 2.0 servers with LangChain/LangGraph."""
    
    def __init__(self, fastmcp_server: Optional[Any] = None, server_url: Optional[str] = None):
        """Initialize the FastMCP adapter.
        
        Args:
            fastmcp_server: A FastMCP server instance for direct integration
            server_url: URL to connect to a remote FastMCP server
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "FastMCP is not available. Please install it with: pip install fastmcp"
            )
        
        if fastmcp_server is None and server_url is None:
            raise ValueError("Either fastmcp_server or server_url must be provided")
        
        self.fastmcp_server = fastmcp_server
        self.server_url = server_url
        self._client: Optional[FastMCPClient] = None
    
    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[FastMCPClient]:
        """Get a FastMCP client for the server."""
        if self.fastmcp_server is not None:
            # Direct connection to FastMCP server instance
            async with FastMCPClient(self.fastmcp_server) as client:
                yield client
        elif self.server_url is not None:
            # Connect to remote FastMCP server
            async with FastMCPClient(self.server_url) as client:
                yield client
        else:
            raise ValueError("No server or URL configured")
    
    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from the FastMCP server as LangChain tools.
        
        Returns:
            List of LangChain BaseTool instances
        """
        async with self._get_client() as client:
            # List all available tools from the FastMCP server
            tools_response = await client.list_tools()
            tools = tools_response.tools if hasattr(tools_response, 'tools') else []
            
            langchain_tools = []
            for tool in tools:
                langchain_tool = await self._convert_fastmcp_tool_to_langchain(client, tool)
                langchain_tools.append(langchain_tool)
            
            return langchain_tools
    
    async def _convert_fastmcp_tool_to_langchain(self, client: FastMCPClient, tool: Any) -> BaseTool:
        """Convert a FastMCP tool to a LangChain tool.
        
        Args:
            client: FastMCP client instance
            tool: FastMCP tool definition
            
        Returns:
            LangChain BaseTool instance
        """
        # Create the tool execution function
        async def execute_tool(**kwargs: Any) -> Any:
            try:
                # Call the tool through the FastMCP client
                result = await client.call_tool(tool.name, kwargs)
                
                # Handle different result types
                if hasattr(result, 'content'):
                    if isinstance(result.content, list):
                        # Multiple content items
                        text_content = []
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                text_content.append(content_item.text)
                            elif isinstance(content_item, str):
                                text_content.append(content_item)
                        return '\n'.join(text_content) if text_content else str(result.content)
                    elif hasattr(result.content, 'text'):
                        return result.content.text
                    else:
                        return str(result.content)
                elif hasattr(result, 'text'):
                    return result.text
                else:
                    return str(result)
                    
            except Exception as e:
                raise ToolException(f"Error executing FastMCP tool '{tool.name}': {str(e)}")
        
        # Create args schema from tool input schema
        args_schema = self._create_args_schema_from_tool(tool)
        
        # Create the LangChain tool
        return StructuredTool(
            name=tool.name,
            description=tool.description or f"FastMCP tool: {tool.name}",
            args_schema=args_schema,
            coroutine=execute_tool,
            response_format="content_and_artifact"
        )
    
    def _create_args_schema_from_tool(self, tool: Any) -> type[BaseModel]:
        """Create a Pydantic model for tool arguments from FastMCP tool schema.
        
        Args:
            tool: FastMCP tool definition
            
        Returns:
            Pydantic BaseModel class for tool arguments
        """
        # Get the input schema from the tool
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            schema = tool.inputSchema
            
            # Extract properties from JSON schema
            properties = schema.get('properties', {})
            required_fields = set(schema.get('required', []))
            
            # Create field definitions for Pydantic model
            field_definitions = {}
            for field_name, field_schema in properties.items():
                field_type = self._json_schema_type_to_python_type(field_schema)
                
                # Set default value for optional fields
                if field_name in required_fields:
                    field_definitions[field_name] = (field_type, ...)
                else:
                    field_definitions[field_name] = (Optional[field_type], None)
            
            # Create the Pydantic model
            return create_model(
                f"{tool.name}Args",
                **field_definitions
            )
        else:
            # No input schema, create empty model
            return create_model(f"{tool.name}Args")
    
    def _json_schema_type_to_python_type(self, field_schema: Dict[str, Any]) -> type:
        """Convert JSON schema type to Python type.
        
        Args:
            field_schema: JSON schema field definition
            
        Returns:
            Python type
        """
        schema_type = field_schema.get('type', 'string')
        
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        
        return type_mapping.get(schema_type, str)


class FastMCPServerAdapter:
    """Adapter for running FastMCP servers and extracting tools."""
    
    def __init__(self, fastmcp_server: Any):
        """Initialize with a FastMCP server instance.
        
        Args:
            fastmcp_server: FastMCP server instance
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "FastMCP is not available. Please install it with: pip install fastmcp"
            )
        
        self.fastmcp_server = fastmcp_server
    
    async def get_tools(self) -> List[BaseTool]:
        """Get tools directly from the FastMCP server instance.
        
        Returns:
            List of LangChain BaseTool instances
        """
        adapter = FastMCPAdapter(fastmcp_server=self.fastmcp_server)
        return await adapter.get_tools()
    
    def get_tools_sync(self) -> List[BaseTool]:
        """Synchronous version of get_tools for easier integration.
        
        Returns:
            List of LangChain BaseTool instances
        """
        return asyncio.run(self.get_tools())


# Convenience functions for easy integration
async def load_fastmcp_tools(
    fastmcp_server: Optional[Any] = None,
    server_url: Optional[str] = None
) -> List[BaseTool]:
    """Load tools from a FastMCP server.
    
    Args:
        fastmcp_server: FastMCP server instance
        server_url: URL to FastMCP server
        
    Returns:
        List of LangChain tools
    """
    adapter = FastMCPAdapter(fastmcp_server=fastmcp_server, server_url=server_url)
    return await adapter.get_tools()


def load_fastmcp_tools_sync(
    fastmcp_server: Optional[Any] = None,
    server_url: Optional[str] = None
) -> List[BaseTool]:
    """Synchronously load tools from a FastMCP server.
    
    Args:
        fastmcp_server: FastMCP server instance
        server_url: URL to FastMCP server
        
    Returns:
        List of LangChain tools
    """
    return asyncio.run(load_fastmcp_tools(fastmcp_server, server_url))


def create_fastmcp_tool_from_function(func, name: Optional[str] = None, description: Optional[str] = None):
    """Create a FastMCP tool from a Python function.
    
    This is a helper function to create FastMCP tools that can then be
    converted to LangChain tools.
    
    Args:
        func: Python function to convert
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to function docstring)
        
    Returns:
        FastMCP tool that can be added to a FastMCP server
    """
    if not FASTMCP_AVAILABLE:
        raise ImportError(
            "FastMCP is not available. Please install it with: pip install fastmcp"
        )
    
    # This would be implemented based on FastMCP 2.0 API
    # The exact implementation depends on how FastMCP 2.0 creates tools
    raise NotImplementedError(
        "This function needs to be implemented based on FastMCP 2.0 API. "
        "Please refer to FastMCP documentation for tool creation."
    )