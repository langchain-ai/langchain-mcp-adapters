"""
FastMCP 2.0 SDK Integration for LangChain and LangGraph.

This module provides seamless integration with the independent FastMCP 2.0 SDK
(https://github.com/jlowin/fastmcp) for LangChain and LangGraph applications.

FastMCP 2.0 is a powerful, Pythonic way to build MCP servers and clients
with advanced features like proxying, composition, and deployment.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Callable
import inspect

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, create_model

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


class FastMCPv2Adapter:
    """Adapter for integrating FastMCP 2.0 servers with LangChain/LangGraph."""
    
    def __init__(
        self, 
        fastmcp_server: Optional[Any] = None, 
        server_url: Optional[str] = None,
        server_script: Optional[str] = None
    ):
        """Initialize the FastMCP 2.0 adapter.
        
        Args:
            fastmcp_server: A FastMCP 2.0 server instance for direct integration
            server_url: URL to connect to a remote FastMCP 2.0 server
            server_script: Path to a FastMCP 2.0 server script
        """
        if not FASTMCP_V2_AVAILABLE:
            raise ImportError(
                "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
            )
        
        if not any([fastmcp_server, server_url, server_script]):
            raise ValueError("Either fastmcp_server, server_url, or server_script must be provided")
        
        self.fastmcp_server = fastmcp_server
        self.server_url = server_url
        self.server_script = server_script
    
    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from the FastMCP 2.0 server as LangChain tools.
        
        Returns:
            List of LangChain BaseTool instances
        """
        if self.fastmcp_server is not None:
            # Direct server instance integration
            return await self._get_tools_from_server_instance()
        elif self.server_url is not None:
            # Remote server integration
            return await self._get_tools_from_url()
        elif self.server_script is not None:
            # Script-based server integration
            return await self._get_tools_from_script()
        else:
            raise ValueError("No valid server configuration provided")
    
    async def _get_tools_from_server_instance(self) -> List[BaseTool]:
        """Get tools directly from a FastMCP 2.0 server instance."""
        if not hasattr(self.fastmcp_server, '_tools'):
            # Try to access tools through different possible attributes
            tools_attr = None
            for attr in ['_tools', 'tools', '_registered_tools', 'registered_tools']:
                if hasattr(self.fastmcp_server, attr):
                    tools_attr = getattr(self.fastmcp_server, attr)
                    break
            
            if tools_attr is None:
                raise AttributeError("Cannot find tools in FastMCP server instance")
        else:
            tools_attr = self.fastmcp_server._tools
        
        langchain_tools = []
        
        # Convert FastMCP tools to LangChain tools
        if isinstance(tools_attr, dict):
            for tool_name, tool_func in tools_attr.items():
                langchain_tool = self._convert_fastmcp_function_to_langchain_tool(
                    tool_name, tool_func
                )
                langchain_tools.append(langchain_tool)
        elif isinstance(tools_attr, list):
            for tool in tools_attr:
                if hasattr(tool, 'name') and hasattr(tool, 'fn'):
                    langchain_tool = self._convert_fastmcp_function_to_langchain_tool(
                        tool.name, tool.fn
                    )
                    langchain_tools.append(langchain_tool)
        
        return langchain_tools
    
    async def _get_tools_from_url(self) -> List[BaseTool]:
        """Get tools from a remote FastMCP 2.0 server."""
        async with FastMCPClient(self.server_url) as client:
            # List tools from the remote server
            tools_response = await client.list_tools()
            
            langchain_tools = []
            for tool in tools_response.tools:
                langchain_tool = await self._convert_remote_tool_to_langchain(client, tool)
                langchain_tools.append(langchain_tool)
            
            return langchain_tools
    
    async def _get_tools_from_script(self) -> List[BaseTool]:
        """Get tools from a FastMCP 2.0 server script."""
        # This would require running the script and connecting to it
        # For now, we'll use the standard MCP client approach
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from ..tools import load_mcp_tools
        
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await load_mcp_tools(session)
    
    def _convert_fastmcp_function_to_langchain_tool(
        self, 
        tool_name: str, 
        tool_func: Callable
    ) -> BaseTool:
        """Convert a FastMCP 2.0 function to a LangChain tool."""
        
        # Get function signature and docstring
        sig = inspect.signature(tool_func)
        doc = inspect.getdoc(tool_func) or f"FastMCP tool: {tool_name}"
        
        # Create args schema from function signature
        args_schema = self._create_args_schema_from_signature(tool_name, sig)
        
        # Create the tool execution function
        async def execute_tool(**kwargs: Any) -> Any:
            try:
                # Call the original function
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**kwargs)
                else:
                    result = tool_func(**kwargs)
                
                # Convert result to string if needed
                if isinstance(result, (dict, list)):
                    import json
                    return json.dumps(result, indent=2)
                else:
                    return str(result)
                    
            except Exception as e:
                raise ToolException(f"Error executing FastMCP tool '{tool_name}': {str(e)}")
        
        return StructuredTool(
            name=tool_name,
            description=doc,
            args_schema=args_schema,
            coroutine=execute_tool,
            response_format="content_and_artifact"
        )
    
    async def _convert_remote_tool_to_langchain(
        self, 
        client: FastMCPClient, 
        tool: Any
    ) -> BaseTool:
        """Convert a remote FastMCP 2.0 tool to a LangChain tool."""
        
        # Create the tool execution function
        async def execute_tool(**kwargs: Any) -> Any:
            try:
                result = await client.call_tool(tool.name, kwargs)
                
                # Handle different result types
                if hasattr(result, 'content'):
                    if isinstance(result.content, list):
                        text_content = []
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                text_content.append(content_item.text)
                            else:
                                text_content.append(str(content_item))
                        return '\n'.join(text_content)
                    else:
                        return str(result.content)
                else:
                    return str(result)
                    
            except Exception as e:
                raise ToolException(f"Error executing remote FastMCP tool '{tool.name}': {str(e)}")
        
        # Create args schema from tool input schema
        args_schema = self._create_args_schema_from_tool_schema(tool)
        
        return StructuredTool(
            name=tool.name,
            description=tool.description or f"FastMCP tool: {tool.name}",
            args_schema=args_schema,
            coroutine=execute_tool,
            response_format="content_and_artifact"
        )
    
    def _create_args_schema_from_signature(
        self, 
        tool_name: str, 
        sig: inspect.Signature
    ) -> type[BaseModel]:
        """Create a Pydantic model from function signature."""
        
        field_definitions = {}
        
        for param_name, param in sig.parameters.items():
            # Skip self and context parameters
            if param_name in ['self', 'ctx', 'context']:
                continue
            
            # Determine parameter type
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            
            # Handle default values
            if param.default != inspect.Parameter.empty:
                field_definitions[param_name] = (Optional[param_type], param.default)
            else:
                field_definitions[param_name] = (param_type, ...)
        
        # Create the Pydantic model
        return create_model(f"{tool_name}Args", **field_definitions)
    
    def _create_args_schema_from_tool_schema(self, tool: Any) -> type[BaseModel]:
        """Create a Pydantic model from tool schema."""
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
            return create_model(f"{tool.name}Args", **field_definitions)
        else:
            # No input schema, create empty model
            return create_model(f"{tool.name}Args")
    
    def _json_schema_type_to_python_type(self, field_schema: Dict[str, Any]) -> type:
        """Convert JSON schema type to Python type."""
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


class FastMCPv2ServerAdapter:
    """Adapter for working with FastMCP 2.0 server instances."""
    
    def __init__(self, fastmcp_server: Any):
        """Initialize with a FastMCP 2.0 server instance.
        
        Args:
            fastmcp_server: FastMCP 2.0 server instance
        """
        if not FASTMCP_V2_AVAILABLE:
            raise ImportError(
                "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
            )
        
        self.fastmcp_server = fastmcp_server
    
    async def get_tools(self) -> List[BaseTool]:
        """Get tools directly from the FastMCP 2.0 server instance.
        
        Returns:
            List of LangChain BaseTool instances
        """
        adapter = FastMCPv2Adapter(fastmcp_server=self.fastmcp_server)
        return await adapter.get_tools()
    
    def get_tools_sync(self) -> List[BaseTool]:
        """Synchronous version of get_tools.
        
        Returns:
            List of LangChain BaseTool instances
        """
        return asyncio.run(self.get_tools())


# Convenience functions for FastMCP 2.0 integration
async def load_fastmcp_v2_tools(
    fastmcp_server: Optional[Any] = None,
    server_url: Optional[str] = None,
    server_script: Optional[str] = None
) -> List[BaseTool]:
    """Load tools from a FastMCP 2.0 server.
    
    Args:
        fastmcp_server: FastMCP 2.0 server instance
        server_url: URL to FastMCP 2.0 server
        server_script: Path to FastMCP 2.0 server script
        
    Returns:
        List of LangChain tools
    """
    adapter = FastMCPv2Adapter(
        fastmcp_server=fastmcp_server,
        server_url=server_url,
        server_script=server_script
    )
    return await adapter.get_tools()


def load_fastmcp_v2_tools_sync(
    fastmcp_server: Optional[Any] = None,
    server_url: Optional[str] = None,
    server_script: Optional[str] = None
) -> List[BaseTool]:
    """Synchronously load tools from a FastMCP 2.0 server.
    
    Args:
        fastmcp_server: FastMCP 2.0 server instance
        server_url: URL to FastMCP 2.0 server
        server_script: Path to FastMCP 2.0 server script
        
    Returns:
        List of LangChain tools
    """
    return asyncio.run(load_fastmcp_v2_tools(
        fastmcp_server=fastmcp_server,
        server_url=server_url,
        server_script=server_script
    ))


def create_fastmcp_v2_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Any:
    """Create a FastMCP 2.0 tool from a Python function.
    
    This is a helper function to create FastMCP 2.0 tools that can then be
    converted to LangChain tools.
    
    Args:
        func: Python function to convert
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to function docstring)
        
    Returns:
        FastMCP 2.0 tool that can be added to a FastMCP server
    """
    if not FASTMCP_V2_AVAILABLE:
        raise ImportError(
            "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
        )
    
    # Create a FastMCP server and add the tool
    server = FastMCP(name or "ToolServer")
    
    # Use the @tool decorator
    tool_func = server.tool(name=name, description=description)(func)
    
    return tool_func


def extract_tools_from_fastmcp_v2_server(server: Any) -> List[BaseTool]:
    """Extract all tools from a FastMCP 2.0 server as LangChain tools.
    
    Args:
        server: FastMCP 2.0 server instance
        
    Returns:
        List of LangChain tools
    """
    adapter = FastMCPv2ServerAdapter(server)
    return adapter.get_tools_sync()