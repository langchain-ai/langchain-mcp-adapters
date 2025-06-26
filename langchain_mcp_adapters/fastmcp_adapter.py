"""
FastMCP 2.0 SDK adapter for LangChain and LangGraph integration.

This module provides adapters to seamlessly integrate FastMCP 2.0 servers
with LangChain tools and LangGraph agents.

FastMCP 2.0 is the independent FastMCP SDK (jlowin/fastmcp), not the
fastmcp module in the official MCP Python SDK.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, create_model

try:
    # FastMCP 2.0 SDK imports
    from fastmcp import FastMCP
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.transports import FastMCPTransport
    FASTMCP_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path for FastMCP 2.0
        import fastmcp
        FastMCP = fastmcp.FastMCP
        FastMCPClient = fastmcp.Client
        FastMCPTransport = getattr(fastmcp, 'FastMCPTransport', None)
        FASTMCP_AVAILABLE = True
    except (ImportError, AttributeError):
        FASTMCP_AVAILABLE = False
        FastMCP = None
        FastMCPClient = None
        FastMCPTransport = None


class FastMCP2Adapter:
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
            server_script: Path to FastMCP 2.0 server script
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
            )
        
        if not any([fastmcp_server, server_url, server_script]):
            raise ValueError("Either fastmcp_server, server_url, or server_script must be provided")
        
        self.fastmcp_server = fastmcp_server
        self.server_url = server_url
        self.server_script = server_script
        self._client: Optional[FastMCPClient] = None
    
    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[FastMCPClient]:
        """Get a FastMCP 2.0 client for the server."""
        if self.fastmcp_server is not None:
            # Direct connection to FastMCP 2.0 server instance using in-memory transport
            async with FastMCPClient(self.fastmcp_server) as client:
                yield client
        elif self.server_url is not None:
            # Connect to remote FastMCP 2.0 server via HTTP/SSE
            async with FastMCPClient(self.server_url) as client:
                yield client
        elif self.server_script is not None:
            # Connect to FastMCP 2.0 server script via stdio
            async with FastMCPClient(self.server_script) as client:
                yield client
        else:
            raise ValueError("No server, URL, or script configured")
    
    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from the FastMCP 2.0 server as LangChain tools.
        
        Returns:
            List of LangChain BaseTool instances
        """
        async with self._get_client() as client:
            # List all available tools from the FastMCP 2.0 server
            tools = await client.list_tools()
            
            langchain_tools = []
            for tool in tools:
                langchain_tool = self._convert_fastmcp2_tool_to_langchain(client, tool)
                langchain_tools.append(langchain_tool)
            
            return langchain_tools
    
    def _convert_fastmcp2_tool_to_langchain(self, client: FastMCPClient, tool: Any) -> BaseTool:
        """Convert a FastMCP 2.0 tool to a LangChain tool.
        
        Args:
            client: FastMCP 2.0 client instance
            tool: FastMCP 2.0 tool definition
            
        Returns:
            LangChain BaseTool instance
        """
        # Create the tool execution function
        async def execute_tool(**kwargs: Any) -> Any:
            try:
                # Call the tool through the FastMCP 2.0 client
                async with self._get_client() as active_client:
                    result = await active_client.call_tool(tool.name, kwargs)
                
                # Handle different result types from FastMCP 2.0
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
                elif isinstance(result, (str, int, float, bool)):
                    return str(result)
                else:
                    return str(result)
                    
            except Exception as e:
                raise ToolException(f"Error executing FastMCP 2.0 tool '{tool.name}': {str(e)}")
        
        # Create args schema from tool input schema
        args_schema = self._create_args_schema_from_fastmcp2_tool(tool)
        
        # Create the LangChain tool
        return StructuredTool(
            name=tool.name,
            description=tool.description or f"FastMCP 2.0 tool: {tool.name}",
            args_schema=args_schema,
            coroutine=execute_tool,
            response_format="content_and_artifact"
        )
    
    def _create_args_schema_from_fastmcp2_tool(self, tool: Any) -> type[BaseModel]:
        """Create a Pydantic model for tool arguments from FastMCP 2.0 tool schema.
        
        Args:
            tool: FastMCP 2.0 tool definition
            
        Returns:
            Pydantic BaseModel class for tool arguments
        """
        # FastMCP 2.0 tools may have different schema formats
        schema = None
        
        # Try different ways to get the schema from FastMCP 2.0 tools
        if hasattr(tool, 'input_schema'):
            schema = tool.input_schema
        elif hasattr(tool, 'inputSchema'):
            schema = tool.inputSchema
        elif hasattr(tool, 'parameters'):
            schema = tool.parameters
        elif hasattr(tool, 'args_schema'):
            # If it's already a Pydantic model
            if isinstance(tool.args_schema, type) and issubclass(tool.args_schema, BaseModel):
                return tool.args_schema
            schema = tool.args_schema
        
        if schema and isinstance(schema, dict):
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


class FastMCP2ServerAdapter:
    """Adapter for running FastMCP 2.0 servers and extracting tools."""
    
    def __init__(self, fastmcp_server: Any):
        """Initialize with a FastMCP 2.0 server instance.
        
        Args:
            fastmcp_server: FastMCP 2.0 server instance
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
            )
        
        self.fastmcp_server = fastmcp_server
    
    async def get_tools(self) -> List[BaseTool]:
        """Get tools directly from the FastMCP 2.0 server instance.
        
        Returns:
            List of LangChain BaseTool instances
        """
        adapter = FastMCP2Adapter(fastmcp_server=self.fastmcp_server)
        return await adapter.get_tools()
    
    def get_tools_sync(self) -> List[BaseTool]:
        """Synchronous version of get_tools for easier integration.
        
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


# Convenience functions for easy integration with FastMCP 2.0
async def load_fastmcp2_tools(
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
    adapter = FastMCP2Adapter(
        fastmcp_server=fastmcp_server, 
        server_url=server_url,
        server_script=server_script
    )
    return await adapter.get_tools()


def load_fastmcp2_tools_sync(
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
    try:
        loop = asyncio.get_running_loop()
        # If we're in an event loop, we can't use asyncio.run()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run, 
                load_fastmcp2_tools(fastmcp_server, server_url, server_script)
            )
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(load_fastmcp2_tools(fastmcp_server, server_url, server_script))


def create_fastmcp2_tool_from_function(func, name: Optional[str] = None, description: Optional[str] = None):
    """Create a FastMCP 2.0 tool from a Python function.
    
    This is a helper function to create FastMCP 2.0 tools that can then be
    converted to LangChain tools.
    
    Args:
        func: Python function to convert
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to function docstring)
        
    Returns:
        FastMCP 2.0 tool that can be added to a FastMCP 2.0 server
    """
    if not FASTMCP_AVAILABLE:
        raise ImportError(
            "FastMCP 2.0 is not available. Please install it with: pip install fastmcp"
        )
    
    # In FastMCP 2.0, tools are typically created using decorators on the server
    # This function would help create tools programmatically
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or f"Tool: {tool_name}"
    
    # This is a placeholder - the actual implementation would depend on
    # FastMCP 2.0's specific API for programmatic tool creation
    return {
        "name": tool_name,
        "description": tool_description,
        "function": func,
        "type": "function"
    }


# Backward compatibility aliases
FastMCPAdapter = FastMCP2Adapter
FastMCPServerAdapter = FastMCP2ServerAdapter
load_fastmcp_tools = load_fastmcp2_tools
load_fastmcp_tools_sync = load_fastmcp2_tools_sync
create_fastmcp_tool_from_function = create_fastmcp2_tool_from_function