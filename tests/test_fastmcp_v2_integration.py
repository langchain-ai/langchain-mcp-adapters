"""
Tests for FastMCP 2.0 SDK integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain_core.tools import BaseTool

# Test imports with proper error handling
try:
    from langchain_mcp_adapters.fastmcp_v2_adapter import (
        FastMCPv2Adapter,
        FastMCPv2ServerAdapter,
        load_fastmcp_v2_tools,
        load_fastmcp_v2_tools_sync,
        FASTMCP_V2_AVAILABLE
    )
    from langchain_mcp_adapters.fastmcp_v2_client import (
        FastMCPv2Client,
        FastMCPv2MultiClient,
        FastMCPv2ServerManager,
        quick_load_fastmcp_v2_tools,
        quick_load_fastmcp_v2_tools_sync,
    )
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="FastMCP 2.0 adapters not available")
class TestFastMCPv2Adapter:
    """Test FastMCP 2.0 adapter functionality."""
    
    def test_init_with_server_instance(self):
        """Test initialization with server instance."""
        mock_server = Mock()
        adapter = FastMCPv2Adapter(fastmcp_server=mock_server)
        assert adapter.fastmcp_server == mock_server
        assert adapter.server_url is None
        assert adapter.server_script is None
    
    def test_init_with_server_url(self):
        """Test initialization with server URL."""
        adapter = FastMCPv2Adapter(server_url="http://localhost:8000")
        assert adapter.server_url == "http://localhost:8000"
        assert adapter.fastmcp_server is None
        assert adapter.server_script is None
    
    def test_init_with_server_script(self):
        """Test initialization with server script."""
        adapter = FastMCPv2Adapter(server_script="./test_server.py")
        assert adapter.server_script == "./test_server.py"
        assert adapter.fastmcp_server is None
        assert adapter.server_url is None
    
    def test_init_without_parameters(self):
        """Test that initialization fails without any parameters."""
        with pytest.raises(ValueError, match="Either fastmcp_server, server_url, or server_script must be provided"):
            FastMCPv2Adapter()
    
    @pytest.mark.asyncio
    async def test_get_tools_from_server_instance(self):
        """Test getting tools from server instance."""
        # Create mock server with tools
        mock_server = Mock()
        mock_server._tools = {
            "add": Mock(__name__="add"),
            "multiply": Mock(__name__="multiply")
        }
        
        adapter = FastMCPv2Adapter(fastmcp_server=mock_server)
        
        with patch.object(adapter, '_convert_fastmcp_function_to_langchain_tool') as mock_convert:
            mock_tool = Mock(spec=BaseTool)
            mock_convert.return_value = mock_tool
            
            tools = await adapter.get_tools()
            
            assert len(tools) == 2
            assert all(isinstance(tool, Mock) for tool in tools)
            assert mock_convert.call_count == 2
    
    def test_create_args_schema_from_signature(self):
        """Test creating args schema from function signature."""
        import inspect
        
        def test_func(a: int, b: str = "default") -> str:
            return f"{a}: {b}"
        
        adapter = FastMCPv2Adapter(fastmcp_server=Mock())
        sig = inspect.signature(test_func)
        schema = adapter._create_args_schema_from_signature("test_func", sig)
        
        # Check that schema is a Pydantic model
        assert hasattr(schema, 'model_fields')
        assert 'a' in schema.model_fields
        assert 'b' in schema.model_fields
    
    def test_json_schema_type_conversion(self):
        """Test JSON schema type to Python type conversion."""
        adapter = FastMCPv2Adapter(fastmcp_server=Mock())
        
        assert adapter._json_schema_type_to_python_type({"type": "string"}) == str
        assert adapter._json_schema_type_to_python_type({"type": "integer"}) == int
        assert adapter._json_schema_type_to_python_type({"type": "number"}) == float
        assert adapter._json_schema_type_to_python_type({"type": "boolean"}) == bool
        assert adapter._json_schema_type_to_python_type({"type": "array"}) == list
        assert adapter._json_schema_type_to_python_type({"type": "object"}) == dict
        assert adapter._json_schema_type_to_python_type({"type": "unknown"}) == str


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="FastMCP 2.0 adapters not available")
class TestFastMCPv2ServerAdapter:
    """Test FastMCP 2.0 server adapter functionality."""
    
    def test_init(self):
        """Test initialization."""
        mock_server = Mock()
        adapter = FastMCPv2ServerAdapter(mock_server)
        assert adapter.fastmcp_server == mock_server
    
    @pytest.mark.asyncio
    async def test_get_tools(self):
        """Test getting tools."""
        mock_server = Mock()
        adapter = FastMCPv2ServerAdapter(mock_server)
        
        with patch('langchain_mcp_adapters.fastmcp_v2_adapter.FastMCPv2Adapter') as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.get_tools.return_value = [Mock(spec=BaseTool)]
            mock_adapter_class.return_value = mock_adapter
            
            tools = await adapter.get_tools()
            
            assert len(tools) == 1
            mock_adapter_class.assert_called_once_with(fastmcp_server=mock_server)
            mock_adapter.get_tools.assert_called_once()


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="FastMCP 2.0 adapters not available")
class TestFastMCPv2Client:
    """Test FastMCP 2.0 client functionality."""
    
    def test_init_with_server_instance(self):
        """Test initialization with server instance."""
        mock_server = Mock()
        client = FastMCPv2Client(server_instance=mock_server)
        assert client.server_instance == mock_server
        assert client.connection_type == "direct"
    
    def test_init_with_server_url(self):
        """Test initialization with server URL."""
        client = FastMCPv2Client(server_url="http://localhost:8000")
        assert client.server_url == "http://localhost:8000"
        assert client.connection_type == "http"
    
    def test_init_with_server_script(self):
        """Test initialization with server script."""
        client = FastMCPv2Client(server_script="./test_server.py")
        assert client.server_script == "./test_server.py"
        assert client.connection_type == "script"
    
    def test_init_without_parameters(self):
        """Test that initialization fails without parameters."""
        with pytest.raises(ValueError, match="Either server_instance, server_url, or server_script must be provided"):
            FastMCPv2Client()
    
    @pytest.mark.asyncio
    async def test_get_tools(self):
        """Test getting tools."""
        mock_server = Mock()
        client = FastMCPv2Client(server_instance=mock_server)
        
        with patch('langchain_mcp_adapters.fastmcp_v2_client.FastMCPv2Adapter') as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.get_tools.return_value = [Mock(spec=BaseTool)]
            mock_adapter_class.return_value = mock_adapter
            
            tools = await client.get_tools()
            
            assert len(tools) == 1
            mock_adapter_class.assert_called_once()
            mock_adapter.get_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_tool_names(self):
        """Test listing tool names."""
        mock_server = Mock()
        client = FastMCPv2Client(server_instance=mock_server)
        
        # Mock tools with names
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = Mock(spec=BaseTool)
        tool2.name = "tool2"
        
        with patch.object(client, 'get_tools', return_value=[tool1, tool2]):
            names = await client.list_tool_names()
            assert names == ["tool1", "tool2"]
    
    @pytest.mark.asyncio
    async def test_get_tool_by_name(self):
        """Test getting tool by name."""
        mock_server = Mock()
        client = FastMCPv2Client(server_instance=mock_server)
        
        # Mock tools
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = Mock(spec=BaseTool)
        tool2.name = "tool2"
        
        with patch.object(client, 'get_tools', return_value=[tool1, tool2]):
            # Test finding existing tool
            found_tool = await client.get_tool_by_name("tool1")
            assert found_tool == tool1
            
            # Test not finding tool
            not_found = await client.get_tool_by_name("nonexistent")
            assert not_found is None


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="FastMCP 2.0 adapters not available")
class TestFastMCPv2MultiClient:
    """Test FastMCP 2.0 multi-client functionality."""
    
    def test_init(self):
        """Test initialization."""
        servers = {
            "server1": {"server_instance": Mock()},
            "server2": {"server_url": "http://localhost:8000"}
        }
        
        multi_client = FastMCPv2MultiClient(servers)
        assert len(multi_client.clients) == 2
        assert "server1" in multi_client.clients
        assert "server2" in multi_client.clients
    
    @pytest.mark.asyncio
    async def test_get_all_tools(self):
        """Test getting tools from all servers."""
        servers = {
            "server1": {"server_instance": Mock()},
            "server2": {"server_instance": Mock()}
        }
        
        multi_client = FastMCPv2MultiClient(servers)
        
        # Mock client tools
        with patch.object(multi_client.clients["server1"], 'get_tools', new_callable=AsyncMock) as mock1:
            with patch.object(multi_client.clients["server2"], 'get_tools', new_callable=AsyncMock) as mock2:
                tool1 = Mock(spec=BaseTool)
                tool1.name = "tool1"
                tool2 = Mock(spec=BaseTool)
                tool2.name = "tool2"
                
                mock1.return_value = [tool1]
                mock2.return_value = [tool2]
                
                all_tools = await multi_client.get_all_tools()
                
                assert len(all_tools) == 2
                assert "server1" in all_tools
                assert "server2" in all_tools
                assert len(all_tools["server1"]) == 1
                assert len(all_tools["server2"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_tools_flat(self):
        """Test getting all tools as flat list."""
        servers = {"server1": {"server_instance": Mock()}}
        multi_client = FastMCPv2MultiClient(servers)
        
        # Mock get_all_tools
        mock_all_tools = {
            "server1": [Mock(spec=BaseTool), Mock(spec=BaseTool)]
        }
        
        with patch.object(multi_client, 'get_all_tools', return_value=mock_all_tools):
            flat_tools = await multi_client.get_tools_flat()
            assert len(flat_tools) == 2
    
    @pytest.mark.asyncio
    async def test_get_tools_from_server(self):
        """Test getting tools from specific server."""
        servers = {"server1": {"server_instance": Mock()}}
        multi_client = FastMCPv2MultiClient(servers)
        
        with patch.object(multi_client.clients["server1"], 'get_tools', new_callable=AsyncMock) as mock:
            mock.return_value = [Mock(spec=BaseTool)]
            
            tools = await multi_client.get_tools_from_server("server1")
            assert len(tools) == 1
            mock.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_tools_from_nonexistent_server(self):
        """Test error when getting tools from nonexistent server."""
        multi_client = FastMCPv2MultiClient({})
        
        with pytest.raises(ValueError, match="Server 'nonexistent' not found"):
            await multi_client.get_tools_from_server("nonexistent")


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="FastMCP 2.0 adapters not available")
class TestFastMCPv2ServerManager:
    """Test FastMCP 2.0 server manager functionality."""
    
    def test_init(self):
        """Test initialization."""
        manager = FastMCPv2ServerManager()
        assert isinstance(manager.servers, dict)
        assert len(manager.servers) == 0
    
    def test_add_remove_server(self):
        """Test adding and removing servers."""
        manager = FastMCPv2ServerManager()
        mock_server = Mock()
        
        # Add server
        manager.add_server("test", mock_server)
        assert "test" in manager.servers
        assert manager.servers["test"] == mock_server
        
        # Remove server
        manager.remove_server("test")
        assert "test" not in manager.servers
    
    def test_create_client_for_server(self):
        """Test creating client for managed server."""
        manager = FastMCPv2ServerManager()
        mock_server = Mock()
        manager.add_server("test", mock_server)
        
        client = manager.create_client_for_server("test")
        assert isinstance(client, FastMCPv2Client)
        assert client.server_instance == mock_server
    
    def test_create_client_for_nonexistent_server(self):
        """Test error when creating client for nonexistent server."""
        manager = FastMCPv2ServerManager()
        
        with pytest.raises(ValueError, match="Server 'nonexistent' not found"):
            manager.create_client_for_server("nonexistent")


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="FastMCP 2.0 adapters not available")
class TestConvenienceFunctions:
    """Test FastMCP 2.0 convenience functions."""
    
    @pytest.mark.asyncio
    async def test_load_fastmcp_v2_tools(self):
        """Test load_fastmcp_v2_tools function."""
        mock_server = Mock()
        
        with patch('langchain_mcp_adapters.fastmcp_v2_adapter.FastMCPv2Adapter') as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.get_tools.return_value = [Mock(spec=BaseTool)]
            mock_adapter_class.return_value = mock_adapter
            
            tools = await load_fastmcp_v2_tools(fastmcp_server=mock_server)
            assert len(tools) == 1
            mock_adapter_class.assert_called_once()
            mock_adapter.get_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quick_load_fastmcp_v2_tools(self):
        """Test quick_load_fastmcp_v2_tools function."""
        with patch('langchain_mcp_adapters.fastmcp_v2_client.load_fastmcp_v2_tools', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = [Mock(spec=BaseTool)]
            
            tools = await quick_load_fastmcp_v2_tools(server_script="test.py")
            assert len(tools) == 1
            mock_load.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])