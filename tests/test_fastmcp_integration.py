"""
Tests for FastMCP integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.tools import BaseTool

from langchain_mcp_adapters.fastmcp_client import (
    FastMCPClient,
    FastMCPMultiClient,
    create_fastmcp_client,
    quick_load_fastmcp_tools,
    quick_load_fastmcp_tools_sync,
)


class TestFastMCPClient:
    """Test FastMCPClient functionality."""
    
    def test_init_with_script(self):
        """Test initialization with server script."""
        client = FastMCPClient(server_script="test_server.py")
        assert client.server_script == "test_server.py"
        assert client.connection_type == "stdio"
        assert client.server_args == ["test_server.py"]
    
    def test_init_with_url(self):
        """Test initialization with server URL."""
        client = FastMCPClient(server_url="http://localhost:8000/mcp")
        assert client.server_url == "http://localhost:8000/mcp"
        assert client.connection_type == "http"
    
    def test_init_with_custom_args(self):
        """Test initialization with custom arguments."""
        client = FastMCPClient(
            server_script="test_server.py",
            server_command="python3",
            server_args=["test_server.py", "--port", "9000"]
        )
        assert client.server_command == "python3"
        assert client.server_args == ["test_server.py", "--port", "9000"]
    
    def test_init_without_script_or_url(self):
        """Test that initialization fails without script or URL."""
        with pytest.raises(ValueError, match="Either server_script or server_url must be provided"):
            FastMCPClient()
    
    @pytest.mark.asyncio
    async def test_get_tools_mock(self):
        """Test get_tools with mocked session."""
        client = FastMCPClient(server_script="test_server.py")
        
        # Mock the session and tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Mock load_mcp_tools
            with patch('langchain_mcp_adapters.fastmcp_client.load_mcp_tools') as mock_load:
                mock_load.return_value = [Mock(spec=BaseTool)]
                
                tools = await client.get_tools()
                assert len(tools) == 1
                assert isinstance(tools[0], Mock)
                mock_load.assert_called_once_with(mock_session)
    
    @pytest.mark.asyncio
    async def test_list_tool_names(self):
        """Test listing tool names."""
        client = FastMCPClient(server_script="test_server.py")
        
        # Mock tools with proper name attributes
        tool1 = Mock()
        tool1.name = "tool1"
        tool2 = Mock()
        tool2.name = "tool2"
        tool3 = Mock()
        tool3.name = "tool3"
        mock_tools = [tool1, tool2, tool3]
        
        with patch.object(client, 'get_tools', return_value=mock_tools):
            names = await client.list_tool_names()
            assert names == ["tool1", "tool2", "tool3"]
    
    @pytest.mark.asyncio
    async def test_get_tool_by_name(self):
        """Test getting tool by name."""
        client = FastMCPClient(server_script="test_server.py")
        
        # Mock tools with proper name attributes
        tool1 = Mock()
        tool1.name = "tool1"
        tool2 = Mock()
        tool2.name = "tool2"
        mock_tools = [tool1, tool2]
        
        with patch.object(client, 'get_tools', return_value=mock_tools):
            # Test finding existing tool
            found_tool = await client.get_tool_by_name("tool1")
            assert found_tool == tool1
            
            # Test not finding tool
            not_found = await client.get_tool_by_name("nonexistent")
            assert not_found is None
    
    def test_sync_methods(self):
        """Test synchronous wrapper methods."""
        client = FastMCPClient(server_script="test_server.py")
        
        # Mock the async methods
        with patch.object(client, 'get_tools', new_callable=AsyncMock) as mock_get_tools:
            test_tool = Mock()
            test_tool.name = "test_tool"
            mock_get_tools.return_value = [test_tool]
            
            # Test sync get_tools
            tools = client.get_tools_sync()
            assert len(tools) == 1
            
        with patch.object(client, 'get_tool_by_name', new_callable=AsyncMock) as mock_get_tool:
            test_tool = Mock()
            test_tool.name = "test_tool"
            mock_get_tool.return_value = test_tool
            
            # Test sync get_tool_by_name
            tool = client.get_tool_by_name_sync("test_tool")
            assert tool.name == "test_tool"
            
        with patch.object(client, 'list_tool_names', new_callable=AsyncMock) as mock_list_names:
            mock_list_names.return_value = ["tool1", "tool2"]
            
            # Test sync list_tool_names
            names = client.list_tool_names_sync()
            assert names == ["tool1", "tool2"]


class TestFastMCPMultiClient:
    """Test FastMCPMultiClient functionality."""
    
    def test_init(self):
        """Test initialization."""
        servers = {
            "server1": {"server_script": "server1.py"},
            "server2": {"server_url": "http://localhost:8000/mcp"}
        }
        
        multi_client = FastMCPMultiClient(servers)
        assert len(multi_client.clients) == 2
        assert "server1" in multi_client.clients
        assert "server2" in multi_client.clients
    
    @pytest.mark.asyncio
    async def test_get_all_tools(self):
        """Test getting tools from all servers."""
        servers = {
            "server1": {"server_script": "server1.py"},
            "server2": {"server_script": "server2.py"}
        }
        
        multi_client = FastMCPMultiClient(servers)
        
        # Mock client tools
        with patch.object(multi_client.clients["server1"], 'get_tools', new_callable=AsyncMock) as mock1:
            with patch.object(multi_client.clients["server2"], 'get_tools', new_callable=AsyncMock) as mock2:
                mock1.return_value = [Mock(name="tool1")]
                mock2.return_value = [Mock(name="tool2")]
                
                all_tools = await multi_client.get_all_tools()
                
                assert len(all_tools) == 2
                assert "server1" in all_tools
                assert "server2" in all_tools
                assert len(all_tools["server1"]) == 1
                assert len(all_tools["server2"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_tools_flat(self):
        """Test getting all tools as flat list."""
        servers = {
            "server1": {"server_script": "server1.py"},
            "server2": {"server_script": "server2.py"}
        }
        
        multi_client = FastMCPMultiClient(servers)
        
        # Mock get_all_tools
        mock_all_tools = {
            "server1": [Mock(name="tool1"), Mock(name="tool2")],
            "server2": [Mock(name="tool3")]
        }
        
        with patch.object(multi_client, 'get_all_tools', return_value=mock_all_tools):
            flat_tools = await multi_client.get_tools_flat()
            assert len(flat_tools) == 3
    
    @pytest.mark.asyncio
    async def test_get_tools_from_server(self):
        """Test getting tools from specific server."""
        servers = {
            "server1": {"server_script": "server1.py"},
            "server2": {"server_script": "server2.py"}
        }
        
        multi_client = FastMCPMultiClient(servers)
        
        with patch.object(multi_client.clients["server1"], 'get_tools', new_callable=AsyncMock) as mock:
            mock.return_value = [Mock(name="tool1")]
            
            tools = await multi_client.get_tools_from_server("server1")
            assert len(tools) == 1
            mock.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_tools_from_nonexistent_server(self):
        """Test error when getting tools from nonexistent server."""
        multi_client = FastMCPMultiClient({})
        
        with pytest.raises(ValueError, match="Server 'nonexistent' not found"):
            await multi_client.get_tools_from_server("nonexistent")


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_fastmcp_client(self):
        """Test create_fastmcp_client function."""
        client = create_fastmcp_client(server_script="test.py")
        assert isinstance(client, FastMCPClient)
        assert client.server_script == "test.py"
    
    @pytest.mark.asyncio
    async def test_quick_load_fastmcp_tools(self):
        """Test quick_load_fastmcp_tools function."""
        with patch('langchain_mcp_adapters.fastmcp_client.FastMCPClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [Mock(name="test_tool")]
            mock_client_class.return_value = mock_client
            
            tools = await quick_load_fastmcp_tools(server_script="test.py")
            assert len(tools) == 1
            mock_client.get_tools.assert_called_once()
    
    def test_quick_load_fastmcp_tools_sync(self):
        """Test quick_load_fastmcp_tools_sync function."""
        with patch('langchain_mcp_adapters.fastmcp_client.quick_load_fastmcp_tools', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = [Mock(name="test_tool")]
            
            tools = quick_load_fastmcp_tools_sync(server_script="test.py")
            assert len(tools) == 1


if __name__ == "__main__":
    pytest.main([__file__])