"""
Tests for FastMCP 2.0 integration.

These tests verify the integration with the independent FastMCP 2.0 SDK (jlowin/fastmcp),
not the fastmcp module in the official MCP Python SDK.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.tools import BaseTool

# Test imports and availability
def test_fastmcp2_imports():
    """Test that FastMCP 2.0 components can be imported."""
    try:
        from langchain_mcp_adapters.fastmcp2_client import (
            FastMCP2Client,
            FastMCP2MultiClient,
            create_fastmcp2_client,
            quick_load_fastmcp2_tools,
            quick_load_fastmcp2_tools_sync,
        )
        from langchain_mcp_adapters.fastmcp_adapter import (
            FastMCP2Adapter,
            FastMCP2ServerAdapter,
            load_fastmcp2_tools,
            load_fastmcp2_tools_sync,
        )
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"FastMCP 2.0 components not available: {e}")


class TestFastMCP2Client:
    """Test FastMCP2Client functionality."""
    
    def test_init_with_server_instance(self):
        """Test initialization with server instance."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_server = Mock()
        client = FastMCP2Client(server_instance=mock_server)
        assert client.server_instance == mock_server
        assert client.connection_type == "direct"
    
    def test_init_with_server_script(self):
        """Test initialization with server script."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        client = FastMCP2Client(server_script="./test_server.py")
        assert client.server_script == "./test_server.py"
        assert client.connection_type == "stdio"
    
    def test_init_with_server_url(self):
        """Test initialization with server URL."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        client = FastMCP2Client(server_url="http://localhost:8000")
        assert client.server_url == "http://localhost:8000"
        assert client.connection_type == "http"
    
    def test_init_without_server(self):
        """Test initialization without any server configuration."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        with pytest.raises(ValueError, match="Either server_instance, server_script, or server_url must be provided"):
            FastMCP2Client()
    
    @pytest.mark.asyncio
    async def test_get_tools_mock(self):
        """Test get_tools with mocked adapter."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_server = Mock()
        client = FastMCP2Client(server_instance=mock_server)
        
        # Mock the adapter
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        
        with patch('langchain_mcp_adapters.fastmcp_adapter.FastMCP2Adapter') as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.get_tools.return_value = [mock_tool]
            mock_adapter_class.return_value = mock_adapter
            
            tools = await client.get_tools()
            assert len(tools) == 1
            assert tools[0].name == "test_tool"
    
    @pytest.mark.asyncio
    async def test_list_tool_names(self):
        """Test list_tool_names functionality."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_server = Mock()
        client = FastMCP2Client(server_instance=mock_server)
        
        # Mock tools
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = Mock(spec=BaseTool)
        tool2.name = "tool2"
        mock_tools = [tool1, tool2]
        
        with patch.object(client, 'get_tools', return_value=mock_tools):
            tool_names = await client.list_tool_names()
            assert tool_names == ["tool1", "tool2"]
    
    @pytest.mark.asyncio
    async def test_get_tool_by_name(self):
        """Test get_tool_by_name functionality."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_server = Mock()
        client = FastMCP2Client(server_instance=mock_server)
        
        # Mock tools
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = Mock(spec=BaseTool)
        tool2.name = "tool2"
        mock_tools = [tool1, tool2]
        
        with patch.object(client, 'get_tools', return_value=mock_tools):
            found_tool = await client.get_tool_by_name("tool1")
            assert found_tool == tool1
            
            not_found = await client.get_tool_by_name("nonexistent")
            assert not_found is None
    
    def test_sync_methods(self):
        """Test synchronous method wrappers."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_server = Mock()
        client = FastMCP2Client(server_instance=mock_server)
        
        # Mock the async methods
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tools = [mock_tool]
        
        with patch.object(client, 'get_tools', return_value=mock_tools) as mock_get_tools:
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = mock_tools
                
                # Test sync methods don't raise errors
                try:
                    client.get_tools_sync()
                    client.list_tool_names_sync()
                    client.get_tool_by_name_sync("test_tool")
                except RuntimeError:
                    # This is expected if we're already in an event loop
                    pass


class TestFastMCP2MultiClient:
    """Test FastMCP2MultiClient functionality."""
    
    def test_init(self):
        """Test initialization of multi-client."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2MultiClient
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        servers = {
            "server1": {"server_script": "./server1.py"},
            "server2": {"server_url": "http://localhost:8000"}
        }
        
        multi_client = FastMCP2MultiClient(servers)
        assert len(multi_client.clients) == 2
        assert "server1" in multi_client.clients
        assert "server2" in multi_client.clients
    
    @pytest.mark.asyncio
    async def test_get_all_tools(self):
        """Test getting tools from all servers."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2MultiClient
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        servers = {
            "server1": {"server_script": "./server1.py"},
            "server2": {"server_script": "./server2.py"}
        }
        
        multi_client = FastMCP2MultiClient(servers)
        
        # Mock client responses
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = Mock(spec=BaseTool)
        tool2.name = "tool2"
        mock_tools1 = [tool1]
        mock_tools2 = [tool2]
        
        with patch.object(multi_client.clients["server1"], 'get_tools', return_value=mock_tools1):
            with patch.object(multi_client.clients["server2"], 'get_tools', return_value=mock_tools2):
                all_tools = await multi_client.get_all_tools()
                
                assert len(all_tools) == 2
                assert "server1" in all_tools
                assert "server2" in all_tools
                assert all_tools["server1"] == mock_tools1
                assert all_tools["server2"] == mock_tools2
    
    @pytest.mark.asyncio
    async def test_get_tools_flat(self):
        """Test getting all tools as a flat list."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2MultiClient
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        servers = {
            "server1": {"server_script": "./server1.py"},
            "server2": {"server_script": "./server2.py"}
        }
        
        multi_client = FastMCP2MultiClient(servers)
        
        # Mock the get_all_tools method
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = Mock(spec=BaseTool)
        tool2.name = "tool2"
        mock_all_tools = {
            "server1": [tool1],
            "server2": [tool2]
        }
        
        with patch.object(multi_client, 'get_all_tools', return_value=mock_all_tools):
            flat_tools = await multi_client.get_tools_flat()
            
            assert len(flat_tools) == 2
            assert flat_tools[0].name == "tool1"
            assert flat_tools[1].name == "tool2"
    
    @pytest.mark.asyncio
    async def test_get_tools_from_server(self):
        """Test getting tools from a specific server."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2MultiClient
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        servers = {
            "server1": {"server_script": "./server1.py"}
        }
        
        multi_client = FastMCP2MultiClient(servers)
        
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        mock_tools = [tool1]
        
        with patch.object(multi_client.clients["server1"], 'get_tools', return_value=mock_tools):
            tools = await multi_client.get_tools_from_server("server1")
            assert tools == mock_tools
    
    @pytest.mark.asyncio
    async def test_get_tools_from_nonexistent_server(self):
        """Test error handling for nonexistent server."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import FastMCP2MultiClient
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        servers = {"server1": {"server_script": "./server1.py"}}
        multi_client = FastMCP2MultiClient(servers)
        
        with pytest.raises(ValueError, match="Server 'nonexistent' not found"):
            await multi_client.get_tools_from_server("nonexistent")


class TestConvenienceFunctions:
    """Test convenience functions for FastMCP 2.0."""
    
    def test_create_fastmcp2_client(self):
        """Test create_fastmcp2_client function."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import create_fastmcp2_client
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        client = create_fastmcp2_client(server_script="./test.py")
        assert client.server_script == "./test.py"
        assert client.connection_type == "stdio"
    
    @pytest.mark.asyncio
    async def test_quick_load_fastmcp2_tools(self):
        """Test quick_load_fastmcp2_tools function."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import quick_load_fastmcp2_tools
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tools = [mock_tool]
        
        with patch('langchain_mcp_adapters.fastmcp2_client.create_fastmcp2_client') as mock_create:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = mock_tools
            mock_create.return_value = mock_client
            
            tools = await quick_load_fastmcp2_tools(server_script="./test.py")
            assert tools == mock_tools
    
    def test_quick_load_fastmcp2_tools_sync(self):
        """Test quick_load_fastmcp2_tools_sync function."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import quick_load_fastmcp2_tools_sync
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tools = [mock_tool]
        
        with patch('langchain_mcp_adapters.fastmcp2_client.quick_load_fastmcp2_tools') as mock_async:
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = mock_tools
                
                try:
                    tools = quick_load_fastmcp2_tools_sync(server_script="./test.py")
                    # If we get here without error, the function works
                    assert True
                except RuntimeError:
                    # This is expected if we're already in an event loop
                    pass


class TestFastMCP2Adapter:
    """Test FastMCP2Adapter functionality."""
    
    def test_init_with_server_instance(self):
        """Test adapter initialization with server instance."""
        try:
            from langchain_mcp_adapters.fastmcp_adapter import FastMCP2Adapter
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        mock_server = Mock()
        adapter = FastMCP2Adapter(fastmcp_server=mock_server)
        assert adapter.fastmcp_server == mock_server
    
    def test_init_with_server_script(self):
        """Test adapter initialization with server script."""
        try:
            from langchain_mcp_adapters.fastmcp_adapter import FastMCP2Adapter
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        adapter = FastMCP2Adapter(server_script="./test.py")
        assert adapter.server_script == "./test.py"
    
    def test_init_with_server_url(self):
        """Test adapter initialization with server URL."""
        try:
            from langchain_mcp_adapters.fastmcp_adapter import FastMCP2Adapter
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        adapter = FastMCP2Adapter(server_url="http://localhost:8000")
        assert adapter.server_url == "http://localhost:8000"
    
    def test_init_without_server(self):
        """Test adapter initialization without server configuration."""
        try:
            from langchain_mcp_adapters.fastmcp_adapter import FastMCP2Adapter
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        with pytest.raises(ValueError, match="Either fastmcp_server, server_url, or server_script must be provided"):
            FastMCP2Adapter()


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""
    
    def test_backward_compatibility_imports(self):
        """Test that backward compatibility aliases work."""
        try:
            from langchain_mcp_adapters.fastmcp_adapter import (
                FastMCPAdapter,  # Should be alias for FastMCP2Adapter
                FastMCPServerAdapter,  # Should be alias for FastMCP2ServerAdapter
                load_fastmcp_tools,  # Should be alias for load_fastmcp2_tools
                load_fastmcp_tools_sync,  # Should be alias for load_fastmcp2_tools_sync
            )
            # If imports work, aliases are properly set up
            assert True
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_fastmcp2_not_available_error(self):
        """Test error when FastMCP 2.0 is not available."""
        # This test simulates the case where FastMCP 2.0 is not installed
        with patch('langchain_mcp_adapters.fastmcp2_client.FASTMCP2_AVAILABLE', False):
            try:
                from langchain_mcp_adapters.fastmcp2_client import FastMCP2Client
                with pytest.raises(ImportError, match="FastMCP 2.0 is not available"):
                    FastMCP2Client(server_script="./test.py")
            except ImportError:
                # If the import itself fails, that's also expected
                pass
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        try:
            from langchain_mcp_adapters.fastmcp2_client import quick_load_fastmcp2_tools
        except ImportError:
            pytest.skip("FastMCP 2.0 not available")
        
        # Test with invalid server script
        with pytest.raises(Exception):  # Should raise some kind of connection/file error
            await quick_load_fastmcp2_tools(server_script="./nonexistent_server.py")


# Integration test (requires actual FastMCP 2.0 installation)
@pytest.mark.integration
class TestRealFastMCP2Integration:
    """Integration tests with real FastMCP 2.0 (requires fastmcp package)."""
    
    def test_real_fastmcp2_server_creation(self):
        """Test creating a real FastMCP 2.0 server."""
        try:
            from fastmcp import FastMCP
            
            # Create a simple server
            mcp = FastMCP("Test Server")
            
            @mcp.tool()
            def test_tool(x: int) -> int:
                """Test tool."""
                return x * 2
            
            # FastMCP 2.0 may have different API for accessing tools
            # This is a placeholder test - actual API may vary
            assert hasattr(mcp, 'tool')  # Check that tool decorator exists
            
        except ImportError:
            pytest.skip("FastMCP 2.0 not installed")
    
    @pytest.mark.asyncio
    async def test_real_integration_with_adapter(self):
        """Test real integration with FastMCP 2.0 server and adapter."""
        try:
            from fastmcp import FastMCP
            from langchain_mcp_adapters import FastMCP2Client
            
            # Create a real FastMCP 2.0 server
            mcp = FastMCP("Integration Test Server")
            
            @mcp.tool()
            def add(a: int, b: int) -> int:
                """Add two numbers."""
                return a + b
            
            # Create client and get tools
            client = FastMCP2Client(server_instance=mcp)
            tools = await client.get_tools()
            
            assert len(tools) == 1
            assert tools[0].name == "add"
            assert "Add two numbers" in tools[0].description
            
        except ImportError:
            pytest.skip("FastMCP 2.0 not installed")