"""
Demo script for FastMCP 2.0 SDK integration with LangGraph.

This script demonstrates the key features of the FastMCP 2.0 integration
without requiring an actual FastMCP 2.0 installation.
"""

import asyncio
from typing import List

# Mock LangGraph components
class MockAgent:
    def __init__(self, model_name: str, tools: List):
        self.model_name = model_name
        self.tools = tools
        print(f"🤖 Created agent with {len(tools)} FastMCP 2.0 tools")
    
    async def ainvoke(self, state):
        message = state.get("messages", "")
        tool_names = [tool.name for tool in self.tools]
        return {
            "response": f"I have {len(self.tools)} FastMCP 2.0 tools available: {', '.join(tool_names)}"
        }

def create_react_agent(model_name: str, tools: List):
    return MockAgent(model_name, tools)

# Mock FastMCP 2.0 server
class MockFastMCPServer:
    def __init__(self, name: str):
        self.name = name
        self._tools = {}
    
    def tool(self, name: str = None, description: str = None):
        def decorator(func):
            tool_name = name or func.__name__
            self._tools[tool_name] = func
            func.name = tool_name
            func.description = description or func.__doc__ or f"Tool: {tool_name}"
            return func
        return decorator

# Import our FastMCP 2.0 integration
try:
    from langchain_mcp_adapters import (
        FastMCPv2Client,
        FastMCPv2Adapter,
        FastMCPv2ServerManager,
        extract_tools_from_fastmcp_v2_server,
        quick_load_fastmcp_v2_tools_sync,
        FASTMCP_V2_AVAILABLE
    )
    print("✅ FastMCP 2.0 adapters imported successfully")
    ADAPTERS_AVAILABLE = True
except ImportError as e:
    print(f"❌ FastMCP 2.0 adapters not available: {e}")
    ADAPTERS_AVAILABLE = False

async def demo_direct_server_integration():
    """Demo direct FastMCP 2.0 server integration."""
    print("\n🚀 Demo: Direct FastMCP 2.0 Server Integration")
    print("=" * 50)
    
    if not ADAPTERS_AVAILABLE:
        print("❌ FastMCP 2.0 adapters not available")
        return
    
    try:
        # Create a mock FastMCP 2.0 server
        server = MockFastMCPServer("Demo Math Server")
        
        # Add some tools
        @server.tool(description="Add two numbers")
        def add(a: float, b: float) -> float:
            return a + b
        
        @server.tool(description="Multiply two numbers")
        def multiply(a: float, b: float) -> float:
            return a * b
        
        @server.tool(description="Calculate power")
        def power(base: float, exp: float) -> float:
            return base ** exp
        
        print(f"✅ Created mock FastMCP 2.0 server with {len(server._tools)} tools")
        
        # Extract tools using our adapter
        tools = extract_tools_from_fastmcp_v2_server(server)
        print(f"✅ Extracted {len(tools)} LangChain tools from server")
        
        # List tool details
        for tool in tools:
            print(f"  🔧 {tool.name}: {tool.description}")
        
        # Create LangGraph agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test the agent
        response = await agent.ainvoke({"messages": "What can you help me with?"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ Direct server integration demo completed!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")

async def demo_client_integration():
    """Demo FastMCP 2.0 client integration."""
    print("\n🌐 Demo: FastMCP 2.0 Client Integration")
    print("=" * 50)
    
    if not ADAPTERS_AVAILABLE:
        print("❌ FastMCP 2.0 adapters not available")
        return
    
    try:
        # Create a mock server instance
        server = MockFastMCPServer("Client Demo Server")
        
        @server.tool(description="Convert text to uppercase")
        def uppercase(text: str) -> str:
            return text.upper()
        
        @server.tool(description="Reverse text")
        def reverse(text: str) -> str:
            return text[::-1]
        
        @server.tool(description="Count characters")
        def count_chars(text: str) -> int:
            return len(text)
        
        print(f"✅ Created mock server with {len(server._tools)} tools")
        
        # Create client
        client = FastMCPv2Client(server_instance=server)
        print("✅ Created FastMCP 2.0 client")
        
        # Get tools
        tools = await client.get_tools()
        print(f"✅ Loaded {len(tools)} tools via client")
        
        # List tool names
        tool_names = await client.list_tool_names()
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Get specific tool
        uppercase_tool = await client.get_tool_by_name("uppercase")
        if uppercase_tool:
            print(f"🔧 Found 'uppercase' tool: {uppercase_tool.description}")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test
        response = await agent.ainvoke({"messages": "Help me with text processing"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ Client integration demo completed!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")

async def demo_server_manager():
    """Demo FastMCP 2.0 server manager."""
    print("\n🏗️ Demo: FastMCP 2.0 Server Manager")
    print("=" * 50)
    
    if not ADAPTERS_AVAILABLE:
        print("❌ FastMCP 2.0 adapters not available")
        return
    
    try:
        # Create server manager
        manager = FastMCPv2ServerManager()
        print("✅ Created server manager")
        
        # Create multiple servers
        math_server = MockFastMCPServer("Math Server")
        
        @math_server.tool(description="Add numbers")
        def add(a: float, b: float) -> float:
            return a + b
        
        @math_server.tool(description="Subtract numbers")
        def subtract(a: float, b: float) -> float:
            return a - b
        
        text_server = MockFastMCPServer("Text Server")
        
        @text_server.tool(description="Make uppercase")
        def upper(text: str) -> str:
            return text.upper()
        
        @text_server.tool(description="Make lowercase")
        def lower(text: str) -> str:
            return text.lower()
        
        # Add servers to manager
        manager.add_server("math", math_server)
        manager.add_server("text", text_server)
        print(f"✅ Added 2 servers to manager")
        
        # Get all tools
        all_tools = await manager.get_all_tools()
        total_tools = sum(len(tools) for tools in all_tools.values())
        print(f"✅ Total tools across all servers: {total_tools}")
        
        for server_name, tools in all_tools.items():
            print(f"  📦 {server_name}: {len(tools)} tools")
        
        # Get tools from specific server
        math_tools = await manager.get_tools_from_server("math")
        print(f"🧮 Math server tools: {len(math_tools)}")
        
        # Create client for specific server
        math_client = manager.create_client_for_server("math")
        client_tools = await math_client.get_tools()
        print(f"🔧 Math client tools: {len(client_tools)}")
        
        print("✅ Server manager demo completed!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")

def demo_sync_integration():
    """Demo synchronous FastMCP 2.0 integration."""
    print("\n🔄 Demo: Synchronous FastMCP 2.0 Integration")
    print("=" * 50)
    
    if not ADAPTERS_AVAILABLE:
        print("❌ FastMCP 2.0 adapters not available")
        return
    
    try:
        # Create mock server
        server = MockFastMCPServer("Sync Demo Server")
        
        @server.tool(description="Simple calculation")
        def calculate(x: float, y: float) -> float:
            return x + y
        
        print(f"✅ Created mock server with {len(server._tools)} tools")
        
        # Extract tools synchronously
        tools = extract_tools_from_fastmcp_v2_server(server)
        print(f"✅ Synchronously extracted {len(tools)} tools")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        print("🤖 Agent created and ready!")
        
        print("✅ Synchronous integration demo completed!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")

async def main():
    """Run all FastMCP 2.0 integration demos."""
    print("🎯 FastMCP 2.0 SDK Integration Demos")
    print("=" * 60)
    print("These demos show the FastMCP 2.0 integration capabilities")
    print("using mock servers (no actual FastMCP 2.0 installation required)")
    print("=" * 60)
    
    # Check availability
    if ADAPTERS_AVAILABLE:
        print("✅ FastMCP 2.0 adapters are available")
        
        # Run demos
        await demo_direct_server_integration()
        await demo_client_integration()
        await demo_server_manager()
        demo_sync_integration()
        
        print("\n🎉 All FastMCP 2.0 integration demos completed!")
        
        print("\n📚 Key Features Demonstrated:")
        print("1. ✅ Direct server instance integration")
        print("2. ✅ Client-based server connections")
        print("3. ✅ Server management capabilities")
        print("4. ✅ Synchronous and asynchronous patterns")
        print("5. ✅ Tool extraction and conversion")
        print("6. ✅ LangGraph agent integration")
        
        print("\n🚀 Ready for Real FastMCP 2.0 Integration:")
        print("1. Install FastMCP 2.0: pip install fastmcp")
        print("2. Replace mock servers with real FastMCP 2.0 servers")
        print("3. Use the same adapter APIs shown in these demos")
        print("4. Build powerful AI applications!")
        
    else:
        print("❌ FastMCP 2.0 adapters not available")
        print("💡 This indicates an issue with the adapter installation")

if __name__ == "__main__":
    asyncio.run(main())