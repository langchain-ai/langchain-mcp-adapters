"""
Complete FastMCP 2.0 Integration Example

This example demonstrates how to integrate FastMCP 2.0 servers with LangGraph
using the langchain-mcp-adapters library.

FastMCP 2.0 is the independent FastMCP SDK (jlowin/fastmcp), not the
fastmcp module in the official MCP Python SDK.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock LangGraph components for demonstration
class MockAgent:
    def __init__(self, model_name: str, tools):
        self.model_name = model_name
        self.tools = tools
        print(f"Created mock agent with model {model_name} and {len(tools)} tools")
    
    async def ainvoke(self, state):
        messages = state.get("messages", "")
        print(f"Agent processing: {messages}")
        
        # Mock response based on the message content
        if "add" in messages.lower() or "+" in messages:
            return {"response": "I can help you add numbers using the FastMCP 2.0 add tool!"}
        elif "multiply" in messages.lower() or "*" in messages:
            return {"response": "I can help you multiply numbers using the FastMCP 2.0 multiply tool!"}
        elif "calculate" in messages.lower():
            return {"response": f"I have access to {len(self.tools)} FastMCP 2.0 math tools: add, subtract, multiply, divide, power, square_root, factorial, fibonacci, is_prime, gcd, lcm"}
        else:
            return {"response": f"I have {len(self.tools)} FastMCP 2.0 tools available to help you!"}

def create_react_agent(model_name: str, tools):
    """Mock function to create a react agent."""
    return MockAgent(model_name, tools)


async def example_fastmcp2_direct_integration():
    """Example of direct FastMCP 2.0 server integration."""
    print("🚀 FastMCP 2.0 Direct Integration")
    print("=" * 50)
    
    try:
        # Import FastMCP 2.0 components
        from langchain_mcp_adapters import (
            FastMCP2Client,
            quick_load_fastmcp2_tools,
            FASTMCP2_AVAILABLE
        )
        
        if not FASTMCP2_AVAILABLE:
            print("❌ FastMCP 2.0 is not available. Please install it with: pip install fastmcp")
            return
        
        # Create a FastMCP 2.0 server instance
        from fastmcp import FastMCP
        
        # Create a simple FastMCP 2.0 server
        mcp_server = FastMCP("Test Math Server")
        
        @mcp_server.tool()
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b
        
        @mcp_server.tool()
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b
        
        print("✅ Created FastMCP 2.0 server with tools")
        
        # Create client for direct server integration
        client = FastMCP2Client(server_instance=mcp_server)
        
        # Get tools from the server
        tools = await client.get_tools()
        print(f"✅ Loaded {len(tools)} tools from FastMCP 2.0 server")
        
        # List tool names
        tool_names = await client.list_tool_names()
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Create LangGraph agent with the tools
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test the agent
        response = await agent.ainvoke({"messages": "What's 5 + 3?"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ Direct integration completed successfully!\n")
        
    except Exception as e:
        print(f"❌ Error in direct integration: {e}")
        import traceback
        traceback.print_exc()


async def example_fastmcp2_script_integration():
    """Example of FastMCP 2.0 script integration."""
    print("📜 FastMCP 2.0 Script Integration")
    print("=" * 50)
    
    try:
        from langchain_mcp_adapters import (
            quick_load_fastmcp2_tools,
            FASTMCP2_AVAILABLE
        )
        
        if not FASTMCP2_AVAILABLE:
            print("❌ FastMCP 2.0 is not available. Please install it with: pip install fastmcp")
            return
        
        # Load tools from FastMCP 2.0 server script
        tools = await quick_load_fastmcp2_tools(
            server_script="./examples/fastmcp2_math_server.py"
        )
        print(f"✅ Loaded {len(tools)} tools from FastMCP 2.0 script")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test the agent
        response = await agent.ainvoke({"messages": "Can you help me with calculations?"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ Script integration completed successfully!\n")
        
    except Exception as e:
        print(f"❌ Error in script integration: {e}")
        print("💡 Make sure the FastMCP 2.0 math server script is available")


async def example_fastmcp2_multi_server():
    """Example of multiple FastMCP 2.0 servers."""
    print("🌐 FastMCP 2.0 Multi-Server Integration")
    print("=" * 50)
    
    try:
        from langchain_mcp_adapters import (
            FastMCP2MultiClient,
            FASTMCP2_AVAILABLE
        )
        
        if not FASTMCP2_AVAILABLE:
            print("❌ FastMCP 2.0 is not available. Please install it with: pip install fastmcp")
            return
        
        # Configure multiple FastMCP 2.0 servers
        servers = {
            "math": {
                "server_script": "./examples/fastmcp2_math_server.py"
            },
            # Note: This would require a running HTTP server
            # "weather": {
            #     "server_url": "http://localhost:8000/"
            # }
        }
        
        # Create multi-client
        multi_client = FastMCP2MultiClient(servers)
        print(f"✅ Created multi-client with {len(servers)} FastMCP 2.0 servers")
        
        # Get tools from all servers
        all_tools = await multi_client.get_tools_flat()
        print(f"✅ Loaded {len(all_tools)} tools from all FastMCP 2.0 servers")
        
        # Get tools from specific server
        math_tools = await multi_client.get_tools_from_server("math")
        print(f"🧮 Math server has {len(math_tools)} tools")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", all_tools)
        
        # Test the agent
        response = await agent.ainvoke({"messages": "Help me with math calculations"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ Multi-server integration completed successfully!\n")
        
    except Exception as e:
        print(f"❌ Error in multi-server integration: {e}")
        print("💡 Make sure the FastMCP 2.0 math server script is available")


def example_fastmcp2_sync_usage():
    """Example of synchronous FastMCP 2.0 usage."""
    print("🔄 FastMCP 2.0 Synchronous Usage")
    print("=" * 50)
    
    try:
        from langchain_mcp_adapters import (
            quick_load_fastmcp2_tools_sync,
            FastMCP2Client,
            FASTMCP2_AVAILABLE
        )
        
        if not FASTMCP2_AVAILABLE:
            print("❌ FastMCP 2.0 is not available. Please install it with: pip install fastmcp")
            return
        
        # Create a simple FastMCP 2.0 server
        from fastmcp import FastMCP
        
        mcp_server = FastMCP("Sync Test Server")
        
        @mcp_server.tool()
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"
        
        # Use sync methods
        client = FastMCP2Client(server_instance=mcp_server)
        tools = client.get_tools_sync()
        print(f"✅ Sync loaded {len(tools)} tools")
        
        tool_names = client.list_tool_names_sync()
        print(f"📋 Tool names: {', '.join(tool_names)}")
        
        greet_tool = client.get_tool_by_name_sync("greet")
        if greet_tool:
            print(f"🔧 Found greet tool: {greet_tool.description}")
        
        print("✅ Synchronous usage completed!\n")
        
    except Exception as e:
        print(f"❌ Error in sync usage: {e}")


async def example_fastmcp2_error_handling():
    """Example with proper error handling."""
    print("🛡️ FastMCP 2.0 Error Handling")
    print("=" * 50)
    
    try:
        from langchain_mcp_adapters import (
            quick_load_fastmcp2_tools,
            FASTMCP2_AVAILABLE
        )
        
        if not FASTMCP2_AVAILABLE:
            print("❌ FastMCP 2.0 is not available. Please install it with: pip install fastmcp")
            return
        
        # Try to connect to non-existent server
        try:
            tools = await quick_load_fastmcp2_tools(
                server_script="./nonexistent_fastmcp2_server.py"
            )
            print(f"Loaded {len(tools)} tools")
        except Exception as e:
            print(f"❌ Expected error connecting to non-existent server: {type(e).__name__}")
            print("💡 Falling back to empty tools list")
            tools = []
        
        # Create agent with fallback
        if tools:
            agent = create_react_agent("openai:gpt-4", tools)
            print("✅ Agent created with FastMCP 2.0 tools")
        else:
            agent = create_react_agent("openai:gpt-4", [])
            print("✅ Agent created without tools (fallback mode)")
        
        print("✅ Error handling example completed!\n")
        
    except Exception as e:
        print(f"❌ Error in error handling example: {e}")


async def main():
    """Run all FastMCP 2.0 examples."""
    print("🎯 FastMCP 2.0 Integration Examples")
    print("=" * 60)
    print("These examples show how to integrate FastMCP 2.0 servers with LangGraph")
    print("FastMCP 2.0 is the independent FastMCP SDK (jlowin/fastmcp)")
    print("=" * 60)
    print()
    
    # Check if FastMCP 2.0 is available
    try:
        import fastmcp
        print("✅ FastMCP 2.0 SDK is available")
    except ImportError:
        print("❌ FastMCP 2.0 SDK is not available")
        print("💡 Install it with: pip install fastmcp")
        print("🔗 GitHub: https://github.com/jlowin/fastmcp")
        return
    
    # Run examples
    await example_fastmcp2_direct_integration()
    await example_fastmcp2_script_integration()
    await example_fastmcp2_multi_server()
    example_fastmcp2_sync_usage()
    await example_fastmcp2_error_handling()
    
    print("🎉 All FastMCP 2.0 examples completed!")
    print()
    print("📚 Key Takeaways:")
    print("1. Use quick_load_fastmcp2_tools() for simple tool loading")
    print("2. Use FastMCP2Client for more control over server connections")
    print("3. FastMCP2MultiClient supports multiple servers")
    print("4. Both async and sync patterns are supported")
    print("5. Always implement error handling for production applications")
    print()
    print("🚀 Ready to build amazing AI applications with FastMCP 2.0 + LangGraph!")


if __name__ == "__main__":
    asyncio.run(main())