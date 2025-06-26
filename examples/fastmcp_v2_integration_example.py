"""
Complete FastMCP 2.0 SDK Integration Example

This example demonstrates how to integrate the independent FastMCP 2.0 SDK
with LangChain and LangGraph for building powerful AI applications.

FastMCP 2.0 provides advanced features like:
- Direct server instance integration
- Proxying and composition
- Advanced deployment options
- Rich client capabilities
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
        print(f"Created mock agent with model {model_name} and {len(tools)} FastMCP 2.0 tools")
    
    async def ainvoke(self, state):
        messages = state.get("messages", "")
        print(f"Agent processing: {messages}")
        
        # Mock intelligent response based on available tools
        tool_names = [tool.name for tool in self.tools]
        
        if "add" in messages.lower() and "add" in tool_names:
            return {"response": "I can help you add numbers using the FastMCP 2.0 add tool!"}
        elif "calculate" in messages.lower():
            return {"response": f"I have access to these FastMCP 2.0 math tools: {', '.join(tool_names)}"}
        else:
            return {"response": f"I have {len(self.tools)} FastMCP 2.0 tools available to help you!"}

def create_react_agent(model_name: str, tools):
    """Mock function to create a react agent."""
    return MockAgent(model_name, tools)

# Check if FastMCP 2.0 is available
try:
    import fastmcp
    from fastmcp import FastMCP
    FASTMCP_V2_AVAILABLE = True
    print("✅ FastMCP 2.0 SDK is available")
except ImportError:
    FASTMCP_V2_AVAILABLE = False
    print("❌ FastMCP 2.0 SDK not available. Please install with: pip install fastmcp")

# Import our FastMCP 2.0 integration
try:
    from langchain_mcp_adapters import (
        FastMCPv2Client,
        FastMCPv2Adapter,
        FastMCPv2ServerManager,
        quick_load_fastmcp_v2_tools,
        quick_load_fastmcp_v2_tools_sync,
        extract_tools_from_fastmcp_v2_server,
        FASTMCP_V2_AVAILABLE as ADAPTER_AVAILABLE
    )
    print("✅ FastMCP 2.0 adapters are available")
except ImportError as e:
    print(f"❌ FastMCP 2.0 adapters not available: {e}")
    ADAPTER_AVAILABLE = False


async def example_direct_server_integration():
    """Example of direct FastMCP 2.0 server instance integration."""
    print("\n🚀 Direct FastMCP 2.0 Server Integration")
    print("=" * 50)
    
    if not FASTMCP_V2_AVAILABLE or not ADAPTER_AVAILABLE:
        print("❌ FastMCP 2.0 or adapters not available")
        return
    
    try:
        # Create a FastMCP 2.0 server instance
        server = FastMCP("Direct Math Server")
        
        # Add some tools directly
        @server.tool()
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b
        
        @server.tool()
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b
        
        @server.tool()
        async def async_power(base: float, exp: float) -> float:
            """Calculate power (async example)."""
            await asyncio.sleep(0.1)  # Simulate async work
            return base ** exp
        
        print(f"✅ Created FastMCP 2.0 server with tools")
        
        # Extract tools using our adapter
        tools = extract_tools_from_fastmcp_v2_server(server)
        print(f"✅ Extracted {len(tools)} tools from server instance")
        
        # List tool names
        tool_names = [tool.name for tool in tools]
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Create LangGraph agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test the agent
        response = await agent.ainvoke({"messages": "Can you add 5 and 3?"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ Direct server integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in direct server integration: {e}")


async def example_fastmcp_v2_client():
    """Example using FastMCP 2.0 client."""
    print("\n🌐 FastMCP 2.0 Client Integration")
    print("=" * 50)
    
    if not ADAPTER_AVAILABLE:
        print("❌ FastMCP 2.0 adapters not available")
        return
    
    try:
        # Create a client for a FastMCP 2.0 server script
        client = FastMCPv2Client(
            server_script="./fastmcp_v2_math_server.py"
        )
        print("✅ Created FastMCP 2.0 client")
        
        # Get tools from the server
        tools = await client.get_tools()
        print(f"✅ Loaded {len(tools)} tools from FastMCP 2.0 server")
        
        # List tool names
        tool_names = await client.list_tool_names()
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Get a specific tool
        add_tool = await client.get_tool_by_name("add")
        if add_tool:
            print(f"🔧 Found 'add' tool: {add_tool.description}")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test the agent
        response = await agent.ainvoke({"messages": "What mathematical operations can you perform?"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ FastMCP 2.0 client integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in client integration: {e}")


async def example_quick_tool_loading():
    """Example of quick tool loading from FastMCP 2.0."""
    print("\n⚡ Quick FastMCP 2.0 Tool Loading")
    print("=" * 50)
    
    if not ADAPTER_AVAILABLE:
        print("❌ FastMCP 2.0 adapters not available")
        return
    
    try:
        # Quick async loading
        tools = await quick_load_fastmcp_v2_tools(
            server_script="./fastmcp_v2_math_server.py"
        )
        print(f"✅ Quick async load: {len(tools)} tools")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test with multiple queries
        queries = [
            "Calculate 10 + 5",
            "What's the factorial of 5?",
            "Is 17 a prime number?"
        ]
        
        for query in queries:
            response = await agent.ainvoke({"messages": query})
            print(f"🤔 Query: {query}")
            print(f"🤖 Response: {response['response']}")
        
        print("✅ Quick loading completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in quick loading: {e}")


def example_sync_integration():
    """Example of synchronous FastMCP 2.0 integration."""
    print("\n🔄 Synchronous FastMCP 2.0 Integration")
    print("=" * 50)
    
    if not ADAPTER_AVAILABLE:
        print("❌ FastMCP 2.0 adapters not available")
        return
    
    try:
        # Synchronous tool loading
        tools = quick_load_fastmcp_v2_tools_sync(
            server_script="./fastmcp_v2_math_server.py"
        )
        print(f"✅ Synchronously loaded {len(tools)} tools")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        print("🤖 Agent created and ready for use!")
        
        print("✅ Synchronous integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in sync integration: {e}")


async def example_server_manager():
    """Example using FastMCP 2.0 server manager."""
    print("\n🏗️ FastMCP 2.0 Server Manager")
    print("=" * 50)
    
    if not FASTMCP_V2_AVAILABLE or not ADAPTER_AVAILABLE:
        print("❌ FastMCP 2.0 or adapters not available")
        return
    
    try:
        # Create server manager
        manager = FastMCPv2ServerManager()
        
        # Create and add multiple servers
        math_server = FastMCP("Math Server")
        
        @math_server.tool()
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b
        
        @math_server.tool()
        def subtract(a: float, b: float) -> float:
            """Subtract two numbers."""
            return a - b
        
        # Add server to manager
        manager.add_server("math", math_server)
        
        # Create another server
        string_server = FastMCP("String Server")
        
        @string_server.tool()
        def uppercase(text: str) -> str:
            """Convert text to uppercase."""
            return text.upper()
        
        @string_server.tool()
        def reverse(text: str) -> str:
            """Reverse text."""
            return text[::-1]
        
        manager.add_server("string", string_server)
        
        print(f"✅ Created server manager with 2 servers")
        
        # Get all tools
        all_tools = await manager.get_all_tools()
        total_tools = sum(len(tools) for tools in all_tools.values())
        print(f"✅ Total tools across all servers: {total_tools}")
        
        # Get tools from specific server
        math_tools = await manager.get_tools_from_server("math")
        print(f"🧮 Math server tools: {len(math_tools)}")
        
        # Create client for specific server
        math_client = manager.create_client_for_server("math")
        client_tools = await math_client.get_tools()
        print(f"🔧 Client tools: {len(client_tools)}")
        
        print("✅ Server manager example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in server manager: {e}")


async def example_advanced_features():
    """Example of advanced FastMCP 2.0 features."""
    print("\n🌟 Advanced FastMCP 2.0 Features")
    print("=" * 50)
    
    if not FASTMCP_V2_AVAILABLE or not ADAPTER_AVAILABLE:
        print("❌ FastMCP 2.0 or adapters not available")
        return
    
    try:
        # Create server with resources and prompts
        server = FastMCP("Advanced Server")
        
        @server.tool()
        def calculate(expression: str) -> str:
            """Safely evaluate mathematical expressions."""
            try:
                # Simple expression evaluator (demo only)
                import re
                if re.match(r'^[0-9+\-*/().\s]+$', expression):
                    result = eval(expression, {"__builtins__": {}}, {})
                    return f"{expression} = {result}"
                else:
                    return "Invalid expression"
            except:
                return "Error in calculation"
        
        @server.resource("math://constants")
        def get_constants():
            """Get mathematical constants."""
            import math
            return {
                "pi": math.pi,
                "e": math.e,
                "golden_ratio": (1 + 5**0.5) / 2
            }
        
        @server.prompt()
        def math_help(topic: str) -> str:
            """Generate math help prompt."""
            return f"Please explain {topic} in simple terms with examples."
        
        print("✅ Created advanced FastMCP 2.0 server with tools, resources, and prompts")
        
        # Extract tools
        tools = extract_tools_from_fastmcp_v2_server(server)
        print(f"✅ Extracted {len(tools)} tools")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test advanced calculation
        response = await agent.ainvoke({"messages": "Can you calculate (5 + 3) * 2?"})
        print(f"🤖 Agent response: {response['response']}")
        
        print("✅ Advanced features example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in advanced features: {e}")


async def main():
    """Run all FastMCP 2.0 integration examples."""
    print("🎯 FastMCP 2.0 SDK Integration Examples")
    print("=" * 60)
    print("These examples show how to integrate the independent FastMCP 2.0 SDK")
    print("with LangChain and LangGraph for building powerful AI applications.")
    print("=" * 60)
    
    # Check availability
    if not FASTMCP_V2_AVAILABLE:
        print("⚠️ FastMCP 2.0 SDK not available. Install with: pip install fastmcp")
        print("📝 Note: This is the independent FastMCP 2.0 SDK, not the MCP SDK's fastmcp module")
    
    if not ADAPTER_AVAILABLE:
        print("⚠️ FastMCP 2.0 adapters not available in langchain-mcp-adapters")
    
    if FASTMCP_V2_AVAILABLE and ADAPTER_AVAILABLE:
        # Run all examples
        await example_direct_server_integration()
        await example_fastmcp_v2_client()
        await example_quick_tool_loading()
        example_sync_integration()
        await example_server_manager()
        await example_advanced_features()
        
        print("\n🎉 All FastMCP 2.0 integration examples completed!")
        
        print("\n📚 Key Features Demonstrated:")
        print("1. Direct server instance integration")
        print("2. Client-based server connections")
        print("3. Quick tool loading utilities")
        print("4. Synchronous and asynchronous patterns")
        print("5. Server management capabilities")
        print("6. Advanced FastMCP 2.0 features (tools, resources, prompts)")
        
        print("\n🚀 Next Steps:")
        print("1. Install FastMCP 2.0: pip install fastmcp")
        print("2. Create your own FastMCP 2.0 servers")
        print("3. Use the adapters to integrate with LangGraph")
        print("4. Build powerful AI applications!")
    else:
        print("\n💡 To run these examples:")
        print("1. Install FastMCP 2.0: pip install fastmcp")
        print("2. Ensure langchain-mcp-adapters has FastMCP 2.0 support")
        print("3. Run this script again")


if __name__ == "__main__":
    asyncio.run(main())