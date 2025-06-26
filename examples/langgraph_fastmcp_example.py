"""
Complete example of using FastMCP with LangGraph.

This example demonstrates how to create a complete LangGraph application
that uses FastMCP tools for mathematical operations.
"""

import asyncio
import os
from typing import Dict, Any

# Mock LangGraph components for demonstration
# In a real application, you would import these from langgraph
class MockMessagesState:
    def __init__(self, messages=None):
        self.messages = messages or []

class MockAgent:
    def __init__(self, model_name: str, tools):
        self.model_name = model_name
        self.tools = tools
        print(f"Created mock agent with model {model_name} and {len(tools)} tools")
    
    async def ainvoke(self, state: Dict[str, Any]):
        messages = state.get("messages", "")
        print(f"Agent processing: {messages}")
        
        # Mock response based on the message content
        if "add" in messages.lower() or "+" in messages:
            return {"response": "I can help you add numbers using the add tool!"}
        elif "multiply" in messages.lower() or "*" in messages:
            return {"response": "I can help you multiply numbers using the multiply tool!"}
        elif "calculate" in messages.lower():
            return {"response": "I have access to various math tools: add, subtract, multiply, divide, power, square_root, factorial, fibonacci, is_prime, gcd, lcm"}
        else:
            return {"response": f"I have {len(self.tools)} math tools available to help you!"}

def create_react_agent(model_name: str, tools):
    """Mock function to create a react agent."""
    return MockAgent(model_name, tools)

# Import our FastMCP integration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_mcp_adapters import (
    FastMCPClient,
    quick_load_fastmcp_tools,
    quick_load_fastmcp_tools_sync,
)


async def example_simple_langgraph_integration():
    """Simple example of LangGraph + FastMCP integration."""
    print("🚀 Simple LangGraph + FastMCP Integration")
    print("=" * 50)
    
    try:
        # Load tools from FastMCP server
        tools = await quick_load_fastmcp_tools(
            server_script="./fastmcp_math_server.py"
        )
        print(f"✅ Loaded {len(tools)} tools from FastMCP server")
        
        # Create LangGraph agent with FastMCP tools
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Test the agent with different queries
        test_queries = [
            "What's 5 + 3?",
            "Can you multiply 7 by 9?",
            "Help me with calculations",
            "What tools do you have?"
        ]
        
        for query in test_queries:
            print(f"\n🤔 Query: {query}")
            response = await agent.ainvoke({"messages": query})
            print(f"🤖 Response: {response['response']}")
        
        print("\n✅ Simple integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_advanced_langgraph_integration():
    """Advanced example with multiple servers and custom configuration."""
    print("\n🌟 Advanced LangGraph + FastMCP Integration")
    print("=" * 50)
    
    try:
        # Create FastMCP client with custom configuration
        client = FastMCPClient(
            server_script="./fastmcp_math_server.py",
            server_command="python",
        )
        
        # Get tools and inspect them
        tools = await client.get_tools()
        print(f"✅ Loaded {len(tools)} tools")
        
        # List available tools
        tool_names = await client.list_tool_names()
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Get specific tools
        add_tool = await client.get_tool_by_name("add")
        multiply_tool = await client.get_tool_by_name("multiply")
        
        if add_tool and multiply_tool:
            print(f"🔧 Add tool: {add_tool.description}")
            print(f"🔧 Multiply tool: {multiply_tool.description}")
            
            # Create agent with specific tools
            selected_tools = [add_tool, multiply_tool]
            agent = create_react_agent("openai:gpt-4", selected_tools)
            
            # Test with mathematical queries
            response = await agent.ainvoke({
                "messages": "I need to add some numbers and multiply others"
            })
            print(f"🤖 Agent response: {response['response']}")
        
        print("\n✅ Advanced integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_sync_langgraph_integration():
    """Example using synchronous tool loading."""
    print("\n🔄 Synchronous LangGraph + FastMCP Integration")
    print("=" * 50)
    
    try:
        # Load tools synchronously
        tools = quick_load_fastmcp_tools_sync(
            server_script="./fastmcp_math_server.py"
        )
        print(f"✅ Synchronously loaded {len(tools)} tools")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Note: Agent execution would still be async in real LangGraph
        print("🤖 Agent created and ready for use!")
        print("💡 In real usage, you'd still use await agent.ainvoke() for execution")
        
        print("\n✅ Synchronous integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_error_handling():
    """Example with proper error handling."""
    print("\n🛡️ Error Handling Example")
    print("=" * 50)
    
    # Try to connect to non-existent server
    try:
        tools = await quick_load_fastmcp_tools(
            server_script="./nonexistent_server.py"
        )
        print(f"Loaded {len(tools)} tools")
    except Exception as e:
        print(f"❌ Expected error connecting to non-existent server: {type(e).__name__}")
        print("💡 Falling back to empty tools list")
        tools = []
    
    # Create agent with fallback
    if tools:
        agent = create_react_agent("openai:gpt-4", tools)
        print("✅ Agent created with FastMCP tools")
    else:
        agent = create_react_agent("openai:gpt-4", [])
        print("✅ Agent created without tools (fallback mode)")
    
    print("✅ Error handling example completed!")


async def example_real_world_usage():
    """Example showing real-world usage patterns."""
    print("\n🌍 Real-World Usage Example")
    print("=" * 50)
    
    try:
        # Configuration that might come from environment or config file
        server_config = {
            "server_script": "./fastmcp_math_server.py",
            # In real usage, you might have:
            # "server_url": os.getenv("FASTMCP_SERVER_URL"),
            # "auth_token": os.getenv("FASTMCP_AUTH_TOKEN"),
        }
        
        # Load tools with error handling
        tools = []
        try:
            tools = await quick_load_fastmcp_tools(**server_config)
            print(f"✅ Loaded {len(tools)} tools from FastMCP server")
        except Exception as e:
            print(f"⚠️ Could not load FastMCP tools: {e}")
            print("💡 Continuing with built-in tools only")
        
        # Create agent
        agent = create_react_agent("openai:gpt-4", tools)
        
        # Simulate real application usage
        user_queries = [
            "Calculate the factorial of 5",
            "What's the 10th Fibonacci number?",
            "Is 17 a prime number?",
            "Find the GCD of 48 and 18"
        ]
        
        for query in user_queries:
            print(f"\n👤 User: {query}")
            try:
                response = await agent.ainvoke({"messages": query})
                print(f"🤖 Assistant: {response['response']}")
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        print("\n✅ Real-world usage example completed!")
        
    except Exception as e:
        print(f"❌ Error in real-world example: {e}")


async def main():
    """Run all examples."""
    print("🎯 LangGraph + FastMCP Integration Examples")
    print("=" * 60)
    print("These examples show how to integrate FastMCP servers with LangGraph agents")
    print("=" * 60)
    
    # Run examples
    await example_simple_langgraph_integration()
    await example_advanced_langgraph_integration()
    example_sync_langgraph_integration()
    await example_error_handling()
    await example_real_world_usage()
    
    print("\n🎉 All examples completed!")
    print("\n📚 Key Takeaways:")
    print("1. Use quick_load_fastmcp_tools() for simple tool loading")
    print("2. Use FastMCPClient for more control over server connections")
    print("3. Always implement error handling for production applications")
    print("4. FastMCP tools work seamlessly with LangGraph agents")
    print("5. Both async and sync patterns are supported")
    
    print("\n🚀 Ready to build amazing AI applications with FastMCP + LangGraph!")


if __name__ == "__main__":
    asyncio.run(main())