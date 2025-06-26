#!/usr/bin/env python3
"""
Demo script for FastMCP integration with LangGraph.

This script demonstrates how to use the new FastMCP integration features
to seamlessly connect FastMCP servers with LangGraph agents.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from langchain_mcp_adapters import (
    FastMCPClient,
    FastMCPMultiClient,
    quick_load_fastmcp_tools,
    quick_load_fastmcp_tools_sync,
)


async def demo_basic_fastmcp_integration():
    """Demo basic FastMCP integration."""
    print("🚀 Demo: Basic FastMCP Integration")
    print("=" * 50)
    
    try:
        # Create a FastMCP client
        client = FastMCPClient(server_script="./examples/fastmcp_math_server.py")
        print(f"✅ Created FastMCP client for math server")
        
        # Get tools from the server
        tools = await client.get_tools()
        print(f"✅ Loaded {len(tools)} tools from FastMCP server")
        
        # List tool names
        tool_names = await client.list_tool_names()
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Get a specific tool
        add_tool = await client.get_tool_by_name("add")
        if add_tool:
            print(f"🔧 Found 'add' tool: {add_tool.description}")
        
        print("✅ Basic integration demo completed successfully!\n")
        
    except Exception as e:
        print(f"❌ Error in basic demo: {e}")
        print("💡 Make sure the FastMCP math server is available\n")


async def demo_quick_tool_loading():
    """Demo quick tool loading functions."""
    print("⚡ Demo: Quick Tool Loading")
    print("=" * 50)
    
    try:
        # Quick async loading
        tools = await quick_load_fastmcp_tools(
            server_script="./examples/fastmcp_math_server.py"
        )
        print(f"✅ Quick async load: {len(tools)} tools")
        
        # Quick sync loading
        tools_sync = quick_load_fastmcp_tools_sync(
            server_script="./examples/fastmcp_math_server.py"
        )
        print(f"✅ Quick sync load: {len(tools_sync)} tools")
        
        print("✅ Quick loading demo completed successfully!\n")
        
    except Exception as e:
        print(f"❌ Error in quick loading demo: {e}")
        print("💡 Make sure the FastMCP math server is available\n")


async def demo_multi_server_integration():
    """Demo multi-server integration."""
    print("🌐 Demo: Multi-Server Integration")
    print("=" * 50)
    
    try:
        # Configure multiple servers
        servers = {
            "math": {
                "server_script": "./examples/fastmcp_math_server.py"
            },
            # Note: This would require a running HTTP server
            # "weather": {
            #     "server_url": "http://localhost:8000/mcp/"
            # }
        }
        
        # Create multi-client
        multi_client = FastMCPMultiClient(servers)
        print(f"✅ Created multi-client with {len(servers)} servers")
        
        # Get tools from all servers
        all_tools = await multi_client.get_tools_flat()
        print(f"✅ Loaded {len(all_tools)} tools from all servers")
        
        # Get tools from specific server
        math_tools = await multi_client.get_tools_from_server("math")
        print(f"🧮 Math server has {len(math_tools)} tools")
        
        print("✅ Multi-server demo completed successfully!\n")
        
    except Exception as e:
        print(f"❌ Error in multi-server demo: {e}")
        print("💡 Make sure the FastMCP math server is available\n")


async def demo_tool_execution():
    """Demo tool execution (mock)."""
    print("🔧 Demo: Tool Execution (Mock)")
    print("=" * 50)
    
    try:
        # Load tools
        tools = await quick_load_fastmcp_tools(
            server_script="./examples/fastmcp_math_server.py"
        )
        
        if tools:
            print(f"✅ Loaded {len(tools)} tools")
            
            # Find the add tool
            add_tool = None
            for tool in tools:
                if tool.name == "add":
                    add_tool = tool
                    break
            
            if add_tool:
                print(f"🔧 Found add tool: {add_tool.description}")
                print("💡 Tool is ready for LangGraph integration")
                
                # Show tool schema
                if hasattr(add_tool, 'args_schema'):
                    schema = add_tool.args_schema.model_json_schema()
                    print(f"📋 Tool schema: {schema}")
            else:
                print("❌ Add tool not found")
        else:
            print("❌ No tools loaded")
            
        print("✅ Tool execution demo completed!\n")
        
    except Exception as e:
        print(f"❌ Error in tool execution demo: {e}")
        print("💡 Make sure the FastMCP math server is available\n")


def demo_sync_usage():
    """Demo synchronous usage."""
    print("🔄 Demo: Synchronous Usage")
    print("=" * 50)
    
    try:
        # Create client
        client = FastMCPClient(server_script="./examples/fastmcp_math_server.py")
        
        # Use sync methods
        tools = client.get_tools_sync()
        print(f"✅ Sync loaded {len(tools)} tools")
        
        tool_names = client.list_tool_names_sync()
        print(f"📋 Tool names: {', '.join(tool_names)}")
        
        add_tool = client.get_tool_by_name_sync("add")
        if add_tool:
            print(f"🔧 Found add tool: {add_tool.description}")
        
        print("✅ Synchronous usage demo completed!\n")
        
    except Exception as e:
        print(f"❌ Error in sync demo: {e}")
        print("💡 Make sure the FastMCP math server is available\n")


async def main():
    """Run all demos."""
    print("🎯 FastMCP Integration Demo")
    print("=" * 60)
    print("This demo shows how to integrate FastMCP servers with LangGraph")
    print("=" * 60)
    print()
    
    # Run async demos
    await demo_basic_fastmcp_integration()
    await demo_quick_tool_loading()
    await demo_multi_server_integration()
    await demo_tool_execution()
    
    # Run sync demo
    demo_sync_usage()
    
    print("🎉 All demos completed!")
    print()
    print("Next steps:")
    print("1. Create your own FastMCP server")
    print("2. Use quick_load_fastmcp_tools() to load tools")
    print("3. Create LangGraph agents with the tools")
    print("4. Build amazing AI applications!")
    print()
    print("See FASTMCP_INTEGRATION.md for detailed documentation.")


if __name__ == "__main__":
    asyncio.run(main())