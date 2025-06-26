"""
FastMCP Integration Examples

This file demonstrates how to use the FastMCP adapters with LangGraph.
"""

import asyncio
from langchain_mcp_adapters import (
    FastMCPClient,
    FastMCPMultiClient,
    quick_load_fastmcp_tools,
    quick_load_fastmcp_tools_sync,
)
from langgraph.prebuilt import create_react_agent


# Example 1: Simple FastMCP server integration
async def example_simple_fastmcp_integration():
    """Example of connecting to a single FastMCP server."""
    
    # Create a FastMCP client pointing to your server script
    client = FastMCPClient(server_script="./examples/math_server.py")
    
    # Get tools from the FastMCP server
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from FastMCP server")
    
    # List available tools
    tool_names = await client.list_tool_names()
    print(f"Available tools: {tool_names}")
    
    # Create a LangGraph agent with the tools
    agent = create_react_agent("openai:gpt-4", tools)
    
    # Use the agent
    response = await agent.ainvoke({"messages": "What's 5 + 3?"})
    print(f"Agent response: {response}")


# Example 2: Multiple FastMCP servers
async def example_multiple_fastmcp_servers():
    """Example of connecting to multiple FastMCP servers."""
    
    # Configure multiple servers
    servers = {
        "math": {
            "server_script": "./examples/math_server.py"
        },
        "weather": {
            "server_url": "http://localhost:8000/mcp/"
        }
    }
    
    # Create multi-client
    multi_client = FastMCPMultiClient(servers)
    
    # Get all tools from all servers
    all_tools = await multi_client.get_tools_flat()
    print(f"Loaded {len(all_tools)} tools from all servers")
    
    # Get tools from specific server
    math_tools = await multi_client.get_tools_from_server("math")
    print(f"Math server has {len(math_tools)} tools")
    
    # Create agent with all tools
    agent = create_react_agent("openai:gpt-4", all_tools)
    
    # Use the agent
    response = await agent.ainvoke({
        "messages": "Calculate 10 * 5 and tell me the weather in New York"
    })
    print(f"Agent response: {response}")


# Example 3: Quick tool loading (synchronous)
def example_sync_tool_loading():
    """Example of synchronously loading tools."""
    
    # Quick load tools synchronously
    tools = quick_load_fastmcp_tools_sync(
        server_script="./examples/math_server.py"
    )
    
    print(f"Synchronously loaded {len(tools)} tools")
    
    # Create agent
    agent = create_react_agent("openai:gpt-4", tools)
    
    # Note: For async agent usage, you'd still need to use asyncio.run()
    # or run within an async context


# Example 4: HTTP server integration
async def example_http_server_integration():
    """Example of connecting to FastMCP server via HTTP."""
    
    # Connect to HTTP server
    tools = await quick_load_fastmcp_tools(
        server_url="http://localhost:8000/mcp/"
    )
    
    print(f"Loaded {len(tools)} tools from HTTP server")
    
    # Create agent
    agent = create_react_agent("openai:gpt-4", tools)
    
    # Use the agent
    response = await agent.ainvoke({"messages": "Help me with calculations"})
    print(f"Agent response: {response}")


# Example 5: Custom server configuration
async def example_custom_server_config():
    """Example with custom server configuration."""
    
    # Custom configuration
    client = FastMCPClient(
        server_command="python",
        server_args=["./examples/custom_server.py", "--port", "9000"],
        # Additional kwargs can be passed for connection configuration
    )
    
    # Get specific tool
    calculator_tool = await client.get_tool_by_name("calculator")
    if calculator_tool:
        print(f"Found calculator tool: {calculator_tool.description}")
    
    # Get all tools
    tools = await client.get_tools()
    
    # Create agent
    agent = create_react_agent("openai:gpt-4", tools)


# Example 6: Error handling
async def example_error_handling():
    """Example with proper error handling."""
    
    try:
        client = FastMCPClient(server_script="./nonexistent_server.py")
        tools = await client.get_tools()
    except Exception as e:
        print(f"Error connecting to server: {e}")
        # Fallback to another server or handle gracefully
        tools = []
    
    if tools:
        agent = create_react_agent("openai:gpt-4", tools)
    else:
        print("No tools available, creating agent without tools")
        agent = create_react_agent("openai:gpt-4", [])


# Example 7: Integration with LangGraph StateGraph
async def example_langgraph_stategraph_integration():
    """Example of using FastMCP tools with LangGraph StateGraph."""
    
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain.chat_models import init_chat_model
    
    # Load tools from FastMCP server
    tools = await quick_load_fastmcp_tools(
        server_script="./examples/math_server.py"
    )
    
    # Initialize model
    model = init_chat_model("openai:gpt-4")
    
    # Define the call_model function
    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}
    
    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    
    # Compile the graph
    graph = builder.compile()
    
    # Use the graph
    response = await graph.ainvoke({"messages": "Calculate 15 * 8"})
    print(f"Graph response: {response}")


if __name__ == "__main__":
    # Run examples
    print("Running FastMCP integration examples...")
    
    # Run async examples
    asyncio.run(example_simple_fastmcp_integration())
    asyncio.run(example_multiple_fastmcp_servers())
    asyncio.run(example_http_server_integration())
    asyncio.run(example_custom_server_config())
    asyncio.run(example_error_handling())
    asyncio.run(example_langgraph_stategraph_integration())
    
    # Run sync example
    example_sync_tool_loading()
    
    print("All examples completed!")