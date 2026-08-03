import asyncio
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

async def run_stateful_playwright_agent():
    """
    Example demonstrating how to use a stateful MCP server (like Playwright)
    with LangChain agents.

    Stateful servers require a persistent session across multiple tool calls.
    If you use `client.get_tools()`, a new session is created for every tool call,
    which will cause stateful operations (like navigating to a page, then clicking)
    to fail because the browser context is lost between calls.

    Instead, use `client.session()` to keep the connection open for the entire
    agent run.
    """
    
    # 1. Configure the client
    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"],
                "transport": "stdio",
            }
        }
    )

    # Note: Make sure OPENAI_API_KEY is set in your environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable.")
        return

    model = ChatOpenAI(model="gpt-4o")

    # 2. Use a persistent session
    async with client.session("playwright") as session:
        print("Connected to Playwright MCP Server.")
        
        # 3. Load tools using the persistent session
        tools = await load_mcp_tools(session)
        print(f"Loaded {len(tools)} tools.")
        
        # 4. Create the agent
        agent = create_agent(model, tools)
        
        prompt = (
            "Navigate to https://example.com. "
            "Then, capture a snapshot of the page and tell me what the main heading says."
        )
        
        print(f"Running agent with prompt: {prompt}")
        
        # 5. Invoke the agent. The browser context will persist across the multiple 
        # tool calls (browser_navigate, browser_snapshot, etc.) because the 
        # 'async with client.session()' block is keeping the server connection alive.
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        
        print("\nAgent Response:")
        print(response)

if __name__ == "__main__":
    asyncio.run(run_stateful_playwright_agent())
