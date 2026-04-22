"""
Sylex Search MCP Agent Example
===============================

This example demonstrates how to use the Sylex Search MCP server with a
LangGraph ReAct agent to discover, compare, and evaluate software products.

Sylex Search is a hosted MCP server with 11,000+ indexed tools, libraries,
APIs, and SaaS products. No authentication or local setup required.

Requirements:
    pip install langchain-mcp-adapters langgraph "langchain[openai]"
    export OPENAI_API_KEY=<your_api_key>
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Sylex Search hosted SSE endpoint — no auth required
SYLEX_SEARCH_URL = "https://mcp-server-production-38c9.up.railway.app/sse"


async def run_agent(query: str) -> str:
    """Run the agent with the given query and return its response."""
    client = MultiServerMCPClient(
        {
            "sylex": {
                "url": SYLEX_SEARCH_URL,
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent("openai:gpt-4.1", tools)
    response = await agent.ainvoke({"messages": query})
    # The last message in the response contains the final answer
    return response["messages"][-1].content


async def main() -> None:
    # ------------------------------------------------------------------ #
    # Example 1: Discover products by keyword                             #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Example 1: Product Discovery")
    print("=" * 60)
    result = await run_agent(
        "Find me a good Python HTTP client library. "
        "Search for 'python http client' and tell me the top results."
    )
    print(result)

    # ------------------------------------------------------------------ #
    # Example 2: Get detailed information about a specific product        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Example 2: Product Details")
    print("=" * 60)
    result = await run_agent(
        "Search for 'httpx' and then get the full details for the httpx package. "
        "What are its main features and how popular is it?"
    )
    print(result)

    # ------------------------------------------------------------------ #
    # Example 3: Compare multiple products                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Example 3: Product Comparison")
    print("=" * 60)
    result = await run_agent(
        "I need to choose a Python HTTP client. "
        "Compare 'requests', 'httpx', and 'aiohttp'. "
        "First search for each one to get their IDs, then use the compare tool "
        "to give me a side-by-side comparison. "
        "Which one would you recommend for an async application?"
    )
    print(result)

    # ------------------------------------------------------------------ #
    # Example 4: Find alternatives                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Example 4: Find Alternatives")
    print("=" * 60)
    result = await run_agent(
        "I'm currently using 'pandas' for data manipulation in Python. "
        "Search for pandas to find its product ID, then use the alternatives tool "
        "to discover what other options exist. "
        "Summarize the top 3 alternatives."
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
