"""
Chart Library MCP + LangChain: Financial Pattern Analysis

This example shows how to connect to the Chart Library MCP server
and use LangChain/LangGraph to build a financial pattern analysis agent.

Chart Library provides 24M+ historical chart pattern embeddings across
19K+ symbols. The agent can search for similar patterns, get forward
return predictions, detect chart formations, and analyze market regimes.

Setup:
    pip install langchain-mcp-adapters langgraph "langchain[anthropic]" chartlibrary-mcp

    # Get a free API key at https://chartlibrary.io/developers
    export CHART_LIBRARY_API_KEY=your_api_key
    export ANTHROPIC_API_KEY=your_api_key

Usage:
    python examples/chart_library_finance.py
"""

import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


async def main():
    # Connect to Chart Library MCP server via stdio
    # The server exposes 19 tools for pattern search, market intelligence,
    # and trading analysis
    client = MultiServerMCPClient(
        {
            "chart-library": {
                "command": "python",
                "args": ["-m", "chartlibrary_mcp"],
                "transport": "stdio",
                "env": {
                    "CHART_LIBRARY_API_KEY": os.environ["CHART_LIBRARY_API_KEY"],
                },
            },
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from Chart Library MCP")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")

    # Create a ReAct agent with Claude
    agent = create_agent("anthropic:claude-sonnet-4-20250514", tools)

    # Example 1: Search for similar chart patterns
    print("\n--- Pattern Search ---")
    response = await agent.ainvoke(
        {
            "messages": (
                "Search for chart patterns similar to NVDA on 2025-01-15. "
                "What do the historical matches suggest happens next? "
                "Include the forward return predictions."
            )
        }
    )
    print(response["messages"][-1].content)

    # Example 2: Market regime analysis
    print("\n--- Regime Analysis ---")
    response = await agent.ainvoke(
        {
            "messages": (
                "What is the current market regime? "
                "How do pattern win rates differ in this regime vs others?"
            )
        }
    )
    print(response["messages"][-1].content)

    # Example 3: Pattern detection
    print("\n--- Pattern Detection ---")
    response = await agent.ainvoke(
        {
            "messages": (
                "Check AAPL for any active chart patterns like breakouts, "
                "bull flags, or ascending wedges. If you find any, simulate "
                "a trade with a 2% stop loss and 5% profit target."
            )
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
