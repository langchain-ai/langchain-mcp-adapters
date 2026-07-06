"""Connect to the remote BGPT MCP server via streamable HTTP.

BGPT exposes a hosted MCP endpoint for scientific paper search with structured
evidence fields (methods, limitations, conflicts of interest, falsifiability).

Endpoints:
- MCP (streamable HTTP): https://bgpt.pro/mcp/stream
- REST search: POST https://bgpt.pro/api/mcp-search
- REST DOI lookup: POST https://bgpt.pro/api/mcp-doi-lookup

Free tier works without an API key (50 results per network).

Usage:
    pip install langchain-mcp-adapters mcp

    # List available BGPT MCP tools (no API key required)
    python bgpt_mcp_client.py

    # Run a sample search (requires OPENAI_API_KEY)
    export OPENAI_API_KEY=...
    python bgpt_mcp_client.py --query "GLP-1 alcohol craving"
"""

from __future__ import annotations

import argparse
import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

BGPT_MCP_URL = "https://bgpt.pro/mcp/stream"


async def list_bgpt_tools() -> list:
    """Load tools from the remote BGPT MCP server."""
    client = MultiServerMCPClient(
        {
            "bgpt": {
                "transport": "streamable_http",
                "url": BGPT_MCP_URL,
            }
        }
    )
    return await client.get_tools()


async def run_agent(query: str) -> None:
    """Run a LangChain agent with BGPT MCP tools."""
    from langchain.agents import create_agent

    tools = await list_bgpt_tools()
    print(f"Loaded {len(tools)} BGPT tool(s):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")

    agent = create_agent("openai:gpt-4.1-mini", tools)
    response = await agent.ainvoke({"messages": query})
    print("\nAgent response:")
    print(response["messages"][-1].content)


async def main() -> None:
    parser = argparse.ArgumentParser(description="BGPT remote MCP client example")
    parser.add_argument(
        "--query",
        help="Optional search query to run through a LangChain agent",
    )
    args = parser.parse_args()

    if args.query:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY to run an agent query.")
        await run_agent(args.query)
    else:
        tools = await list_bgpt_tools()
        print(f"Connected to {BGPT_MCP_URL}")
        print(f"Available tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}")


if __name__ == "__main__":
    asyncio.run(main())
