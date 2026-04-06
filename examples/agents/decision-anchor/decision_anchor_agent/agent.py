"""LangGraph ReAct agent with Decision Anchor MCP integration.

Connects to Decision Anchor's remote MCP server to create
external accountability records for agent actions.

Use case: an agent that makes API calls involving payments
records each decision boundary externally via DA, so that
payment disputes can reference externally-anchored proof
rather than self-reported logs.
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")


async def main() -> None:
    """Run the Decision Anchor integration example."""
    async with MultiServerMCPClient(
        {
            "decision-anchor": {
                "url": "https://mcp.decision-anchor.com/mcp",
                "transport": "streamable_http",
            }
        }
    ) as client:
        tools = client.get_tools()
        agent = create_react_agent(model, tools)

        # Register as an agent (first time only, no API key needed)
        result = await agent.ainvoke(
            {
                "messages": [
                    "Register me as a new agent on Decision Anchor. "
                    "Use agent name 'langgraph-demo-agent'."
                ]
            }
        )
        print("Registration:", result["messages"][-1].content)

        # Create a DD (Decision Declaration) — records the
        # accountability boundary for a payment action
        result = await agent.ainvoke(
            {
                "messages": [
                    "Create a Decision Declaration on Decision Anchor. "
                    "Action type: execute. "
                    "Summary: 'Authorized API payment of $12.50 to "
                    "vendor-xyz for data enrichment service'. "
                    "Use retention short, integrity basic, "
                    "disclosure internal, responsibility minimal."
                ]
            }
        )
        print("DD Created:", result["messages"][-1].content)

        # Check DAC balance
        result = await agent.ainvoke(
            {"messages": ["Check my DAC balance on Decision Anchor."]}
        )
        print("Balance:", result["messages"][-1].content)


def main_sync() -> None:
    """Synchronous wrapper for the async main function."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
