"""
Global Chat MCP Server — Cross-protocol agent discovery for LangGraph agents.

This example connects a LangGraph ReAct agent to the Global Chat MCP server,
enabling it to search for agents and services across MCP registries,
A2A protocol, and agents.txt endpoints.

Requirements:
    pip install langchain-mcp-adapters langgraph langchain-openai

Usage:
    export OPENAI_API_KEY=<your_api_key>
    python global_chat_example.py
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


async def main():
    async with MultiServerMCPClient(
        {
            "global-chat": {
                "command": "npx",
                "args": ["-y", "@globalchatadsapp/mcp-server"],
                "transport": "stdio",
            }
        }
    ) as client:
        tools = await client.get_tools()

        agent = create_react_agent(model, tools)

        # Example: search for payment-related MCP servers
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Find MCP servers related to payments"}]}
        )

        for message in response["messages"]:
            print(message.pretty_print())


if __name__ == "__main__":
    asyncio.run(main())
