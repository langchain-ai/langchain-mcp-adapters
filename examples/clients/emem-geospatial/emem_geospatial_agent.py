"""emem geospatial MCP client example.

Connects to the live emem MCP server over Streamable HTTP, auto-discovers
all tools, and runs a LangGraph ReAct agent that answers a geospatial
verification question about coordinates 60.3172, 24.9633.

Install:
    pip install langchain-mcp-adapters langgraph langchain-openai

Usage:
    export OPENAI_API_KEY="sk-..."
    python emem_geospatial_agent.py
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

EMEM_MCP_URL = "https://emem.dev/mcp"

QUESTION = (
    "I have coordinates 60.3172, 24.9633. "
    "Locate this place, recall its geospatial facts (elevation, "
    "land cover, vegetation, surface water), and summarise what "
    "you find. Cite the signed receipt fact_cids in your answer."
)


async def main():
    llm = ChatOpenAI(model="gpt-4.1-mini")

    async with MultiServerMCPClient(
        {
            "emem": {
                "transport": "streamable_http",
                "url": EMEM_MCP_URL,
            }
        }
    ) as client:
        tools = client.get_tools()
        print(f"Loaded {len(tools)} emem tools via MCP\n")

        agent = create_react_agent(llm, tools)

        response = await agent.ainvoke({"messages": [("user", QUESTION)]})

        for msg in response["messages"]:
            if hasattr(msg, "content") and msg.content:
                print(f"[{msg.type}] {msg.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
