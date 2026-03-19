"""
x402 payments for LangGraph agents via agentpay-mcp

Any LangGraph agent can pay for x402-protected API endpoints automatically.
agentpay-mcp handles the 402 detect -> sign -> retry loop — no custom
payment code needed in the agent.

Setup:
    pip install langchain-mcp-adapters langgraph langchain-anthropic
    npm install -g agentpay-mcp

    export AGENT_PRIVATE_KEY=0x...
    export AGENT_WALLET_ADDRESS=0x...
    export ANTHROPIC_API_KEY=...
"""

import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic


async def main():
    # Connect agentpay-mcp as an MCP server — runs as a subprocess, no browser
    client = MultiServerMCPClient(
        {
            "payments": {
                "command": "npx",
                "args": ["agentpay-mcp"],
                "env": {
                    "AGENT_PRIVATE_KEY": os.environ["AGENT_PRIVATE_KEY"],
                    "AGENT_WALLET_ADDRESS": os.environ["AGENT_WALLET_ADDRESS"],
                    # Optional: Base Sepolia testnet (84532) or mainnet (8453)
                    "CHAIN_ID": os.environ.get("CHAIN_ID", "84532"),
                },
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    # Agent now has pay_x402_endpoint available as a tool

    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    agent = create_react_agent(model, tools)

    # When the agent calls an x402-protected endpoint and gets a 402,
    # it calls pay_x402_endpoint automatically — signs the USDC transfer,
    # submits payment to the facilitator, retries the original request.
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call https://api.example.com/data — it requires x402 payment. "
                        "Use the pay_x402_endpoint tool if you get a 402 response."
                    ),
                }
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
