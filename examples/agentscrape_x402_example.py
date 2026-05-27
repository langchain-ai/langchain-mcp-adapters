"""AgentScrape x402 MCP Example.

Connect LangChain to AgentScrape, a pay-per-call web-scraping MCP server that
uses the x402 payment protocol on Base USDC. Agents pay autonomously per call,
no signup or API keys required.

AgentScrape exposes six tools as a remote MCP server via Streamable HTTP:
    - scrape_webpage          ($0.003) - markdown/html/text/json scrape
    - extract_structured_data ($0.005) - AI extraction via Groq + Llama 4 Scout
    - screenshot_webpage      ($0.003) - PNG screenshot with viewport control
    - extract_metadata        ($0.002) - title, OG, Twitter, JSON-LD
    - create_browser_session  ($0.001) - stateful browser session
    - run_workflow            ($0.008) - multi-step atomic workflow up to 20 steps

Free tier: 10 calls per wallet in the first 30 days. Beyond that, payment in
USDC on Base mainnet (eip155:8453). The first 402 Payment Required response
carries the canonical x402 payment requirements header; this example shows the
unpaid (free-tier) usage. For paid usage, supply the X-PAYMENT-RESPONSE header
in the headers kwarg below.

Service URLs:
    - Worker:    https://agent-scrape.healingsunhaven.workers.dev
    - MCP:       https://agent-scrape.healingsunhaven.workers.dev/mcp
    - x402:      https://agent-scrape.healingsunhaven.workers.dev/.well-known/x402.json
    - A2A card:  https://agent-scrape.healingsunhaven.workers.dev/.well-known/agent.json
    - GitHub:    https://github.com/hshintelligence/agent-scrape

Run:
    uv run python examples/agentscrape_x402_example.py
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


async def main() -> None:
    # Connect to AgentScrape's remote MCP endpoint over Streamable HTTP.
    # AgentScrape automatically negotiates Streamable HTTP and falls back to
    # SSE for legacy clients.
    client = MultiServerMCPClient(
        {
            "agentscrape": {
                "url": "https://agent-scrape.healingsunhaven.workers.dev/mcp",
                "transport": "streamable_http",
                # Uncomment and supply a real x402 payment receipt for paid calls:
                # "headers": {
                #     "X-PAYMENT-RESPONSE": "<base64-encoded x402 payment receipt>",
                # },
            }
        }
    )

    # Pull all of AgentScrape's tools as LangChain tools.
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from AgentScrape:")
    for t in tools:
        print(f"  - {t.name}")

    # Build a LangGraph agent that uses these tools.
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(model, tools)

    # Ask the agent to scrape a page.
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Scrape https://www.x402.org and tell me the "
                        "headline plus a 2-sentence summary."
                    ),
                }
            ]
        }
    )

    print("\nAgent response:\n", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
