"""Example: Trust-gated MCP tool calls with Dominion Observatory.

This example shows how to use TrustGateInterceptor to check MCP server
behavioral trust scores before allowing tool calls. Servers below the
threshold are blocked automatically.

Requirements:
    pip install langchain-mcp-adapters httpx
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.trust_gate import TrustGateInterceptor

# Create a trust gate — blocks calls to servers scoring below 60
trust_gate = TrustGateInterceptor(
    min_trust_score=60,
    # observatory_url can be changed to any trust oracle
    # that returns {"trust_score": <0-100>}
    observatory_url="https://dominion-observatory.sgdata.workers.dev",
)


async def main():
    # Pass the interceptor when creating the client
    async with MultiServerMCPClient(
        {
            "observatory": {
                "url": "https://dominion-observatory.sgdata.workers.dev/mcp",
                "transport": "streamable_http",
            },
        },
        interceptors=[trust_gate],
    ) as client:
        tools = client.get_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        # When the agent invokes a tool, the trust gate checks the server
        # trust score first. If trusted, the call proceeds. If not, the
        # agent receives an explanation of why the call was blocked.


if __name__ == "__main__":
    asyncio.run(main())
