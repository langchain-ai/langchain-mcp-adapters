# TWZRD Agent Intel — Remote Streamable-HTTP MCP Example

This example demonstrates how to connect a LangGraph agent to [TWZRD Agent Intel](https://intel.twzrd.xyz), a production remote MCP server that trust-scores Solana wallets before making HTTP 402 micropayments.

No local server required — TWZRD Agent Intel runs at `https://intel.twzrd.xyz/mcp` and exposes three tools:

| Tool | Cost | Description |
|------|------|-------------|
| `score_agent` | Free | On-chain trust score (0–100) for a Solana wallet |
| `preflight_check` | Free | Boolean readiness check — is this wallet safe to pay? |
| `get_trust_receipt` | Paid (x402) | Signed `twzrd.receipt.v5` receipt via USDC micropayment |

## Use case

Before an agent sends a USDC micropayment to an unknown Solana wallet address, call `score_agent` or `preflight_check` to verify the recipient's on-chain reputation.

## Installation

```bash
pip install langchain-mcp-adapters langgraph langchain-openai
export OPENAI_API_KEY=<your_api_key>
```

## Usage

### Quick check — trust-score a wallet before paying

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def check_wallet_before_paying(wallet: str):
    async with MultiServerMCPClient({
        "twzrd": {
            "url": "https://intel.twzrd.xyz/mcp",
            "transport": "streamable_http",
        }
    }) as client:
        tools = client.get_tools()
        model = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(model, tools)

        response = await agent.ainvoke({
            "messages": f"Should I send a USDC payment to {wallet}? "
                        "Use the preflight_check tool to verify it's safe."
        })
        return response["messages"][-1].content

# Example
wallet = "4LkEFjwNNg6XRSXqD6UFMB6neLEQPKFSVBRzXQo8kNgf"
result = asyncio.run(check_wallet_before_paying(wallet))
print(result)
```

### Direct tool invocation

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def score_wallet(wallet: str):
    async with MultiServerMCPClient({
        "twzrd": {
            "url": "https://intel.twzrd.xyz/mcp",
            "transport": "streamable_http",
        }
    }) as client:
        tools = client.get_tools()
        score_tool = next(t for t in tools if t.name == "score_agent")
        result = await score_tool.ainvoke({"wallet": wallet})
        print(f"Trust score for {wallet}: {result}")

asyncio.run(score_wallet("4LkEFjwNNg6XRSXqD6UFMB6neLEQPKFSVBRzXQo8kNgf"))
```

### LangGraph StateGraph with trust-gated payment

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

async def build_trust_gated_agent():
    client = MultiServerMCPClient({
        "twzrd": {
            "url": "https://intel.twzrd.xyz/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

    def call_model(state: MessagesState):
        return {"messages": model.invoke(state["messages"])}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    return builder.compile(), client

async def main():
    graph, client = await build_trust_gated_agent()

    wallet = "4LkEFjwNNg6XRSXqD6UFMB6neLEQPKFSVBRzXQo8kNgf"
    response = await graph.ainvoke({
        "messages": f"Run preflight_check on {wallet}. "
                    "If it passes, tell me the trust score using score_agent."
    })
    print(response["messages"][-1].content)
    await client.aclose()

asyncio.run(main())
```

## MCP configuration

To add TWZRD Agent Intel to any MCP-compatible client:

```json
{
  "mcpServers": {
    "twzrd-agent-intel": {
      "url": "https://intel.twzrd.xyz/mcp"
    }
  }
}
```

## Links

- Website: https://intel.twzrd.xyz
- MCP endpoint: https://intel.twzrd.xyz/mcp
- OpenAPI: https://intel.twzrd.xyz/openapi.json
