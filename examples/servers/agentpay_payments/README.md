# agentpay-mcp Example — x402 Payments for LangGraph Agents

LangGraph agents can call x402-protected API endpoints and pay automatically,
with no custom payment handling code in the agent.

[agentpay-mcp](https://www.npmjs.com/package/agentpay-mcp) runs as an MCP
subprocess. The agent gets a `pay_x402_endpoint` tool — when it hits a 402,
it calls the tool, signs the USDC transfer, and retries. The agent framework
never needs to know about payment mechanics.

## Setup

```bash
pip install langchain-mcp-adapters langgraph langchain-anthropic
npm install -g agentpay-mcp
```

```bash
export AGENT_PRIVATE_KEY=0x...       # Agent hot wallet key
export AGENT_WALLET_ADDRESS=0x...    # Deployed AgentAccountV2 address
export CHAIN_ID=84532                # 84532 = Base Sepolia testnet
export ANTHROPIC_API_KEY=...
```

## Run

```bash
python agentpay_example.py
```

## How it works

agentpay-mcp wraps [agentwallet-sdk](https://www.npmjs.com/package/agentwallet-sdk)
behind an MCP interface. On-chain spend limits (set when deploying the wallet)
cap what the agent can pay per transaction and per day — operators configure
this before handing keys to an agent.

No browser. No wallet extension. Fully server-side.

## Tested with

- langchain-mcp-adapters 0.1.x
- agentpay-mcp 1.2.0
- Base Sepolia testnet
