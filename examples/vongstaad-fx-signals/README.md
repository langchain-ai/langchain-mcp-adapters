# Vongstaad FX Signals with LangChain MCP Adapters

This example shows how to use the Vongstaad MCP server with LangChain
to give your agent access to real-time crypto quant signals.

## Setup
```bash
npm install @langchain/mcp-adapters @vongstaad/mcp-fx
```

## Usage
```typescript
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { ChatOpenAI } from "@langchain/openai";

const client = new MultiServerMCPClient({
  vongstaad: {
    command: "npx",
    args: ["@vongstaad/mcp-fx"],
    env: { VONGSTAAD_ENV: "prod" }
  }
});

const tools = await client.getTools();
const model = new ChatOpenAI({ model: "gpt-4o" });
const agent = model.bindTools(tools);
```

## Available Models
7 quantitative models: Correlation, Regime, Momentum, Volatility,
Mean Reversion, SMA, Price. 10 crypto instruments.
Pay per call via x402 on Base. No subscription. No API key.

Discovery: https://vongstaad.com/.well-known/x402.json
MCP Server: https://github.com/VGSTAAD/vongstaad-mcp
