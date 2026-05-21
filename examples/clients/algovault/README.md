# AlgoVault MCP Client Example

A streamable-HTTP client example for `langchain-mcp-adapters`. Connects to the
public [AlgoVault MCP server](https://api.algovault.com/mcp) and invokes
`get_trade_call` for one coin + timeframe, printing the parsed
`{call, confidence, indicators, regime}` JSON. No LLM API key required.

AlgoVault MCP returns a composite BUY/SELL/HOLD trade call across 5 crypto perp
venues (Binance, Bybit, OKX, Bitget, Hyperliquid). Verified track record,
Merkle-anchored on Base L2 ([agentId 44544](https://basescan.org/token/0x8004A169FB4a3325136EB29fA0ceB6D2e539a432?a=44544)).
102,828 calls and 41 on-chain batches recorded as of 2026-05-21 — verify at
<https://algovault.com/track-record>.

## Features

- Uses `MultiServerMCPClient` with `transport="streamable_http"`
- Calls a tool directly via `tool.ainvoke({...})` (no LLM in the loop)
- Demonstrates how to parse the wrapped MCP tool response back into a Python dict
- Optional follow-on snippet wires the same tools into a `create_react_agent`

## Prerequisites

- Python `>=3.10`
- A network path to `https://api.algovault.com/mcp` (public, no auth required)

## Usage

Install dependencies (via `uv`, matching the sibling server example):

```bash
uv sync
```

Run the demo (defaults: `coin=BTC`, `timeframe=4h`):

```bash
uv run algovault-mcp-client
```

Override coin and timeframe:

```bash
uv run algovault-mcp-client ETH 1h
```

Or invoke the module directly:

```bash
uv run python -m algovault_mcp_client BTC 4h
```

Plain `pip` works too:

```bash
pip install -e .
algovault-mcp-client BTC 4h
```

## Expected output

End-to-end against `https://api.algovault.com/mcp` (captured 2026-05-21):

```json
{
  "call": "HOLD",
  "confidence": 52,
  "indicators": {
    "funding_rate": 2.98e-05,
    "funding_24h_avg": 2.98e-05,
    "funding_state": "NORMAL",
    "oi_change_pct": 0,
    "volume_24h": 8134661689.47,
    "trend_persistence": "HIGH",
    "breakout_pending": "IMMINENT"
  },
  "regime": "TRENDING_DOWN"
}
```

The four keys (`call`, `confidence`, `indicators`, `regime`) are the documented
contract. The full MCP response also includes `price`, `reasoning`, `timestamp`,
and an `_algovault` provenance block — all available in the unparsed payload if
your agent needs them.

## Tools exposed by AlgoVault MCP

`client.get_tools()` returns four `BaseTool` objects:

| Tool                | Use case                                                                        |
| ------------------- | ------------------------------------------------------------------------------- |
| `get_trade_call`    | Composite BUY/SELL/HOLD verdict + confidence + regime + indicators              |
| `get_trade_signal`  | Back-compat alias of `get_trade_call` (same payload shape)                      |
| `scan_funding_arb`  | Rank funding-spread opportunities across the 5 venues                           |
| `get_market_regime` | Regime classification (TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE)        |

Each is invokable via `tool.ainvoke({...})`.

## Wire into a LangGraph ReAct agent

If you want an LLM to decide when to call which tool, install the optional
`agent` extras and drop the same `tools` list into `create_react_agent`:

```bash
uv sync --extra agent
export ANTHROPIC_API_KEY=...   # or use any LangChain chat-model identifier
```

```python
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def run() -> None:
    client = MultiServerMCPClient(
        {
            "algovault": {
                "url": "https://api.algovault.com/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent("anthropic:claude-sonnet-4-5", tools)
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Get a trade call for BTC on the 4h timeframe.",
                }
            ]
        }
    )
    print(result["messages"][-1].content)


asyncio.run(run())
```

Swap the model string for any LangChain chat-model identifier you have a key
for.

## Further reading

A longer walkthrough — including the optional helper-skills install, full
indicator field reference, and an end-to-end agent recipe — lives at
<https://github.com/AlgoVaultLabs/algovault-skills/blob/main/docs/integrations/langchain.md>.

## License

MIT, matching the parent repository.
