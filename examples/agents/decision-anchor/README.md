# Decision Anchor MCP Integration Example

A LangGraph ReAct agent that connects to [Decision Anchor](https://github.com/zse4321/decision-anchor-sdk)'s remote MCP server for external accountability proof.

Decision Anchor provides external decision records for agent payments, delegation, and disputes via MCP. It does not monitor, judge, or intervene.

## What this example does

1. Connects to DA's remote MCP server (`https://mcp.decision-anchor.com/mcp`) via `MultiServerMCPClient`
2. Registers as a new agent (one-time, no API key required)
3. Creates a Decision Declaration (DD) with accountability scope
4. Checks DAC balance

## Usage

```bash
cd examples/agents/decision-anchor
uv run decision-anchor-agent
```

Or run directly:

```bash
cd examples/agents/decision-anchor
export OPENAI_API_KEY=your_key
uv run python -m decision_anchor_agent
```

## Notes

- No DA API key needed — agents register via MCP tools and receive Trial 500 DAC / 30 days
- Running this example creates a real agent registration and decision record on DA
- The MCP server uses streamable HTTP transport
