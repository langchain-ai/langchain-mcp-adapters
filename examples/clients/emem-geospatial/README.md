# emem Geospatial MCP Client Example

A LangGraph ReAct agent that connects to [emem](https://github.com/Vortx-AI/emem), a remote MCP server for signed geospatial facts, over Streamable HTTP.

## What it does

1. Connects to the public emem MCP endpoint at `https://emem.dev/mcp`.
2. Auto-discovers all emem tools (locate, recall, compare, verify, etc.).
3. Creates a ReAct agent that answers a geospatial verification question.
4. The agent locates coordinates, recalls elevation / land cover / vegetation / surface water, and summarises with signed receipt `fact_cid`s.

No API key is needed for emem reads.

## Install

```bash
pip install langchain-mcp-adapters langgraph langchain-openai
```

## Run

```bash
export OPENAI_API_KEY="sk-..."
python emem_geospatial_agent.py
```

## Links

- emem GitHub: https://github.com/Vortx-AI/emem
- MCP endpoint: https://emem.dev/mcp
- MCP Registry: `io.github.Vortx-AI/emem`
