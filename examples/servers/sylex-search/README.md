# Sylex Search MCP Server Example

[Sylex Search](https://sylex.ai) is a universal product discovery MCP server for AI agents. It provides a curated, searchable database of 11,000+ tools, libraries, APIs, and SaaS products — enabling agents to discover software without making web searches or LLM calls.

## Features

- Zero-latency, deterministic full-text search (no LLM required on the server side)
- 11,000+ indexed products: npm packages, PyPI libraries, Rust crates, GitHub repos, SaaS tools, and more
- No authentication required
- Hosted SSE endpoint — nothing to install or run locally
- 10 MCP tools covering search, comparison, and product management

## Available Tools

| Tool | Description |
|------|-------------|
| `search.discover` | Search for products by keyword or description |
| `search.details` | Get full details for a specific product |
| `search.compare` | Compare multiple products side by side |
| `search.categories` | Browse products by category |
| `search.alternatives` | Find alternatives to a given product |
| `search.feedback` | Submit feedback on a search result |
| `manage.register` | Register a new product in the catalog |
| `manage.claim` | Claim ownership of an existing listing |
| `manage.update` | Update an owned product listing |
| `manage.list_mcp` | Add MCP connection config to a listing |

## Usage

No server setup required. The Sylex Search MCP server is hosted at:

```
https://mcp-server-production-38c9.up.railway.app/sse
```

Install the required dependencies:

```bash
pip install langchain-mcp-adapters langgraph "langchain[openai]"

export OPENAI_API_KEY=<your_api_key>
```

Run the example agent:

```bash
python agent.py
```

## Example

The included [`agent.py`](agent.py) script creates a LangGraph ReAct agent that uses Sylex Search to:

1. Search for products matching a query
2. Get detailed information about a specific product
3. Compare multiple products side by side

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_react_agent
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient(
    {
        "sylex": {
            "url": "https://mcp-server-production-38c9.up.railway.app/sse",
            "transport": "sse",
        }
    }
)
tools = await client.get_tools()
agent = create_react_agent("openai:gpt-4.1", tools)
response = await agent.ainvoke({"messages": "Find me a good Python HTTP client library"})
```
