# Global Chat MCP Server Example

A LangGraph agent example that connects to [Global Chat](https://global-chat.io)'s MCP server for cross-protocol agent and service discovery.

## About Global Chat

Global Chat is a cross-protocol agent discovery platform. Agents can search for other agents and services across MCP registries, A2A (Agent-to-Agent) protocol, and agents.txt endpoints.

- **Repository:** [github.com/pumanitro/global-chat](https://github.com/pumanitro/global-chat)
- **npm:** [@globalchatadsapp/mcp-server](https://www.npmjs.com/package/@globalchatadsapp/mcp-server)
- **Transport:** stdio (via npx)

## Tools Provided

The MCP server exposes tools for:

- Searching the agent/service directory
- Discovering MCP servers, A2A agents, and agents.txt endpoints
- Querying cross-protocol service metadata

## Setup

Install dependencies:

```bash
pip install langchain-mcp-adapters langgraph langchain-openai
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=<your_api_key>
```

Ensure `npx` is available (requires Node.js):

```bash
node --version  # v18+ recommended
```

## Usage

```bash
python global_chat_example.py
```

The agent will connect to the Global Chat MCP server and can answer queries like:

- "Find MCP servers related to payments"
- "Search for AI agents that handle document processing"
- "What agent discovery services are available?"
