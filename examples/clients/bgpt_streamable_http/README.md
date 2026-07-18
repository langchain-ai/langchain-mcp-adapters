# BGPT Remote MCP Client Example

Connect to the hosted [BGPT](https://bgpt.pro/mcp/) scientific paper search MCP server using `MultiServerMCPClient` and streamable HTTP transport.

## BGPT endpoints

| Type | URL |
|------|-----|
| MCP (streamable HTTP) | `https://bgpt.pro/mcp/stream` |
| REST search | `POST https://bgpt.pro/api/mcp-search` |
| REST DOI lookup | `POST https://bgpt.pro/api/mcp-doi-lookup` |

Free tier works without an API key.

## Setup

```bash
pip install langchain-mcp-adapters mcp
```

## List tools (no API key)

```bash
python bgpt_mcp_client.py
```

## Run an agent query

Requires `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=sk-...
python bgpt_mcp_client.py --query "GLP-1 alcohol craving"
```

The agent uses BGPT MCP tools to retrieve structured evidence fields (methods, limitations, conflicts of interest, falsifiability) before summarizing.

## Related

- BGPT docs: https://bgpt.pro/mcp/
- BGPT GitHub: https://github.com/connerlambden/bgpt-mcp
