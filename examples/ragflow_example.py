"""
Example: Connect LangChain agents to RAGFlow via MCP.

This example demonstrates how to use `langchain-mcp-adapters` to connect
a LangChain application to RAGFlow's MCP server, enabling agents to
search knowledge bases, list datasets, and discover chat assistants.

Prerequisites:
    1. RAGFlow MCP server running (default: http://localhost:9382)
    2. A valid RAGFlow API key
    3. `langchain-mcp-adapters` installed (`pip install langchain-mcp-adapters`)

Start the RAGFlow MCP server:

    .. code-block:: bash

        cd /path/to/ragflow
        uv run mcp/server/server.py \\
            --host=127.0.0.1 --port=9382 \\
            --mode=self-host --api-key=ragflow-xxxxx

Usage:

    .. code-block:: bash

        export RAGFLOW_API_KEY="ragflow-xxxxx"
        python ragflow_example.py
"""

import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

# ── Configuration ──────────────────────────────────────────────────────────

RAGFLOW_MCP_URL = os.environ.get(
    "RAGFLOW_MCP_URL", "http://localhost:9382/mcp"
)
RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY", "")


# ── Agentic RAG helper ─────────────────────────────────────────────────────


async def demonstrate_ragflow_mcp():
    """Connect to RAGFlow MCP and demonstrate available tools."""

    if not RAGFLOW_API_KEY:
        print(
            "⚠️  RAGFLOW_API_KEY not set. "
            "Export it or pass via config.\n"
        )

    # Step 1: Configure the MCP client with RAGFlow connection
    client = MultiServerMCPClient(
        {
            "ragflow": {
                "url": RAGFLOW_MCP_URL,
                "transport": "http",
                "headers": {
                    "api_key": RAGFLOW_API_KEY,
                    "Accept": "application/json, text/event-stream",
                },
            }
        }
    )

    # Step 2: Discover all MCP tools registered by RAGFlow
    print("🔄 Loading MCP tools from RAGFlow...")
    tools = await client.get_tools()

    print(f"✅ Found {len(tools)} tool(s):\n")
    for tool in tools:
        print(f"   🔧 {tool.name}")
        print(f"      {tool.description[:120]}...\n")

    # Step 3: Call ragflow_retrieval to search a knowledge base
    retrieval_tool = next(
        (t for t in tools if t.name == "ragflow_retrieval"), None
    )
    if retrieval_tool:
        print("🔍 Calling ragflow_retrieval...")
        try:
            result = await retrieval_tool.ainvoke({
                "question": "What is the main functionality?",
                "page_size": 5,
            })
            print(f"   Result: {str(result)[:300]}...\n")
        except Exception as e:
            print(f"   ⚠️  Retrieval failed (expected if no backend): {e}\n")
    else:
        print("⚠️  ragflow_retrieval not available\n")

    # Step 4: Call ragflow_list_datasets
    list_datasets_tool = next(
        (t for t in tools if t.name == "ragflow_list_datasets"), None
    )
    if list_datasets_tool:
        print("📚 Calling ragflow_list_datasets...")
        try:
            result = await list_datasets_tool.ainvoke({"page_size": 10})
            print(f"   Result: {str(result)[:200]}...\n")
        except Exception as e:
            print(f"   ⚠️  Failed: {e}\n")

    # Step 5: Call ragflow_list_chats
    list_chats_tool = next(
        (t for t in tools if t.name == "ragflow_list_chats"), None
    )
    if list_chats_tool:
        print("💬 Calling ragflow_list_chats...")
        try:
            result = await list_chats_tool.ainvoke({"page_size": 10})
            print(f"   Result: {str(result)[:200]}...\n")
        except Exception as e:
            print(f"   ⚠️  Failed: {e}\n")

    print("✅ Demonstration complete.")


# ── Entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    asyncio.run(demonstrate_ragflow_mcp())
