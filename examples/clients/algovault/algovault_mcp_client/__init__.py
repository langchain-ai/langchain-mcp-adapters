"""AlgoVault MCP client example for langchain-mcp-adapters.

Demonstrates calling a public streamable-HTTP MCP server (AlgoVault) via
``MultiServerMCPClient`` and invoking a tool directly without an LLM.
"""

from .main import fetch_trade_call, main

__all__ = ["fetch_trade_call", "main"]
