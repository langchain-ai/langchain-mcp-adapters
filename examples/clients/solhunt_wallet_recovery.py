"""
SolHunt MCP Client Example

Shows how to connect to SolHunt's remote MCP server for Solana wallet recovery.

SolHunt provides:
- Wallet health analysis
- Recoverable SOL detection from zero-balance token accounts
- Trustless recovery transaction building

Prerequisites:
    pip install langchain-mcp-adapters langgraph langchain-openai

Usage:
    export OPENAI_API_KEY=your_key
    python solhunt_wallet_recovery.py
"""

import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def analyze_wallet(wallet_address: str):
    """Analyze a Solana wallet using SolHunt MCP tools."""
    
    # Configure SolHunt MCP server (remote HTTP)
    async with MultiServerMCPClient({
        "solhunt": {
            "transport": "http",
            "url": "https://solhunt.dev/.netlify/functions/mcp",
            "timeout": 30,
        }
    }) as client:
        # Load SolHunt tools
        tools = client.get_tools()
        print(f"Loaded {len(tools)} tools from SolHunt:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")
        
        # Create agent with SolHunt tools
        model = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(model, tools)
        
        # Ask agent to analyze wallet
        prompt = f"""
        Analyze this Solana wallet for recoverable SOL: {wallet_address}
        
        Use the check_wallet_health tool first to see:
        1. The wallet's health score
        2. How much SOL is recoverable
        3. How many accounts can be closed
        
        Then explain what these numbers mean in simple terms.
        """
        
        result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        
        # Print the agent's response
        for message in result["messages"]:
            if message.type == "ai":
                print(f"\n🤖 Agent: {message.content}")
                
        return result

async def get_recovery_plan(wallet_address: str):
    """Get detailed recovery opportunities for a wallet."""
    
    async with MultiServerMCPClient({
        "solhunt": {
            "transport": "http",
            "url": "https://solhunt.dev/.netlify/functions/mcp",
            "timeout": 30,
        }
    }) as client:
        
        tools = client.get_tools()
        model = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(model, tools)
        
        prompt = f"""
        Get the recovery opportunities for wallet: {wallet_address}
        
        Use the get_recovery_opportunities tool to see:
        - Which specific token accounts can be closed
- How much SOL will be recovered from each
        - The fee SolHunt takes (15%)
        - Net amount to the user
        
        Present this as a clear action plan.
        """
        
        result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        
        for message in result["messages"]:
            if message.type == "ai":
                print(f"\n📋 Recovery Plan: {message.content}")
                
        return result

if __name__ == "__main__":
    # Example: Analyze a wallet (replace with actual address)
    example_wallet = "ExampleSolanaWalletAddress11111111111111111111"
    
    print("🔍 SolHunt Wallet Analysis Demo")
    print("=" * 50)
    print(f"Analyzing wallet: {example_wallet}\n")
    
    # Run analysis
    asyncio.run(analyze_wallet(example_wallet))
    
    # Uncomment to get detailed recovery plan:
    # asyncio.run(get_recovery_plan(example_wallet))