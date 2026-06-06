"""
TWZRD Agent Intel — LangChain MCP Adapters Example Client

Demonstrates connecting to a remote Streamable-HTTP MCP server (TWZRD Agent Intel)
to trust-score Solana wallets before making x402 micropayments.

Usage:
    pip install langchain-mcp-adapters langgraph langchain-openai
    export OPENAI_API_KEY=<your_key>
    python client.py
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langchain_openai import ChatOpenAI

TWZRD_MCP_URL = "https://intel.twzrd.xyz/mcp"

# --- Example 1: Direct tool call ---

async def score_wallet(wallet: str) -> None:
    """Directly invoke the score_agent tool on a Solana wallet."""
    async with MultiServerMCPClient({
        "twzrd": {"url": TWZRD_MCP_URL, "transport": "streamable_http"}
    }) as client:
        tools = client.get_tools()
        score_tool = next(t for t in tools if t.name == "score_agent")
        result = await score_tool.ainvoke({"wallet": wallet})
        print(f"[score_agent] {wallet}: {result}")


# --- Example 2: React agent with preflight check ---

async def preflight_agent(wallet: str) -> str:
    """Use a ReAct agent to decide whether a wallet is safe to pay."""
    async with MultiServerMCPClient({
        "twzrd": {"url": TWZRD_MCP_URL, "transport": "streamable_http"}
    }) as client:
        tools = client.get_tools()
        model = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(model, tools)
        response = await agent.ainvoke({
            "messages": (
                f"Run preflight_check on wallet {wallet}. "
                "Report whether it is safe to send a USDC payment and why."
            )
        })
        return response["messages"][-1].content


# --- Example 3: LangGraph StateGraph with trust-gated payment logic ---

async def build_payment_guard_graph():
    """
    Build a LangGraph agent that uses TWZRD trust tools to gate payments.
    The agent calls score_agent and preflight_check before approving a transfer.
    """
    client = MultiServerMCPClient({
        "twzrd": {"url": TWZRD_MCP_URL, "transport": "streamable_http"}
    })
    tools = await client.get_tools()
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

    def call_model(state: MessagesState):
        return {"messages": model.invoke(state["messages"])}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph, client


async def main():
    wallet = "4LkEFjwNNg6XRSXqD6UFMB6neLEQPKFSVBRzXQo8kNgf"

    print("=== Example 1: Direct score_agent tool call ===")
    await score_wallet(wallet)

    print("\n=== Example 2: ReAct agent preflight check ===")
    result = await preflight_agent(wallet)
    print(result)

    print("\n=== Example 3: LangGraph StateGraph trust-gated payment ===")
    graph, client = await build_payment_guard_graph()
    response = await graph.ainvoke({
        "messages": (
            f"I need to send 1 USDC to {wallet}. "
            "First run preflight_check. If it passes, run score_agent and give me "
            "a final recommendation on whether to proceed with the payment."
        )
    })
    print(response["messages"][-1].content)
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
