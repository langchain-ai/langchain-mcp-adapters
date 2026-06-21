import os
from pathlib import Path
from typing import Annotated, NotRequired

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from typing_extensions import TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.langgraph_prompt import create_mcp_prompt_injection_node


class MessagesAndDomainState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    domain: NotRequired[str]


@pytest.fixture
def math_client() -> MultiServerMCPClient:
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    return MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
        },
    )


async def test_injection_prepend_mcp_before_user(math_client: MultiServerMCPClient):
    inject = create_mcp_prompt_injection_node(
        math_client,
        "math",
        "configure_assistant",
        arguments={"skills": "unit-test"},
        placement="prepend",
    )
    builder = StateGraph(MessagesState)
    builder.add_node("mcp_prompt", inject)
    builder.add_edge(START, "mcp_prompt")
    builder.add_edge("mcp_prompt", END)
    graph = builder.compile()

    out = await graph.ainvoke({"messages": [HumanMessage(content="hello")]})
    msgs = out["messages"]
    assert len(msgs) == 2
    assert isinstance(msgs[0], AIMessage)
    assert "unit-test" in (msgs[0].content or "")
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == "hello"


async def test_injection_node_append(math_client: MultiServerMCPClient):
    inject = create_mcp_prompt_injection_node(
        math_client,
        "math",
        "calculation_guide",
        placement="append",
    )
    builder = StateGraph(MessagesState)
    builder.add_node("mcp_prompt", inject)
    builder.add_edge(START, "mcp_prompt")
    builder.add_edge("mcp_prompt", END)
    graph = builder.compile()

    out = await graph.ainvoke({"messages": [HumanMessage(content="first")]})
    msgs = out["messages"]
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "first"
    assert isinstance(msgs[1], AIMessage)
    assert "calculation" in (msgs[1].content or "").lower()


async def test_injection_node_arguments_resolver(math_client: MultiServerMCPClient):
    inject = create_mcp_prompt_injection_node(
        math_client,
        "math",
        "configure_assistant",
        arguments_resolver=lambda s: {"skills": s["domain"]},
    )
    builder = StateGraph(MessagesAndDomainState)
    builder.add_node("mcp_prompt", inject)
    builder.add_edge(START, "mcp_prompt")
    builder.add_edge("mcp_prompt", END)
    graph = builder.compile()

    out = await graph.ainvoke(
        {"messages": [HumanMessage(content="x")], "domain": "resolver-skills"},
    )
    assert "resolver-skills" in (out["messages"][0].content or "")
