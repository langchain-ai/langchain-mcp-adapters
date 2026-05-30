import os
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompt_binder import bind_mcp_prompt


async def _fake_model(model_input: object) -> AIMessage:
    if isinstance(model_input, list):
        msgs = model_input
    elif isinstance(model_input, dict) and "messages" in model_input:
        msgs = model_input["messages"]
    else:
        msgs = []
    return AIMessage(content=f"count={len(msgs)}")


async def test_bind_mcp_prompt_prepends_messages_list_input():
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
        },
    )
    model = RunnableLambda(_fake_model)
    bound = bind_mcp_prompt(
        model,
        client=client,
        server_name="math",
        prompt_name="configure_assistant",
        arguments={"skills": "algebra"},
    )
    out = await bound.ainvoke([HumanMessage(content="solve 2+2")])
    assert isinstance(out, AIMessage)
    assert out.content == "count=2"


async def test_bind_mcp_prompt_dict_state_messages():
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
        },
    )
    model = RunnableLambda(_fake_model)
    bound = bind_mcp_prompt(
        model,
        client=client,
        server_name="math",
        prompt_name="calculation_guide",
        arguments=None,
    )
    state = {"messages": [HumanMessage(content="hi")], "extra": 1}
    out = await bound.ainvoke(state)
    assert isinstance(out, AIMessage)
    assert out.content == "count=2"
    assert state["extra"] == 1


async def test_bind_mcp_prompt_arguments_resolver():
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
        },
    )
    model = RunnableLambda(_fake_model)
    bound = bind_mcp_prompt(
        model,
        client=client,
        server_name="math",
        prompt_name="configure_assistant",
        arguments_resolver=lambda s: {"skills": s["skills"]},
    )
    out = await bound.ainvoke(
        {"messages": [HumanMessage(content="x")], "skills": "geometry"},
    )
    assert isinstance(out, AIMessage)
    assert out.content == "count=2"


async def test_bind_mcp_prompt_uses_mock_client_for_resolver_priority():
    client = AsyncMock(spec=MultiServerMCPClient)
    client.get_prompt = AsyncMock(
        return_value=[AIMessage(content="from mcp")],
    )
    model = RunnableLambda(_fake_model)
    bound = bind_mcp_prompt(
        model,
        client=client,
        server_name="srv",
        prompt_name="p",
        arguments={"a": 1},
        arguments_resolver=lambda _: {"skills": "from_resolver"},
    )
    await bound.ainvoke([HumanMessage(content="u")])
    client.get_prompt.assert_awaited_once()
    call_kw = client.get_prompt.await_args
    assert call_kw.kwargs["arguments"] == {"skills": "from_resolver"}
