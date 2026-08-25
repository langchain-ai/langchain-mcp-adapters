"""Tests for surfacing MCP elicitation as a LangGraph interrupt.

The composition only works on MCP 2026-07-28. There, a `tools/call` that needs
input *completes* — the server returns an `InputRequiredResult` carrying its
questions and a resumable `request_state` — so the connection can be torn down
while a human answers. On a handshake-era connection the server is instead left
blocked on a request the interrupt would abandon, which is why that combination
is refused rather than silently broken.
"""

from typing import TypedDict

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CreateMessageRequest,
    CreateMessageRequestParams,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    ListRootsRequest,
    SamplingMessage,
    TextContent,
)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interrupts import (
    UnansweredInputRequestError,
    coerce_resume_value,
    describe_input_requests,
    in_langgraph_runtime,
)
from tests.servers.counting_server import create_counting_server
from tests.utils import run_streamable_http

try:
    import langgraph.func  # noqa: F401

    LANGGRAPH_INSTALLED = True
except ImportError:
    LANGGRAPH_INSTALLED = False

requires_langgraph = pytest.mark.skipif(
    not LANGGRAPH_INSTALLED, reason="langgraph not installed"
)

PORT = 8260
SCHEMA = {
    "type": "object",
    "properties": {"email": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["email", "age"],
}


def _elicit(message: str = "give me details") -> ElicitRequest:
    return ElicitRequest(
        params=ElicitRequestFormParams(message=message, requested_schema=SCHEMA)
    )


# --- interrupt payload ------------------------------------------------------


def test_describe_elicit_request() -> None:
    payload = describe_input_requests(
        {"k": _elicit()}, server_name="profiles", tool_name="create_profile"
    )

    assert payload["type"] == "mcp_input_required"
    assert payload["server"] == "profiles"
    assert payload["tool"] == "create_profile"
    assert payload["requests"]["k"] == {
        "kind": "elicit",
        "mode": "form",
        "message": "give me details",
        "requested_schema": SCHEMA,
    }


def test_describe_payload_is_json_safe() -> None:
    """The payload goes through a checkpointer, so it must be plain data."""
    import json

    payload = describe_input_requests(
        {
            "e": _elicit(),
            "s": CreateMessageRequest(
                params=CreateMessageRequestParams(
                    messages=[
                        SamplingMessage(
                            role="user", content=TextContent(type="text", text="hi")
                        )
                    ],
                    maxTokens=10,
                )
            ),
            "r": ListRootsRequest(),
        }
    )

    assert json.loads(json.dumps(payload)) == payload
    assert payload["requests"]["s"]["kind"] == "sample"
    assert payload["requests"]["r"]["kind"] == "list_roots"
    # Nothing to identify when the caller did not say.
    assert "server" not in payload


# --- resume value coercion --------------------------------------------------


def test_coerce_bare_content_for_a_single_request() -> None:
    responses = coerce_resume_value({"k": _elicit()}, {"email": "a@b.c", "age": 28})

    assert responses["k"] == ElicitResult(
        action="accept", content={"email": "a@b.c", "age": 28}
    )


def test_coerce_keyed_mapping() -> None:
    requests = {"one": _elicit("first"), "two": _elicit("second")}
    responses = coerce_resume_value(
        requests,
        {
            "one": {"email": "a@b.c", "age": 1},
            "two": {"action": "decline"},
        },
    )

    assert responses["one"].action == "accept"
    assert responses["two"].action == "decline"


@pytest.mark.parametrize("answer", ["decline", "cancel"])
def test_coerce_bare_refusal(answer) -> None:
    assert coerce_resume_value({"k": _elicit()}, answer)["k"].action == answer


def test_coerce_passes_typed_results_through() -> None:
    typed = ElicitResult(action="accept", content={"email": "x@y.z", "age": 3})

    assert coerce_resume_value({"k": _elicit()}, {"k": typed})["k"] is typed


def test_coerce_reports_unanswered_requests() -> None:
    requests = {"one": _elicit(), "two": _elicit()}

    with pytest.raises(UnansweredInputRequestError) as exc:
        coerce_resume_value(requests, {"one": {"email": "a@b.c", "age": 1}})

    assert exc.value.missing == ["two"]
    assert "two" in str(exc.value)


def test_coerce_rejects_an_unreadable_answer() -> None:
    with pytest.raises(TypeError, match="Cannot read"):
        coerce_resume_value({"k": _elicit()}, 42)


def test_not_in_a_langgraph_runtime_by_default() -> None:
    assert in_langgraph_runtime() is False


# --- end to end -------------------------------------------------------------


class _State(TypedDict, total=False):
    result: str


def _messages_in(exc: BaseException) -> list[str]:
    """Flatten an exception and any groups it nests into their messages.

    A failure raised inside a graph node comes back wrapped in the anyio task
    group's `ExceptionGroup`, so asserting on the top-level type is not enough.
    """
    if isinstance(exc, BaseExceptionGroup):
        return [m for sub in exc.exceptions for m in _messages_in(sub)]
    nested = _messages_in(exc.__cause__) if exc.__cause__ is not None else []
    return [str(exc), *nested]


def _graph(tool):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    async def node(state: _State) -> _State:
        result = await tool.ainvoke(
            {"args": {"name": "Alice"}, "id": "1", "type": "tool_call"}
        )
        return {"result": str(result.content)}

    graph = StateGraph(_State)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile(checkpointer=InMemorySaver())


@requires_langgraph
async def test_elicitation_becomes_an_interrupt_and_resumes(socket_enabled) -> None:
    from langgraph.types import Command

    with run_streamable_http(create_counting_server, PORT):
        client = MultiServerMCPClient(
            {"profiles": {"url": f"http://localhost:{PORT}/mcp", "transport": "http"}},
            elicitation="interrupt",
        )
        tools = await client.get_tools()
        app = _graph(next(t for t in tools if t.name == "create_profile"))
        config = {"configurable": {"thread_id": "t"}}

        paused = await app.ainvoke({}, config)
        (interrupt,) = paused["__interrupt__"]
        (key,) = interrupt.value["requests"]

        assert interrupt.value["type"] == "mcp_input_required"
        assert interrupt.value["server"] == "profiles"
        assert interrupt.value["tool"] == "create_profile"
        assert "Alice" in interrupt.value["requests"][key]["message"]
        assert interrupt.value["requests"][key]["requested_schema"]["required"] == [
            "email",
            "age",
        ]

        resumed = await app.ainvoke(
            Command(resume={"email": "alice@example.com", "age": 28}), config
        )

    assert "alice@example.com" in resumed["result"]
    assert "28" in resumed["result"]


@requires_langgraph
async def test_resuming_does_not_re_ask_the_server(socket_enabled) -> None:
    """The whole point of memoizing: one round-trip before, one after.

    Without it, resuming would replay the node from the top and re-issue the
    original `tools/call` just to mint a fresh `request_state`, costing a third
    round-trip and asking the server the same question twice.
    """
    from langgraph.types import Command

    with run_streamable_http(create_counting_server, PORT + 1):
        client = MultiServerMCPClient(
            {
                "profiles": {
                    "url": f"http://localhost:{PORT + 1}/mcp",
                    "transport": "http",
                }
            },
            elicitation="interrupt",
        )
        tools = await client.get_tools()
        counter = next(t for t in tools if t.name == "rounds_seen")

        async def rounds() -> str:
            result = await counter.ainvoke({"args": {}, "id": "c", "type": "tool_call"})
            return result.content[0]["text"]

        app = _graph(next(t for t in tools if t.name == "create_profile"))
        config = {"configurable": {"thread_id": "t"}}

        await app.ainvoke({}, config)
        assert await rounds() == "1"

        await app.ainvoke(Command(resume={"email": "a@b.c", "age": 1}), config)
        assert await rounds() == "2"


@requires_langgraph
async def test_declining_through_the_resume_value(socket_enabled) -> None:
    from langgraph.types import Command

    with run_streamable_http(create_counting_server, PORT + 2):
        client = MultiServerMCPClient(
            {
                "profiles": {
                    "url": f"http://localhost:{PORT + 2}/mcp",
                    "transport": "http",
                }
            },
            elicitation="interrupt",
        )
        tools = await client.get_tools()
        app = _graph(next(t for t in tools if t.name == "create_profile"))
        config = {"configurable": {"thread_id": "t"}}

        await app.ainvoke({}, config)
        resumed = await app.ainvoke(Command(resume="decline"), config)

    # The tool consumes the plain model, so declining aborts the call and the
    # adapter reports it back to the model rather than raising.
    assert "declin" in resumed["result"].lower() or "error" in resumed["result"].lower()


@requires_langgraph
async def test_a_tool_needing_no_input_is_unaffected(socket_enabled) -> None:
    """Interrupt mode must not require a graph for calls that never elicit."""
    with run_streamable_http(create_counting_server, PORT + 3):
        client = MultiServerMCPClient(
            {
                "profiles": {
                    "url": f"http://localhost:{PORT + 3}/mcp",
                    "transport": "http",
                }
            },
            elicitation="interrupt",
        )
        tools = await client.get_tools()
        counter = next(t for t in tools if t.name == "rounds_seen")

        # Called straight from a test, with no graph anywhere in sight.
        result = await counter.ainvoke({"args": {}, "id": "c", "type": "tool_call"})

    assert result.content[0]["text"] == "0"


async def test_eliciting_outside_a_graph_explains_itself(socket_enabled) -> None:
    """No graph means nothing to interrupt; say so instead of hanging.

    This raises rather than returning a tool error: it is a misconfiguration on
    the caller's side, not something the model can recover from by retrying.
    """
    with run_streamable_http(create_counting_server, PORT + 4):
        client = MultiServerMCPClient(
            {
                "profiles": {
                    "url": f"http://localhost:{PORT + 4}/mcp",
                    "transport": "http",
                }
            },
            elicitation="interrupt",
        )
        tools = await client.get_tools()
        tool = next(t for t in tools if t.name == "create_profile")

        with pytest.raises(MCPError, match="not called from inside a running"):
            await tool.ainvoke(
                {"args": {"name": "Alice"}, "id": "1", "type": "tool_call"}
            )


@requires_langgraph
async def test_interrupt_mode_on_a_legacy_connection_explains_itself(
    socket_enabled,
) -> None:
    """Pinning legacy removes the resumable state interrupting depends on."""
    with run_streamable_http(create_counting_server, PORT + 5):
        client = MultiServerMCPClient(
            {
                "profiles": {
                    "url": f"http://localhost:{PORT + 5}/mcp",
                    "transport": "http",
                    "protocol": "legacy",
                }
            },
            elicitation="interrupt",
        )
        tools = await client.get_tools()
        app = _graph(next(t for t in tools if t.name == "create_profile"))

        with pytest.raises(BaseException) as exc:
            await app.ainvoke({}, {"configurable": {"thread_id": "t"}})

    assert any(
        "handshake-era protocol revision" in m for m in _messages_in(exc.value)
    ), _messages_in(exc.value)


def test_client_rejects_an_unknown_elicitation_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported elicitation mode"):
        MultiServerMCPClient({}, elicitation="interupt")
