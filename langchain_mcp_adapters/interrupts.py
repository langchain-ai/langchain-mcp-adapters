"""Surface MCP elicitation as a LangGraph [interrupt][langgraph.types.interrupt].

Through 2025-11-25 this composition was not really possible. Elicitation was a
server-to-client request sent *during* a `tools/call`, so the server sat blocked
on an open connection waiting for an answer. Raising out of the callback to
suspend a graph would tear down that connection and abandon the call; on resume
the tool would start over with the server remembering nothing.

2026-07-28 turns the exchange inside out. The server *returns* an
[`InputRequiredResult`][mcp.types.InputRequiredResult] carrying its questions
and an opaque `request_state` token, and the call is over at the transport
level. So the client can close the connection, suspend for as long as it takes a
human to answer, and then re-issue the call against a completely new session
with the answers and the token attached. That is exactly the shape of a
LangGraph interrupt, which is what this module wires up.

The two rounds are memoized as LangGraph
[tasks][langgraph.func.task], so resuming replays the first round from the
checkpoint instead of asking the server the same question again. A tool that
elicits once costs two round-trips, not three.

!!! note

    `request_state` is minted and sealed by the server, and is subject to its
    TTL and signing-key lifetime. A server using the default process-local key
    will reject a token that outlives its own restart, regardless of how long
    the graph held it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from mcp.types import (
    INVALID_REQUEST,
    CreateMessageRequest,
    CreateMessageResult,
    ElicitRequest,
    ElicitResult,
    ErrorData,
    InputRequiredResult,
    ListRootsRequest,
    ListRootsResult,
)
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.types import InputRequest, InputResponse, InputResponses

_ResultT = TypeVar("_ResultT", bound=BaseModel)

DEFAULT_MAX_INPUT_ROUNDS = 10
"""Rounds of questions tolerated before giving up.

Mirrors `langchain_mcp_adapters.input_required.DEFAULT_MAX_INPUT_ROUNDS`.
"""

LANGGRAPH_REQUIRED_ERROR = (
    "Interrupt-based elicitation needs LangGraph, which is not installed. "
    "Install it with `pip install langgraph`, or use callback-based "
    "elicitation instead by passing `Callbacks(on_elicitation=...)`."
)

OUTSIDE_GRAPH_ERROR = (
    "This server asked for input, but the tool was not called from inside a "
    "running LangGraph execution, so there is nothing to interrupt. Suspending "
    "and resuming a graph is what carries the server's `request_state` across "
    "the pause. Either invoke this tool from a graph with a checkpointer "
    "configured, or use callback-based elicitation via "
    "`Callbacks(on_elicitation=...)`."
)

_PREGEL_CALL_CONFIG_KEY = "__pregel_call"
"""Marker LangGraph puts in `configurable` for code running inside a Pregel task.

Checked by string rather than imported: the constant lives in
`langgraph._internal._constants`, and the wire value is the more stable of the
two. `langgraph.func.task` reads this exact key, so its presence is precisely
the condition under which a task can be created.
"""


def in_langgraph_runtime() -> bool:
    """Whether the caller is executing inside a LangGraph task.

    Interrupting and memoizing both require it, so this decides whether
    `elicitation="interrupt"` can be honored for a given call.
    """
    try:
        from langgraph.config import get_config  # noqa: PLC0415
    except ImportError:
        return False
    try:
        config = get_config()
    except RuntimeError:
        # Not in a runnable context at all.
        return False
    return _PREGEL_CALL_CONFIG_KEY in (config.get("configurable") or {})


INTERRUPT_ON_LEGACY_ERROR = (
    "This server sent an `elicitation/create` request mid-call, which means the "
    "connection negotiated a handshake-era protocol revision. Interrupt-based "
    "elicitation cannot answer it: suspending the graph tears down the "
    "connection the server is blocked on, and there is no `request_state` to "
    "resume against. Either connect to a server that speaks 2026-07-28, or use "
    "callback-based elicitation via `Callbacks(on_elicitation=...)`."
)


async def declare_elicitation_only(
    _context: Any,
    _params: Any,
) -> ErrorData:
    """Elicitation callback used in interrupt mode.

    Its real job is to exist: `ClientSession` advertises the elicitation
    capability based on whether a callback is registered, and servers refuse to
    elicit from a client that has not advertised it. On a 2026-07-28 connection
    it is never actually called, because the questions come back as an
    `InputRequiredResult` rather than as a request.

    Being called at all therefore means interrupting was not possible — either
    the connection is handshake-era, or the tool ran outside a graph — so it
    declines with the applicable explanation rather than hanging or silently
    answering.
    """
    reason = (
        INTERRUPT_ON_LEGACY_ERROR if in_langgraph_runtime() else OUTSIDE_GRAPH_ERROR
    )
    return ErrorData(code=INVALID_REQUEST, message=reason)


class UnansweredInputRequestError(RuntimeError):
    """A resumed interrupt did not supply an answer for every question asked."""

    def __init__(self, missing: list[str], supplied: list[str]) -> None:
        super().__init__(
            f"Resuming did not answer every input request the server asked for. "
            f"Missing: {sorted(missing)}. Supplied: {sorted(supplied)}. Resume "
            f"with a mapping keyed by request key, e.g. "
            f"`Command(resume={{{missing[0]!r}: {{'action': 'accept', "
            f"'content': {{...}}}}}})`."
        )
        self.missing = missing
        self.supplied = supplied


def _require_interrupt() -> Callable[[Any], Any]:
    """Import `langgraph.types.interrupt`, or explain why it is needed."""
    try:
        from langgraph.types import interrupt  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(LANGGRAPH_REQUIRED_ERROR) from e
    return interrupt


def _require_task() -> Callable[..., Any]:
    """Import `langgraph.func.task`, or explain why it is needed."""
    try:
        from langgraph.func import task  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(LANGGRAPH_REQUIRED_ERROR) from e
    return task


def describe_input_requests(
    requests: dict[str, InputRequest],
    *,
    server_name: str | None = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Build the JSON-safe interrupt payload for a set of input requests.

    This is what a human-facing application receives in
    `Interrupt.value`, so it is plain data rather than SDK models: the payload
    goes through the checkpointer, and pydantic models are not guaranteed to
    survive that round-trip.

    !!! warning

        `message` and `requested_schema` are written by the MCP server. Treat
        them as untrusted input before rendering them to a user or feeding them
        to a model.

    Args:
        requests: The questions the server embedded in its `InputRequiredResult`.
        server_name: Which configured server asked, when known.
        tool_name: Which tool was being called, when known.

    Returns:
        A JSON-serializable description of every question.
    """
    described: dict[str, Any] = {}
    for key, request in requests.items():
        if isinstance(request, ElicitRequest):
            params = request.params
            entry: dict[str, Any] = {
                "kind": "elicit",
                "mode": getattr(params, "mode", "form"),
                "message": params.message,
            }
            # URL-mode elicitation carries a destination instead of a schema;
            # the interaction happens out of band, deliberately not through us.
            if (schema := getattr(params, "requested_schema", None)) is not None:
                entry["requested_schema"] = schema
            if (url := getattr(params, "url", None)) is not None:
                entry["url"] = url
        elif isinstance(request, CreateMessageRequest):
            entry = {
                "kind": "sample",
                "params": request.params.model_dump(by_alias=True, mode="json"),
            }
        elif isinstance(request, ListRootsRequest):
            entry = {"kind": "list_roots"}
        else:  # pragma: no cover - the union has no other members
            entry = {"kind": "unknown", "method": getattr(request, "method", None)}
        described[key] = entry

    payload: dict[str, Any] = {"type": "mcp_input_required", "requests": described}
    if server_name is not None:
        payload["server"] = server_name
    if tool_name is not None:
        payload["tool"] = tool_name
    return payload


def _coerce_one(request: InputRequest, answer: Any) -> InputResponse:
    """Turn one resume value into the response type its question expects.

    Raises:
        TypeError: The answer cannot be read as a response to this question.
    """
    if isinstance(
        answer, ElicitResult | CreateMessageResult | ListRootsResult
    ):  # already typed
        return answer

    if isinstance(request, ElicitRequest):
        if isinstance(answer, str):
            # A bare "decline"/"cancel" is a natural thing to resume with.
            if answer in ("decline", "cancel"):
                return ElicitResult(action=answer)
            msg = (
                f"Cannot read {answer!r} as an elicitation response. Resume with "
                f"a content mapping, an ElicitResult, or 'decline'/'cancel'."
            )
            raise TypeError(msg)
        if isinstance(answer, dict):
            # Either a full {"action": ..., "content": ...} envelope, or just
            # the content the schema asked for.
            if "action" in answer:
                return ElicitResult.model_validate(answer)
            return ElicitResult(action="accept", content=answer)
        msg = (
            f"Cannot read {type(answer).__name__} as an elicitation response. "
            f"Resume with a content mapping, an ElicitResult, or "
            f"'decline'/'cancel'."
        )
        raise TypeError(msg)

    if isinstance(request, CreateMessageRequest):
        return CreateMessageResult.model_validate(answer)
    if isinstance(request, ListRootsRequest):
        return ListRootsResult.model_validate(answer)

    msg = (
        f"Unsupported input request type: {type(request).__name__}"  # pragma: no cover
    )
    raise TypeError(msg)


def coerce_resume_value(
    requests: dict[str, InputRequest],
    resumed: Any,
) -> InputResponses:
    """Map what a graph was resumed with onto the server's question keys.

    Accepts a mapping keyed by request key, or — when the server asked exactly
    one question — the bare answer to it.

    Raises:
        UnansweredInputRequestError: A question was left unanswered.
        TypeError: An answer could not be read as a response to its question.
    """
    if len(requests) == 1 and not (
        isinstance(resumed, dict) and set(resumed) <= set(requests)
    ):
        # Sole question: let the caller resume with the answer itself rather
        # than wrapping it in a single-entry mapping.
        (only_key,) = requests
        resumed = {only_key: resumed}

    if not isinstance(resumed, dict):
        msg = (
            f"Resuming an MCP input request needs a mapping keyed by request "
            f"key, got {type(resumed).__name__}. Keys asked for: "
            f"{sorted(requests)}."
        )
        raise TypeError(msg)

    if missing := [key for key in requests if key not in resumed]:
        raise UnansweredInputRequestError(missing, list(resumed))

    return {key: _coerce_one(requests[key], resumed[key]) for key in requests}


async def drive_input_required_via_interrupts(
    first: _ResultT | InputRequiredResult,
    retry: Callable[
        [InputResponses | None, str | None],
        Awaitable[_ResultT | InputRequiredResult],
    ],
    *,
    result_type: type[_ResultT],
    server_name: str | None = None,
    tool_name: str | None = None,
    max_rounds: int = DEFAULT_MAX_INPUT_ROUNDS,
) -> _ResultT:
    """Resolve an `InputRequiredResult` by interrupting the graph.

    Each round of questions becomes one `interrupt()`; each retry is memoized as
    a LangGraph task so resuming replays it from the checkpoint rather than
    re-asking the server.

    Args:
        first: What the original request returned. A terminal result passes
            straight through, so this is safe to wrap around any call.
        retry: Re-issues the original request with answers and the echoed
            `request_state`. Must be able to run against a *new* session: the
            connection is not expected to survive the interrupt.
        result_type: The terminal result model, used to rebuild it from the
            checkpointed JSON.
        server_name: Included in the interrupt payload, when known.
        tool_name: Included in the interrupt payload, when known.
        max_rounds: Rounds of questions tolerated before giving up.

    Returns:
        The terminal result.

    Raises:
        RuntimeError: `max_rounds` was exhausted.
    """
    if not isinstance(first, InputRequiredResult):
        return first

    interrupt = _require_interrupt()
    task = _require_task()

    @task
    async def retry_round(
        responses: InputResponses | None, request_state: str | None
    ) -> dict[str, Any]:
        # Memoized so a later interrupt in the same call does not re-issue this
        # round. Returned as plain JSON: pydantic models are not a supported
        # checkpoint payload.
        result = await retry(responses, request_state)
        return {
            "input_required": isinstance(result, InputRequiredResult),
            "data": result.model_dump(by_alias=True, mode="json"),
        }

    current: _ResultT | InputRequiredResult = first
    rounds = 0
    while isinstance(current, InputRequiredResult):
        rounds += 1
        if rounds > max_rounds:
            msg = (
                f"Server kept asking for input for more than {max_rounds} "
                f"rounds; giving up."
            )
            raise RuntimeError(msg)

        responses: InputResponses | None = None
        if current.input_requests:
            resumed = interrupt(
                describe_input_requests(
                    current.input_requests,
                    server_name=server_name,
                    tool_name=tool_name,
                )
            )
            responses = coerce_resume_value(current.input_requests, resumed)

        raw = await retry_round(responses, current.request_state)
        if raw["input_required"]:
            current = InputRequiredResult.model_validate(raw["data"])
        else:
            return result_type.model_validate(raw["data"])

    return current  # pragma: no cover - loop only exits via the returns above


__all__ = [
    "DEFAULT_MAX_INPUT_ROUNDS",
    "INTERRUPT_ON_LEGACY_ERROR",
    "LANGGRAPH_REQUIRED_ERROR",
    "OUTSIDE_GRAPH_ERROR",
    "UnansweredInputRequestError",
    "coerce_resume_value",
    "declare_elicitation_only",
    "describe_input_requests",
    "drive_input_required_via_interrupts",
    "in_langgraph_runtime",
]
