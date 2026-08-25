"""Multi-round-trip input resolution for MCP protocol revision 2026-07-28+.

Through 2025-11-25, a server that needed input mid-call (elicitation, sampling,
a roots listing) sent a server-to-client request over the open connection and
blocked until the client answered. 2026-07-28 removes server-initiated requests
entirely: instead the server *returns* an
[`InputRequiredResult`][mcp.types.InputRequiredResult] in place of the normal
result, carrying the questions it needs answered plus an opaque `request_state`
token. The client answers the questions and re-issues the original request with
those answers and the token echoed back verbatim (SEP-2322).

This module drives that loop. The `dispatch` seam decides *how* a question gets
answered; the default routes each one through the session's own callback table,
so a `Callbacks` instance behaves identically on both sides of the protocol
boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from mcp.client import ClientRequestContext
from mcp.client._input_required import run_input_required_driver
from mcp.types import InputRequiredResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp import ClientSession
    from mcp.types import ErrorData, InputRequest, InputResponse, InputResponses

DEFAULT_MAX_INPUT_ROUNDS = 10
"""Rounds of `InputRequiredResult` tolerated before giving up.

Matches the MCP SDK default and the TypeScript, C# and Go clients. Bounded so
that a server which never stops asking cannot hold a tool call open forever.
"""

_ResultT = TypeVar("_ResultT")


class InputRequestDispatcher(Protocol):
    """Answers one question a server embedded in an `InputRequiredResult`.

    Returning [`ErrorData`][mcp.types.ErrorData] declines the question and
    aborts the surrounding call.
    """

    async def __call__(
        self,
        session: ClientSession,
        key: str,
        request: InputRequest,
    ) -> InputResponse | ErrorData:
        """Answer the server-assigned question `key`."""
        ...


async def dispatch_via_callbacks(
    session: ClientSession,
    key: str,
    request: InputRequest,
) -> InputResponse | ErrorData:
    """Answer a question through the session's own callback table.

    Routes to the same elicitation / sampling / roots callbacks the SDK uses for
    handshake-era server-to-client requests, so a caller's `Callbacks` behave
    the same whichever revision the connection negotiated.
    """
    context = ClientRequestContext(
        session=session,
        request_id=key,
        meta=request.params.meta if request.params else None,
    )
    return await session.dispatch_input_request(context, request)


async def resolve_input_required(
    session: ClientSession,
    first: _ResultT | InputRequiredResult,
    retry: Callable[
        [InputResponses | None, str | None],
        Awaitable[_ResultT | InputRequiredResult],
    ],
    *,
    dispatch: InputRequestDispatcher = dispatch_via_callbacks,
    max_rounds: int = DEFAULT_MAX_INPUT_ROUNDS,
) -> _ResultT:
    """Resolve an `InputRequiredResult` to the terminal result it stands in for.

    A result that is already terminal passes straight through, so this is safe
    to wrap around every interactive call regardless of negotiated revision.

    Args:
        session: The session the original request was issued on.
        first: What the original request returned.
        retry: Re-issues the original request with the collected answers and
            the echoed `request_state`.
        dispatch: How to answer one question. Defaults to the session's
            callback table.
        max_rounds: Rounds tolerated before raising.

    Returns:
        The terminal result.

    Raises:
        InputRequiredRoundsExceededError: `max_rounds` was exhausted.
        McpError: A dispatcher declined a question.
    """
    if not isinstance(first, InputRequiredResult):
        return first

    async def dispatch_one(
        key: str, request: InputRequest
    ) -> InputResponse | ErrorData:
        return await dispatch(session, key, request)

    return await run_input_required_driver(
        first,
        dispatch=dispatch_one,
        retry=retry,
        max_rounds=max_rounds,
    )


__all__ = [
    "DEFAULT_MAX_INPUT_ROUNDS",
    "InputRequestDispatcher",
    "dispatch_via_callbacks",
    "resolve_input_required",
]
