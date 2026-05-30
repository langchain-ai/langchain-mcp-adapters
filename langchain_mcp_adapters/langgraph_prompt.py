"""LangGraph helpers for MCP ``prompts/get`` (prompt injection nodes)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import BaseMessage, RemoveMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_mcp_adapters.client import MultiServerMCPClient

Placement = Literal["prepend", "append"]


def create_mcp_prompt_injection_node(
    client: MultiServerMCPClient,
    server_name: str,
    prompt_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    arguments_resolver: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    placement: Placement = "prepend",
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build an async LangGraph node that fetches MCP prompt content into ``messages``.

    Uses MCP ``prompts/get`` via ``MultiServerMCPClient.get_prompt``. Intended for
    ``StateGraph`` states whose ``messages`` channel uses ``add_messages`` (e.g.
    ``MessagesState`` from ``langgraph.graph.message``).

    Args:
        client: ``MultiServerMCPClient`` instance.
        server_name: Connection name configured on the client.
        prompt_name: MCP prompt template name.
        arguments: Static template arguments. Ignored when ``arguments_resolver`` is
            set.
        arguments_resolver: Callable mapping ``state`` to template argument dicts.
        placement: ``"prepend"`` inserts MCP messages before history (remove-all +
            replay). ``"append"`` appends MCP messages after existing ones.

    Returns:
        Async ``node(state) -> {"messages": ...}`` for ``add_node("inject", node)``.

    Raises:
        ImportError: If ``langgraph`` is not installed.

    !!! example

        ```python
        from langgraph.graph import END, MessagesState, START, StateGraph

        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.langgraph_prompt import (
            create_mcp_prompt_injection_node,
        )

        client = MultiServerMCPClient({...})
        inject = create_mcp_prompt_injection_node(
            client,
            "math",
            "configure_assistant",
            arguments={"skills": "algebra"},
        )

        builder = StateGraph(MessagesState)
        builder.add_node("mcp_prompt", inject)
        builder.add_edge(START, "mcp_prompt")
        builder.add_edge("mcp_prompt", END)
        graph = builder.compile()
        ```

    """
    try:
        from langgraph.graph.message import REMOVE_ALL_MESSAGES  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "langgraph is required for create_mcp_prompt_injection_node. "
            "Install with: pip install langgraph"
        )
        raise ImportError(msg) from e

    async def mcp_prompt_injection_node(state: dict[str, Any]) -> dict[str, Any]:
        if arguments_resolver is not None:
            resolved = arguments_resolver(state)
        else:
            resolved = arguments
        extra = await client.get_prompt(
            server_name,
            prompt_name,
            arguments=resolved,
        )
        existing: list[BaseMessage] = list(state.get("messages", []))
        if placement == "append":
            return {"messages": extra}
        prepend_payload: list[BaseMessage] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *extra,
            *existing,
        ]
        return {"messages": prepend_payload}

    return mcp_prompt_injection_node
