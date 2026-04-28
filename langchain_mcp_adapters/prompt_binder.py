"""Compose MCP ``prompts/get`` with a chat model like ``bind_tools`` composes tools.

MCP has no protocol equivalent to tool-binding on the model: templates are fetched
with ``prompts/get`` and become plain messages. This module exposes a **Runnable**
that prepends those messages then delegates to the model, so you can write
``bind_mcp_prompt(...)`` in the same spirit as ``model.bind_tools(tools)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import BaseMessage, convert_to_messages
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableLambda

if TYPE_CHECKING:
    from langchain_core.language_models.base import LanguageModelInput

    from langchain_mcp_adapters.client import MultiServerMCPClient

_Out = TypeVar("_Out")


def _messages_from_input(model_input: object) -> list[BaseMessage]:
    """Normalize chat model input to a list of messages."""
    if isinstance(model_input, dict) and "messages" in model_input:
        raw = model_input["messages"]
        return convert_to_messages(raw)
    if isinstance(model_input, PromptValue):
        return list(model_input.to_messages())
    if isinstance(model_input, str):
        return convert_to_messages([model_input])
    if isinstance(model_input, BaseMessage):
        return [model_input]
    if isinstance(model_input, Sequence):
        return convert_to_messages(model_input)
    msg = (
        "Unsupported input for bind_mcp_prompt: expected str, BaseMessage, "
        "sequence of messages, dict with 'messages', or PromptValue. "
        f"Got {type(model_input)!r}."
    )
    raise TypeError(msg)


def _merge_input_with_messages(
    model_input: object,
    messages: list[BaseMessage],
) -> LanguageModelInput:
    """Rebuild the same input shape with ``messages`` replaced / inserted."""
    if isinstance(model_input, dict) and "messages" in model_input:
        merged_dict = dict(model_input)
        merged_dict["messages"] = messages
        return merged_dict
    if isinstance(model_input, PromptValue):
        return ChatPromptValue(messages=messages)
    return messages


def bind_mcp_prompt(
    model: Runnable[LanguageModelInput, _Out],
    *,
    client: MultiServerMCPClient,
    server_name: str,
    prompt_name: str,
    arguments: dict[str, Any] | None = None,
    arguments_resolver: Callable[[object], dict[str, Any] | None] | None = None,
) -> Runnable[LanguageModelInput, _Out]:
    """Prepend MCP ``prompts/get`` messages, then run ``model``.

    Mirrors the tools pattern: ``bind_tools`` attaches tool schemas; here we
    **materialize** MCP prompt messages and prepend them before user / graph
    messages.

    Args:
        model: A LangChain chat model runnable (``BaseChatModel`` or bound variant).
        client: Client used for ``get_prompt`` (MCP ``prompts/get``).
        server_name: MCP server key in the client configuration.
        prompt_name: Template name on that server.
        arguments: Static ``prompts/get`` arguments. Ignored if ``arguments_resolver``
            is set.
        arguments_resolver: ``(model_input) -> dict | None`` for per-call arguments
            (e.g. read fields from a LangGraph state dict).

    Returns:
        A runnable with the same input and output types as ``model``.

    !!! note

        MCP I/O is async: prefer ``await bound.ainvoke(...)``. Sync ``invoke`` only
        works if your stack runs async ``RunnableLambda`` correctly.

    !!! example "LangGraph-style state"

        ```python
        bound = bind_mcp_prompt(
            model,
            client=client,
            server_name="knowledge",
            prompt_name="classify_intent",
            arguments_resolver=lambda s: {"domain": s["domain"]},
        )
        await bound.ainvoke({"messages": state["messages"], "domain": "legal"})
        ```

    """

    async def _prepend_prompt(model_input: object) -> LanguageModelInput:
        if arguments_resolver is not None:
            resolved = arguments_resolver(model_input)
        else:
            resolved = arguments
        extra = await client.get_prompt(
            server_name,
            prompt_name,
            arguments=resolved,
        )
        base = _messages_from_input(model_input)
        merged = [*extra, *base]
        return _merge_input_with_messages(model_input, merged)

    return RunnableLambda(_prepend_prompt) | model
