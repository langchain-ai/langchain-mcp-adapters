"""Prompts adapter for converting MCP prompts to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages).

This module provides functionality to convert MCP prompt messages into LangChain
message objects, handling both user and assistant message types.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession
from mcp.client import ClientRequestContext
from mcp.client._input_required import run_input_required_driver
from mcp.types import InputRequiredResult, PromptMessage


def convert_mcp_prompt_message_to_langchain_message(
    message: PromptMessage,
) -> HumanMessage | AIMessage:
    """Convert an MCP prompt message to a LangChain message.

    Args:
        message: MCP prompt message to convert

    Returns:
        A LangChain message

    """
    if message.content.type == "text":
        if message.role == "user":
            return HumanMessage(content=message.content.text)
        if message.role == "assistant":
            return AIMessage(content=message.content.text)
        msg = f"Unsupported prompt message role: {message.role}"
        raise ValueError(msg)

    msg = f"Unsupported prompt message content type: {message.content.type}"
    raise ValueError(msg)


async def load_mcp_prompt(
    session: ClientSession,
    name: str,
    *,
    arguments: dict[str, Any] | None = None,
) -> list[HumanMessage | AIMessage]:
    """Load MCP prompt and convert to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages).

    Args:
        session: The MCP client session.
        name: Name of the prompt to load.
        arguments: Optional arguments to pass to the prompt.

    Returns:
        A list of LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages)
            converted from the MCP prompt.
    """
    response = await _resolve(
        session,
        lambda r, st: session.get_prompt(
            name,
            arguments,
            input_responses=r,
            request_state=st,
            allow_input_required=True,
        ),
    )
    return [
        convert_mcp_prompt_message_to_langchain_message(message)
        for message in response.messages
    ]


async def _resolve(session, attempt):
    """Run an interactive request, answering any input the server asks for.

    On 2026-07-28 a server may return an `InputRequiredResult` in place of the
    real result; the SDK driver answers each question through the session's
    callback table and retries.
    """
    first = await attempt(None, None)
    if not isinstance(first, InputRequiredResult):
        return first
    return await run_input_required_driver(
        first,
        dispatch=lambda key, req: session.dispatch_input_request(
            ClientRequestContext(session=session, request_id=key, meta=None), req
        ),
        retry=attempt,
    )
