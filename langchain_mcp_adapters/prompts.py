"""Prompts adapter for MCP `prompts/list` and `prompts/get`.

Maps MCP prompt operations to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages):

- `list_all_mcp_prompts` — walks every page of the MCP
  [`prompts/list`](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts#listing-prompts)
  request (metadata only: names, descriptions, argument schemas).
- `load_mcp_prompt` — calls MCP
  [`prompts/get`](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts#getting-a-prompt)
  and converts returned messages to LangChain types.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession
from mcp.types import Prompt, PromptMessage

MAX_ITERATIONS = 1000


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


async def list_all_mcp_prompts(session: ClientSession) -> list[Prompt]:
    """List every prompt template from the server via MCP `prompts/list` (paginated).

    Repeatedly calls `ClientSession.list_prompts` until `nextCursor` is absent,
    per the MCP pagination rules.

    Args:
        session: Initialized MCP client session.

    Returns:
        All `Prompt` metadata entries (no rendered prompt content).

    Raises:
        RuntimeError: If pagination exceeds `MAX_ITERATIONS`.

    """
    current_cursor: str | None = None
    all_prompts: list[Prompt] = []
    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            msg = f"Reached max of {MAX_ITERATIONS} iterations while listing prompts."
            raise RuntimeError(msg)

        list_prompts_page_result = await session.list_prompts(cursor=current_cursor)

        if list_prompts_page_result.prompts:
            all_prompts.extend(list_prompts_page_result.prompts)

        if not list_prompts_page_result.nextCursor:
            break

        current_cursor = list_prompts_page_result.nextCursor

    return all_prompts


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
    response = await session.get_prompt(name, arguments)
    return [
        convert_mcp_prompt_message_to_langchain_message(message)
        for message in response.messages
    ]
