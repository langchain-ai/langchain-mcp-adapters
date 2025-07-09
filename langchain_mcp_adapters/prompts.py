from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession
from mcp.types import PromptMessage


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
    """Load MCP prompt and convert to LangChain messages."""
    response = await session.get_prompt(name, arguments)
    return [
        convert_mcp_prompt_message_to_langchain_message(message) for message in response.messages
    ]
