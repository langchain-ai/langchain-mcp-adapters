from typing import Any, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession
from mcp.types import PromptMessage


class UnsupportedContentError(Exception):
    """Raised when a prompt message contains unsupported content."""

    pass


class UnsupportedRoleError(Exception):
    """Raised when a prompt message contains an unsupported role."""

    pass


def convert_mcp_prompt_message_to_langchain_message(
    message: PromptMessage,
) -> Union[HumanMessage, AIMessage]:
    """Convert an MCP prompt message to a LangChain message.

    Args:
        message: MCP prompt message to convert

    Returns:
        a LangChain message
    """
    if message.content.type == "text":
        if message.role == "user":
            return HumanMessage(content=message.content.text)
        elif message.role == "assistant":
            return AIMessage(content=message.content.text)
        else:
            raise UnsupportedRoleError(
                f"Unsupported prompt message role: {message.role}")

    raise UnsupportedContentError(
        f"Unsupported prompt message content type: {message.content.type}"
    )


async def load_mcp_prompt(
    session: ClientSession, name: str, arguments: Optional[dict[str, Any]] = None
) -> list[Union[HumanMessage, AIMessage]]:
    """Load MCP prompt and convert to LangChain messages."""
    response = await session.get_prompt(name, arguments)
    return [
        convert_mcp_prompt_message_to_langchain_message(message) for message in response.messages
    ]
