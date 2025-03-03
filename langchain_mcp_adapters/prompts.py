from typing import Any, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession
from mcp.types import PromptMessage


class UnsupportedContentError(Exception):
    """Raised when a prompt message contains unsupported content."""

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
        if message.role == "assistant":
            return AIMessage(content=message.content.text)

    if message.content.type == "image":
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{message.content.mimeType};base64,{message.content.data}"},
        }
        if message.role == "user":
            return HumanMessage(content=[image_content])
        if message.role == "assistant":
            return AIMessage(content=[image_content])

    raise UnsupportedContentError(
        f"Unsupported prompt message content type: {message.content.type}"
    )


async def load_mcp_prompt(
    session: ClientSession, name: str, arguments: Optional[dict[str, Any]]
) -> list[Union[HumanMessage, AIMessage]]:
    """Load MCP prompt and convert to LangChain messages."""
    response = await session.get_prompt(name, arguments)
    results = map(convert_mcp_prompt_message_to_langchain_message, response.messages)
    return list(results)
