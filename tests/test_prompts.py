from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from langchain_mcp_adapters.prompts import (
    UnsupportedContentError,
    convert_mcp_prompt_message_to_langchain_message,
    load_mcp_prompt,
)


def test_convert_mcp_prompt_message_to_langchain_message_text_user():
    message = PromptMessage(role="user", content=TextContent(type="text", text="Hello"))
    result = convert_mcp_prompt_message_to_langchain_message(message)
    assert isinstance(result, HumanMessage)
    assert result.content == "Hello"


def test_convert_mcp_prompt_message_to_langchain_message_text_assistant():
    message = PromptMessage(role="assistant", content=TextContent(type="text", text="Hello"))
    result = convert_mcp_prompt_message_to_langchain_message(message)
    assert isinstance(result, AIMessage)
    assert result.content == "Hello"


def test_convert_mcp_prompt_message_to_langchain_message_image_user():
    message = PromptMessage(
        role="user", content=ImageContent(type="image", mimeType="image/png", data="base64data")
    )
    result = convert_mcp_prompt_message_to_langchain_message(message)
    assert isinstance(result, HumanMessage)
    assert result.content[0]["type"] == "image_url"
    assert result.content[0]["image_url"]["url"] == "data:image/png;base64,base64data"


def test_convert_mcp_prompt_message_to_langchain_message_image_assistant():
    message = PromptMessage(
        role="assistant",
        content=ImageContent(type="image", mimeType="image/png", data="base64data"),
    )
    result = convert_mcp_prompt_message_to_langchain_message(message)
    assert isinstance(result, AIMessage)
    assert result.content[0]["type"] == "image_url"
    assert result.content[0]["image_url"]["url"] == "data:image/png;base64,base64data"


def test_convert_mcp_prompt_message_to_langchain_message_unsupported_content():
    message = PromptMessage(
        role="user",
        content=EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="message://greeting", mimeType="text/plain", text="hi"
            ),
        ),
    )
    with pytest.raises(UnsupportedContentError):
        convert_mcp_prompt_message_to_langchain_message(message)


@pytest.mark.asyncio
async def test_load_mcp_prompt():
    session = AsyncMock()
    session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
                PromptMessage(role="assistant", content=TextContent(type="text", text="Hi")),
            ]
        )
    )
    result = await load_mcp_prompt(session, "test_prompt", None)
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "Hi"
