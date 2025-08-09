from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
    PromptArgument,
)

from langchain_mcp_adapters.prompts import (
    convert_mcp_prompt_message_to_langchain_message,
    load_mcp_prompt,
    list_mcp_prompts,
)


@pytest.mark.parametrize(
    "role,text,expected_cls",
    [("assistant", "Hello", AIMessage), ("user", "Hello", HumanMessage)],
)
def test_convert_mcp_prompt_message_to_langchain_message_with_text_content(
    role: str,
    text: str,
    expected_cls: type,
):
    message = PromptMessage(role=role, content=TextContent(type="text", text=text))
    result = convert_mcp_prompt_message_to_langchain_message(message)
    assert isinstance(result, expected_cls)
    assert result.content == text


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_convert_mcp_prompt_message_to_langchain_message_with_resource_content(
    role: str,
):
    message = PromptMessage(
        role=role,
        content=EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="message://greeting",
                mimeType="text/plain",
                text="hi",
            ),
        ),
    )
    with pytest.raises(ValueError):
        convert_mcp_prompt_message_to_langchain_message(message)


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_convert_mcp_prompt_message_to_langchain_message_with_image_content(role: str):
    message = PromptMessage(
        role=role,
        content=ImageContent(type="image", mimeType="image/png", data="base64data"),
    )
    with pytest.raises(ValueError):
        convert_mcp_prompt_message_to_langchain_message(message)


@pytest.mark.asyncio
async def test_load_mcp_prompt():
    session = AsyncMock()
    session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[
                PromptMessage(
                    role="user", content=TextContent(type="text", text="Hello")
                ),
                PromptMessage(
                    role="assistant", content=TextContent(type="text", text="Hi")
                ),
            ],
        ),
    )
    result = await load_mcp_prompt(session, "test_prompt")
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "Hi"

@pytest.mark.asyncio
async def test_list_mcp_prompts():
    greeting_mock = AsyncMock()
    greeting_mock.name = 'greeting'
    greeting_mock.description = 'A simple greeting prompt'
    greeting_mock.arguments = [
        PromptArgument(name="name", description="User's name", required=True),
        PromptArgument(name="language", description="Language code", required=False)
    ]
    
    summary_mock = AsyncMock()
    summary_mock.name = 'summary'
    summary_mock.description = 'Summarize text content'
    summary_mock.arguments = [
        PromptArgument(name="text", description="Text to summarize", required=True),
        PromptArgument(name="max_length", description="Maximum length", required=False)
    ]
    
    session = AsyncMock()
    session.list_prompts = AsyncMock(
        return_value=AsyncMock(
            prompts=[greeting_mock, summary_mock]
        )
    )
    
    result = await list_mcp_prompts(session)
    
    assert len(result) == 2
    assert result[0]["name"] == "greeting"
    assert result[0]["description"] == "A simple greeting prompt"
    assert len(result[0]["arguments"]) == 2
    assert result[0]["arguments"][0].name == "name"
    assert result[0]["arguments"][1].name == "language"
    assert result[1]["name"] == "summary"
    assert result[1]["description"] == "Summarize text content"
    assert len(result[1]["arguments"]) == 2
    assert result[1]["arguments"][0].name == "text"
    assert result[1]["arguments"][1].name == "max_length"