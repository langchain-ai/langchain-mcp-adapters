from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    ListPromptsResult,
    Prompt,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from langchain_mcp_adapters.prompts import (
    convert_mcp_prompt_message_to_langchain_message,
    list_all_mcp_prompts,
    load_mcp_prompt,
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


async def test_list_all_mcp_prompts_paginates():
    """Exhaust MCP prompts/list pagination (nextCursor)."""
    session = AsyncMock()
    first = Prompt(name="page_a", description=None, arguments=[])
    second = Prompt(name="page_b", description=None, arguments=[])
    session.list_prompts = AsyncMock(
        side_effect=[
            ListPromptsResult(prompts=[first], nextCursor="cursor-2"),
            ListPromptsResult(prompts=[second], nextCursor=None),
        ],
    )

    result = await list_all_mcp_prompts(session)

    assert [p.name for p in result] == ["page_a", "page_b"]
    assert session.list_prompts.await_count == 2
    session.list_prompts.assert_any_await(cursor=None)
    session.list_prompts.assert_awaited_with(cursor="cursor-2")
