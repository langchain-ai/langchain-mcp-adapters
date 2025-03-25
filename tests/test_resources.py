from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document
from mcp.types import (
    BlobResourceContents,
    ListResourcesResult,
    ReadResourceResult,
    Resource,
    TextResourceContents,
)

from langchain_mcp_adapters.resources import (
    convert_mcp_resource_to_langchain_document,
    get_mcp_resource,
    load_mcp_resources,
)


def test_convert_mcp_resource_to_langchain_document_with_text():
    uri = "file:///test.txt"
    contents = TextResourceContents(
        uri=uri,
        mimeType="text/plain",
        text="Hello, world!"
    )

    doc = convert_mcp_resource_to_langchain_document(uri, contents)

    assert isinstance(doc, Document)
    assert doc.page_content == "Hello, world!"
    assert doc.metadata["uri"] == uri
    assert doc.metadata["mime_type"] == "text/plain"


def test_convert_mcp_resource_to_langchain_document_with_blob():
    uri = "file:///test.png"
    contents = BlobResourceContents(
        uri=uri,
        mimeType="image/png",
        blob="base64data"
    )

    doc = convert_mcp_resource_to_langchain_document(uri, contents)

    assert isinstance(doc, Document)
    assert doc.page_content == "Binary data of type: image/png"
    assert doc.metadata["uri"] == uri
    assert doc.metadata["mime_type"] == "image/png"
    assert doc.metadata["blob_data"] == "base64data"


@pytest.mark.asyncio
async def test_get_mcp_resource_with_contents():
    session = AsyncMock()
    uri = "file:///test.txt"

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mimeType="text/plain",
                    text="Content 1"
                ),
                TextResourceContents(
                    uri=uri,
                    mimeType="text/plain",
                    text="Content 2"
                )
            ]
        )
    )

    docs = await get_mcp_resource(session, uri)

    assert len(docs) == 2
    assert docs[0].page_content == "Content 1"
    assert docs[1].page_content == "Content 2"
    assert docs[0].metadata["uri"] == uri
    assert docs[1].metadata["uri"] == uri
    session.read_resource.assert_called_once_with(uri)


@pytest.mark.asyncio
async def test_get_mcp_resource_with_empty_contents():
    session = AsyncMock()
    uri = "file:///empty.txt"

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(contents=[])
    )

    docs = await get_mcp_resource(session, uri)

    assert len(docs) == 0
    session.read_resource.assert_called_once_with(uri)


@pytest.mark.asyncio
async def test_load_mcp_resources_with_list_of_uris():
    session = AsyncMock()
    uri1 = "file:///test1.txt"
    uri2 = "file:///test2.txt"

    session.read_resource = AsyncMock()
    session.read_resource.side_effect = [
        ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri1,
                    mimeType="text/plain",
                    text="Content from test1"
                )
            ]
        ),
        ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri2,
                    mimeType="text/plain",
                    text="Content from test2"
                )
            ]
        )
    ]

    docs = await load_mcp_resources(session, [uri1, uri2])

    assert len(docs) == 2
    assert docs[0].page_content == "Content from test1"
    assert docs[1].page_content == "Content from test2"
    assert docs[0].metadata["uri"] == uri1
    assert docs[1].metadata["uri"] == uri2
    assert session.read_resource.call_count == 2


@pytest.mark.asyncio
async def test_load_mcp_resources_with_single_uri_string():
    session = AsyncMock()
    uri = "file:///test.txt"

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mimeType="text/plain",
                    text="Content from test"
                )
            ]
        )
    )

    docs = await load_mcp_resources(session, uri)

    assert len(docs) == 1
    assert docs[0].page_content == "Content from test"
    assert docs[0].metadata["uri"] == uri
    session.read_resource.assert_called_once_with(uri)


@pytest.mark.asyncio
async def test_load_mcp_resources_with_all_resources():
    session = AsyncMock()

    session.list_resources = AsyncMock(
        return_value=ListResourcesResult(
            resources=[
                Resource(
                    uri="file:///test1.txt",
                    name="test1.txt",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="file:///test2.txt",
                    name="test2.txt",
                    mimeType="text/plain"
                )
            ]
        )
    )

    session.read_resource = AsyncMock()
    session.read_resource.side_effect = [
        ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="file:///test1.txt",
                    mimeType="text/plain",
                    text="Content from test1"
                )
            ]
        ),
        ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="file:///test2.txt",
                    mimeType="text/plain",
                    text="Content from test2"
                )
            ]
        )
    ]

    docs = await load_mcp_resources(session)

    assert len(docs) == 2
    assert docs[0].page_content == "Content from test1"
    assert docs[1].page_content == "Content from test2"
    assert session.list_resources.called
    assert session.read_resource.call_count == 2


@pytest.mark.asyncio
async def test_load_mcp_resources_with_error_handling():
    session = AsyncMock()
    uri1 = "file:///valid.txt"
    uri2 = "file:///error.txt"

    session.read_resource = AsyncMock()
    session.read_resource.side_effect = [
        ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri1,
                    mimeType="text/plain",
                    text="Valid content"
                )
            ]
        ),
        Exception("Resource not found")
    ]

    docs = await load_mcp_resources(session, [uri1, uri2])

    assert len(docs) == 1
    assert docs[0].page_content == "Valid content"
    assert docs[0].metadata["uri"] == uri1
    assert session.read_resource.call_count == 2
