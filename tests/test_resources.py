import base64
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document
from langchain_core.documents.base import Blob as LCBlob
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
    original_data = b"binary-image-data"
    base64_blob = base64.b64encode(original_data).decode()

    contents = BlobResourceContents(
        uri=uri,
        mimeType="image/png",
        blob=base64_blob
    )

    blob = convert_mcp_resource_to_langchain_document(uri, contents)

    assert isinstance(blob, LCBlob)
    assert blob.data == original_data
    assert blob.mimetype == "image/png"
    assert blob.metadata["uri"] == uri


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
async def test_get_mcp_resource_with_text_and_blob():
    session = AsyncMock()
    uri = "file:///mixed"

    original_data = b"some-binary-content"
    base64_blob = base64.b64encode(original_data).decode()

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mimeType="text/plain",
                    text="Hello Text"
                ),
                BlobResourceContents(
                    uri=uri,
                    mimeType="application/octet-stream",
                    blob=base64_blob
                )
            ]
        )
    )

    results = await get_mcp_resource(session, uri)

    assert len(results) == 2
    assert isinstance(results[0], Document)
    assert results[0].page_content == "Hello Text"
    assert isinstance(results[1], LCBlob)
    assert results[1].data == original_data
    assert results[1].mimetype == "application/octet-stream"

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


@pytest.mark.asyncio
async def test_load_mcp_resources_with_blob_content():
    session = AsyncMock()
    uri = "file:///with_blob"
    original_data = b"binary data"
    base64_blob = base64.b64encode(original_data).decode()

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                BlobResourceContents(
                    uri=uri,
                    mimeType="application/octet-stream",
                    blob=base64_blob
                )
            ]
        )
    )

    docs = await load_mcp_resources(session, uri)

    assert len(docs) == 1
    assert isinstance(docs[0], LCBlob)
    assert docs[0].data == original_data
    assert docs[0].mimetype == "application/octet-stream"
