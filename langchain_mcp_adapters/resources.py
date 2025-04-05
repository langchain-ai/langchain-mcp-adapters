import base64

from langchain_core.documents import Document
from langchain_core.documents.base import Blob as LCBlob
from mcp import ClientSession
from mcp.types import BlobResourceContents, ResourceContents, TextResourceContents


def convert_mcp_resource_to_langchain_document(
    resource_uri: str,
    contents: ResourceContents,
) -> Document | LCBlob:
    """Convert an MCP resource content to a LangChain Document or Blob.

    Args:
        resource_uri: URI of the resource
        contents: The resource contents

    Returns:
        A LangChain Document for text content, or a Blob for binary content
    """
    if isinstance(contents, TextResourceContents):
        return Document(
            page_content=contents.text,
            metadata={
                "uri": resource_uri,
                "mime_type": contents.mimeType,
            }
        )

    elif isinstance(contents, BlobResourceContents):
        raw_bytes = base64.b64decode(contents.blob)

        return LCBlob.from_data(
            data=raw_bytes,
            mime_type=contents.mimeType,
            metadata={
                "uri": resource_uri,
            }
        )

    raise ValueError(f"Unsupported content type for URI {resource_uri}")

async def get_mcp_resource(
    session: ClientSession,
    uri: str
) -> list[Document | LCBlob]:
    """Fetch a single MCP resource and convert it to LangChain Documents or Blobs.

    Args:
        session: MCP client session
        uri: URI of the resource to fetch

    Returns:
        A list of LangChain Documents or Blobs
    """
    contents_result = await session.read_resource(uri)

    documents = []

    if not contents_result.contents or len(contents_result.contents) == 0:
        return documents

    for content in contents_result.contents:
        doc = convert_mcp_resource_to_langchain_document(uri, content)
        documents.append(doc)

    return documents


async def load_mcp_resources(
    session: ClientSession,
    uris: str | list[str] | None = None,
) -> list[Document | LCBlob]:
    """Load MCP resources and convert them to LangChain Documents or Blobs"""
    documents = []

    if uris is None:
        resources_list = await session.list_resources()
        uri_list = [r.uri for r in resources_list.resources]
    elif isinstance(uris, str):
        uri_list = [uris]
    else:
        uri_list = uris

    for uri in uri_list:
        try:
            resource_docs = await get_mcp_resource(session, uri)
            documents.extend(resource_docs)
        except Exception as e:
            print(f"Error fetching resource {uri}: {e}")

    return documents
