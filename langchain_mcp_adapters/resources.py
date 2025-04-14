import base64

from langchain_core.documents.base import Blob
from mcp import ClientSession
from mcp.types import BlobResourceContents, ResourceContents, TextResourceContents


def convert_mcp_resource_to_langchain_blob(
    resource_uri: str,
    contents: ResourceContents,
) -> Blob:
    """Convert an MCP resource content to a LangChain Blob.

    Args:
        resource_uri: URI of the resource
        contents: The resource contents

    Returns:
        A LangChain Blob
    """
    if isinstance(contents, TextResourceContents):
        data = contents.text
    elif isinstance(contents, BlobResourceContents):
        data = base64.b64decode(contents.blob)
    else:
        raise ValueError(f"Unsupported content type for URI {resource_uri}")

    return Blob.from_data(
        data=data,
        mime_type=contents.mimeType,
        metadata={"uri": resource_uri},
    )


async def get_mcp_resource(session: ClientSession, uri: str) -> list[Blob]:
    """Fetch a single MCP resource and convert it to LangChain Blobs.

    Args:
        session: MCP client session
        uri: URI of the resource to fetch

    Returns:
        A list of LangChain Blobs
    """
    contents_result = await session.read_resource(uri)

    documents = []

    if not contents_result.contents or len(contents_result.contents) == 0:
        return documents

    for content in contents_result.contents:
        doc = convert_mcp_resource_to_langchain_blob(uri, content)
        documents.append(doc)

    return documents


async def load_mcp_resources(
    session: ClientSession,
    uris: str | list[str] | None = None,
) -> list[Blob]:
    """Load MCP resources and convert them to LangChain Blobs"""
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
            raise RuntimeError(f"Error fetching resource {uri}") from e

    return documents
