from langchain_core.documents import Document
from mcp import ClientSession
from mcp.types import BlobResourceContents, ResourceContents, TextResourceContents


def convert_mcp_resource_to_langchain_document(
    resource_uri: str,
    contents: ResourceContents,
) -> Document:
    """Convert an MCP resource content to a LangChain Document.

    Args:
        resource_uri: URI of the resource
        contents: The resource contents

    Returns:
        A LangChain Document
    """
    metadata = {
        "uri": resource_uri,
        "mime_type": contents.mimeType,
    }

    content = ""

    if isinstance(contents, TextResourceContents):
        content = contents.text
    elif isinstance(contents, BlobResourceContents):
        metadata["blob_data"] = contents.blob
        content = f"Binary data of type: {contents.mimeType}"

    return Document(page_content=content, metadata=metadata)

async def get_mcp_resource(
    session: ClientSession,
    uri: str
) -> list[Document]:
    """Fetch a single MCP resource and convert it to LangChain Documents.

    Args:
        session: MCP client session
        uri: URI of the resource to fetch

    Returns:
        A list of LangChain Documents, one for each content in the resource
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
) -> list[Document]:
    """Load MCP resources and convert them to LangChain Documents.

    This is the main function that should be used by external code.

    Args:
        session: MCP client session
        uris: Optional resource URI or list of URIs to load. If not provided, all resources will be loaded.

    Returns:
        A list of LangChain Documents
    """
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
