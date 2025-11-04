"""Resources adapter for converting MCP resources to LangChain [Blob objects][langchain_core.documents.base.Blob].

This module provides functionality to convert MCP resources into LangChain Blob
objects, handling both text and binary resource content types.
"""  # noqa: E501

import base64
from typing import Any

from langchain_core.documents.base import Blob
from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession
from mcp.types import BlobResourceContents, ResourceContents, TextResourceContents
from pydantic import BaseModel, Field
from langchain_mcp_adapters.sessions import Connection
from langchain_core.callbacks import Callbacks

def convert_mcp_resource_to_langchain_blob(
    resource_uri: str, contents: ResourceContents
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
        msg = f"Unsupported content type for URI {resource_uri}"
        raise TypeError(msg)

    return Blob.from_data(
        data=data, mime_type=contents.mimeType, metadata={"uri": resource_uri}
    )


async def get_mcp_resource(session: ClientSession, uri: str) -> list[Blob]:
    """Fetch a single MCP resource and convert it to LangChain [Blob objects][langchain_core.documents.base.Blob].

    Args:
        session: MCP client session.
        uri: URI of the resource to fetch.

    Returns:
        A list of LangChain [Blob][langchain_core.documents.base.Blob] objects.
    """  # noqa: E501
    contents_result = await session.read_resource(uri)
    if not contents_result.contents or len(contents_result.contents) == 0:
        return []

    return [
        convert_mcp_resource_to_langchain_blob(uri, content)
        for content in contents_result.contents
    ]


async def load_mcp_resources(
    session: ClientSession,
    *,
    uris: str | list[str] | None = None,
) -> list[Blob]:
    """Load MCP resources and convert them to LangChain [Blob objects][langchain_core.documents.base.Blob].

    Args:
        session: MCP client session.
        uris: List of URIs to load. If `None`, all resources will be loaded.

            !!! note

                Dynamic resources will NOT be loaded when `None` is specified,
                as they require parameters and are ignored by the MCP SDK's
                `session.list_resources()` method.

    Returns:
        A list of LangChain [Blob][langchain_core.documents.base.Blob] objects.

    Raises:
        RuntimeError: If an error occurs while fetching a resource.
    """  # noqa: E501
    blobs = []

    if uris is None:
        resources_list = await session.list_resources()
        uri_list = [r.uri for r in resources_list.resources]
    elif isinstance(uris, str):
        uri_list = [uris]
    else:
        uri_list = uris

    current_uri = None
    try:
        for uri in uri_list:
            current_uri = uri
            resource_blobs = await get_mcp_resource(session, uri)
            blobs.extend(resource_blobs)
    except Exception as e:
        msg = f"Error fetching resource {current_uri}"
        raise RuntimeError(msg) from e

    return blobs


# Pydantic schemas for tool arguments
class ListResourcesInput(BaseModel):
    """Input schema for list_resources tool."""

    cursor: str | None = Field(
        default=None,
        description="Pagination cursor returned from a previous list_resources call. "
        "If provided, returns the next page of results.",
    )


class ReadResourceInput(BaseModel):
    """Input schema for read_resource tool."""

    uri: str = Field(
        description="The URI of the resource to read. "
        "Use list_resources to discover available resource URIs."
    )


async def load_mcp_resources_as_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
    server_name: str | None = None,
) -> list[BaseTool]:
    """Load MCP resources as LangChain tools for listing and reading resources.

    This function creates two tools that an agent can use to interact with MCP resources:
    - `list_resources`: Lists available resources with pagination support
    - `read_resource`: Reads a specific resource by URI and returns its contents

    Args:
        session: The MCP client session. If `None`, connection must be provided.
        connection: Connection config to create a new session if session is `None`.
        callbacks: Optional `Callbacks` for handling notifications and events.
        server_name: Name of the server these resources belong to.

    Returns:
        A list of two LangChain tools: list_resources and read_resource.

    Raises:
        ValueError: If neither session nor connection is provided.

    Example:
        ```python
        from langchain_mcp_adapters.resources import load_mcp_resources_as_tools
        from langchain_mcp_adapters.sessions import create_session

        connection = {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "transport": "stdio",
        }

        async with create_session(connection) as session:
            await session.initialize()
            tools = await load_mcp_resources_as_tools(session)
            # tools can now be used by an agent
        ```
    """  # noqa: E501
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    mcp_callbacks = (
        callbacks.to_mcp_format(context=CallbackContext(server_name=server_name))
        if callbacks is not None
        else []
    )

    async def list_resources_fn(cursor: str | None = None) -> dict[str, Any]:
        """List available MCP resources with pagination support.

        Args:
            cursor: Optional pagination cursor from a previous call.

        Returns:
            A dictionary containing:
                - resources: List of resource dictionaries with uri, name, description, mimeType
                - nextCursor: Pagination cursor for the next page (if available)
        """
        if session is None:
            if connection is None:
                msg = "Either session or connection must be provided"
                raise ValueError(msg)
            async with create_session(
                connection, mcp_callbacks=mcp_callbacks
            ) as resource_session:
                await resource_session.initialize()
                result = await resource_session.list_resources(cursor=cursor)
        else:
            result = await session.list_resources(cursor=cursor)

        resources = [
            {
                "uri": str(r.uri),
                "name": r.name,
                "description": r.description,
                "mimeType": r.mimeType,
            }
            for r in (result.resources or [])
        ]

        return {
            "resources": resources,
            "nextCursor": result.nextCursor,
        }

    async def read_resource_fn(uri: str) -> dict[str, Any]:
        """Read a specific MCP resource by URI.

        Args:
            uri: The URI of the resource to read.

        Returns:
            A dictionary containing:
                - uri: The resource URI
                - contents: List of content dictionaries with type, data, and mimeType
        """
        blobs = await get_mcp_resource(
            session,
            uri
        )

        contents = []
        for blob in blobs:
            content_dict = {
                "mimeType": blob.mimetype,
            }
            # Return text as string, binary as base64
            if isinstance(blob.data, str):
                content_dict["type"] = "text"
                content_dict["data"] = blob.data
            else:
                content_dict["type"] = "blob"
                content_dict["data"] = base64.b64encode(blob.data).decode()

            contents.append(content_dict)

        return {
            "uri": uri,
            "contents": contents,
        }

    list_tool = StructuredTool(
        name="list_resources",
        description=(
            "List available MCP resources. Resources are data sources that can be read. "
            "Returns a list of resources with their URIs, names, descriptions, and MIME types. "
            "Supports pagination via the cursor parameter. "
            "Use this to discover what resources are available before reading them."
        ),
        args_schema=ListResourcesInput,
        coroutine=list_resources_fn,
    )

    read_tool = StructuredTool(
        name="read_resource",
        description=(
            "Read the contents of a specific MCP resource by its URI. "
            "Returns the resource contents which may include text, binary data, or both. "
            "Use list_resources first to discover available resource URIs."
        ),
        args_schema=ReadResourceInput,
        coroutine=read_resource_fn,
    )

    return [list_tool, read_tool]

