"""OAuth 2.1 authentication helpers for MCP servers.

Provides a convenience factory that wires up the MCP SDK's OAuthClientProvider
with file-based token storage and a local callback server for the browser flow.

Usage::

    from langchain_mcp_adapters.auth import oauth_auth

    connections = {
        "notion": {
            "transport": "streamable_http",
            "url": "https://mcp.notion.com/mcp",
            "auth": oauth_auth("https://mcp.notion.com/mcp"),
        },
    }

Or use the shorthand ``auth: "oauth"`` in the connection config — the adapter
will call ``oauth_auth()`` automatically and inject the server URL.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import httpx
from mcp.client.auth.oauth2 import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN_DIR = Path.home() / ".langchain" / "mcp" / "tokens"


class FileTokenStorage:
    """Persist OAuth tokens and client registration to disk.

    Tokens are stored as JSON at ``{token_dir}/{hash}.json`` where *hash* is
    the first 16 hex characters of the SHA-256 of the server URL.

    File permissions are restricted to owner-only (0o600) and the directory
    is created with 0o700.
    """

    def __init__(
        self,
        server_url: str,
        token_dir: Path | str | None = None,
    ) -> None:
        self._dir = Path(token_dir) if token_dir else _DEFAULT_TOKEN_DIR
        url_hash = hashlib.sha256(server_url.encode()).hexdigest()[:16]
        self._path = self._dir / f"{url_hash}.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            self._dir.chmod(0o700)
        except OSError:
            pass

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text())  # noqa: TRY300
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        self._ensure_dir()
        self._path.write_text(json.dumps(data, indent=2))
        try:
            self._path.chmod(0o600)
        except OSError:
            pass

    # --- TokenStorage protocol ---

    async def get_tokens(self) -> OAuthToken | None:
        """Return cached tokens or ``None``."""
        data = self._read()
        raw = data.get("tokens")
        if raw is None:
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.warning("Corrupt token data in %s — ignoring", self._path)
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist tokens to disk."""
        data = self._read()
        data["tokens"] = tokens.model_dump()
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return cached client registration or ``None``."""
        data = self._read()
        raw = data.get("client_info")
        if raw is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Corrupt client info in %s — ignoring", self._path
            )
            return None

    async def set_client_info(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        """Persist client registration to disk."""
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)


def _bind_callback_socket() -> socket.socket:
    """Bind a TCP socket to a random available port on localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock



_CALLBACK_HTML = (
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Connection: close\r\n\r\n"
    "<html><body><h2>Authorization complete</h2>"
    "<p>You may close this tab.</p></body></html>"
)


class _CallbackServer:
    """Local HTTP server that receives the OAuth redirect.

    The server is started eagerly (before the browser opens) to avoid a race
    where a fast OAuth provider redirects before the server is listening.
    """

    def __init__(self, sock: socket.socket, timeout: float) -> None:
        self._sock = sock
        self._timeout = timeout
        self._server: asyncio.AbstractServer | None = None
        self._result_future: asyncio.Future[tuple[str, str | None]] | None = None

    async def start(self) -> None:
        """Start listening for the redirect callback."""
        self._result_future = asyncio.get_running_loop().create_future()
        self._sock.setblocking(False)
        self._server = await asyncio.start_server(
            self._handle_client, sock=self._sock
        )

    async def wait_for_code(self) -> tuple[str, str | None]:
        """Block until the callback arrives or timeout."""
        if self._result_future is None:
            msg = "Server not started. Call start() first."
            raise RuntimeError(msg)
        try:
            return await asyncio.wait_for(
                self._result_future, timeout=self._timeout
            )
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Shut down the server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        try:
            self._sock.close()
        except OSError:
            pass

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=10)
            decoded = request_line.decode("utf-8", errors="replace")

            parts = decoded.split()
            if len(parts) >= 2 and self._result_future is not None:
                parsed = urlparse(parts[1])
                qs = parse_qs(parsed.query)

                error = qs.get("error", [None])[0]
                if error and not self._result_future.done():
                    error_desc = qs.get("error_description", [error])[0]
                    self._result_future.set_exception(
                        RuntimeError(
                            f"OAuth authorization denied: {error_desc}"
                        )
                    )
                    writer.write(
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/html; charset=utf-8\r\n"
                        "Connection: close\r\n\r\n"
                        "<html><body><h2>Authorization failed</h2>"
                        f"<p>{error_desc}</p></body></html>".encode()
                    )
                    await writer.drain()
                    return

                code = qs.get("code", [None])[0]
                state = qs.get("state", [None])[0]

                if code and not self._result_future.done():
                    self._result_future.set_result((code, state))

            writer.write(_CALLBACK_HTML.encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def oauth_auth(
    server_url: str | None = None,
    *,
    token_dir: Path | str | None = None,
    timeout: float = 300.0,
    client_name: str = "langchain-mcp-adapters",
) -> httpx.Auth:
    """Create an OAuth 2.1 auth provider for an MCP server.

    First run opens a browser for authorization.  Subsequent runs reuse the
    cached token from disk, refreshing automatically when it expires.

    Args:
        server_url: The MCP server URL.  If ``None`` a lazy wrapper is
            returned that resolves the URL when the connection is created
            (used internally when ``auth: "oauth"`` is specified in a
            connection dict).
        token_dir: Directory for cached tokens.
            Default: ``~/.langchain/mcp/tokens/``.
        timeout: Seconds to wait for the user to complete browser
            authorization.
        client_name: Client name sent during dynamic client registration.

    Returns:
        An :class:`httpx.Auth` instance that handles the full OAuth 2.1
        flow including PKCE, auto-discovery, dynamic registration, token
        caching, and automatic refresh.
    """
    if server_url is None:
        return _LazyOAuthAuth(
            token_dir=token_dir, timeout=timeout, client_name=client_name
        )

    return _build_oauth_provider(
        server_url,
        token_dir=token_dir,
        timeout=timeout,
        client_name=client_name,
    )


def _build_oauth_provider(
    server_url: str,
    *,
    token_dir: Path | str | None = None,
    timeout: float = 300.0,
    client_name: str = "langchain-mcp-adapters",
) -> OAuthClientProvider:
    """Build a fully-wired OAuthClientProvider."""
    sock = _bind_callback_socket()
    port = sock.getsockname()[1]
    redirect_uri = f"http://127.0.0.1:{port}/callback"

    storage = FileTokenStorage(server_url, token_dir=token_dir)

    # Clear cached client_info if the redirect URI changed (port is random
    # each process, and OAuth servers validate redirect URIs exactly).
    _invalidate_stale_client_info(storage, redirect_uri)

    callback_server = _CallbackServer(sock, timeout)

    client_metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name=client_name,
    )

    async def _redirect_handler(url: str) -> None:
        # Start the callback server BEFORE opening the browser so fast
        # redirects (auto-approved SSO) don't get connection-refused.
        await callback_server.start()
        logger.info("Opening browser for OAuth authorization...")
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        print(  # noqa: T201
            f"\n  OAuth authorization required.\n"
            f"  If a browser did not open, visit this URL:\n\n"
            f"    {url}\n"
        )

    async def _callback_handler() -> tuple[str, str | None]:
        return await callback_server.wait_for_code()

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
        timeout=timeout,
    )


def _invalidate_stale_client_info(
    storage: FileTokenStorage, redirect_uri: str
) -> None:
    """Clear cached client_info if the stored redirect URI doesn't match.

    Each process binds a random port, so the redirect_uri changes across
    restarts.  OAuth servers that validate redirect URIs exactly will reject
    re-auth attempts with a stale client registration.
    """
    data = storage._read()
    client_info = data.get("client_info")
    if client_info is None:
        return

    stored_uris = client_info.get("redirect_uris", [])
    # Pydantic serializes AnyUrl objects; compare as strings
    stored_str = [str(u) for u in stored_uris]
    if redirect_uri not in stored_str:
        logger.debug(
            "Redirect URI changed (%s not in %s), clearing cached client_info",
            redirect_uri,
            stored_str,
        )
        data.pop("client_info", None)
        storage._write(data)


class _LazyOAuthAuth(httpx.Auth):
    """Deferred ``OAuthClientProvider`` that resolves once a server URL is set.

    Used when ``auth: "oauth"`` is specified in a connection config — the
    adapter calls :meth:`resolve` with the connection URL before the first
    HTTP request.
    """

    requires_response_body = True

    def __init__(
        self,
        *,
        token_dir: Path | str | None = None,
        timeout: float = 300.0,
        client_name: str = "langchain-mcp-adapters",
    ) -> None:
        self._token_dir = token_dir
        self._timeout = timeout
        self._client_name = client_name
        self._inner: OAuthClientProvider | None = None

    def resolve(self, server_url: str) -> None:
        """Bind this lazy auth to a concrete server URL."""
        if self._inner is None:
            self._inner = _build_oauth_provider(
                server_url,
                token_dir=self._token_dir,
                timeout=self._timeout,
                client_name=self._client_name,
            )

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> asyncio.AsyncGenerator[httpx.Request, httpx.Response]:
        """Delegate to the inner provider's auth flow."""
        if self._inner is None:
            msg = (
                "LazyOAuthAuth was not resolved before use. "
                "Call .resolve(server_url) or use oauth_auth(server_url) directly."
            )
            raise RuntimeError(msg)
        async for item in self._inner.async_auth_flow(request):
            yield item
