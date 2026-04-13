"""Tests for the OAuth auth module."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from mcp.client.auth.oauth2 import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from langchain_mcp_adapters.auth import (
    FileTokenStorage,
    _CallbackServer,
    _LazyOAuthAuth,
    _bind_callback_socket,
    oauth_auth,
)


# --- FileTokenStorage ---


@pytest.fixture
def token_dir(tmp_path: Path) -> Path:
    return tmp_path / "tokens"


@pytest.fixture
def storage(token_dir: Path) -> FileTokenStorage:
    return FileTokenStorage("https://mcp.example.com/mcp", token_dir=token_dir)


@pytest.fixture
def sample_token() -> OAuthToken:
    return OAuthToken(
        access_token="test-access-token",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="test-refresh-token",
        scope="read write",
    )


@pytest.fixture
def sample_client_info() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uris=["http://127.0.0.1:12345/callback"],
        token_endpoint_auth_method="none",
    )


async def test_storage_roundtrip_tokens(
    storage: FileTokenStorage, sample_token: OAuthToken
):
    assert await storage.get_tokens() is None
    await storage.set_tokens(sample_token)
    result = await storage.get_tokens()
    assert result is not None
    assert result.access_token == "test-access-token"
    assert result.refresh_token == "test-refresh-token"


async def test_storage_roundtrip_client_info(
    storage: FileTokenStorage, sample_client_info: OAuthClientInformationFull
):
    assert await storage.get_client_info() is None
    await storage.set_client_info(sample_client_info)
    result = await storage.get_client_info()
    assert result is not None
    assert result.client_id == "test-client-id"


async def test_storage_handles_missing_file(token_dir: Path):
    storage = FileTokenStorage("https://mcp.example.com/mcp", token_dir=token_dir)
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None


async def test_storage_handles_corrupt_file(token_dir: Path):
    storage = FileTokenStorage("https://mcp.example.com/mcp", token_dir=token_dir)
    # Write some tokens first to create the directory
    await storage.set_tokens(
        OAuthToken(access_token="x", token_type="Bearer")
    )
    # Corrupt the file
    storage._path.write_text("not valid json{{{")
    assert await storage.get_tokens() is None


async def test_storage_handles_corrupt_token_data(token_dir: Path):
    storage = FileTokenStorage("https://mcp.example.com/mcp", token_dir=token_dir)
    # Write valid JSON but invalid token structure
    storage._ensure_dir()
    storage._path.write_text(json.dumps({"tokens": {"bad": "data"}}))
    assert await storage.get_tokens() is None


async def test_storage_separate_servers_separate_files(token_dir: Path):
    s1 = FileTokenStorage("https://server1.com/mcp", token_dir=token_dir)
    s2 = FileTokenStorage("https://server2.com/mcp", token_dir=token_dir)
    assert s1._path != s2._path


async def test_storage_file_permissions(
    storage: FileTokenStorage, sample_token: OAuthToken
):
    await storage.set_tokens(sample_token)
    # Check file exists and has restricted permissions
    assert storage._path.exists()
    mode = storage._path.stat().st_mode & 0o777
    assert mode == 0o600


# --- Callback server ---


def test_bind_callback_socket():
    sock = _bind_callback_socket()
    try:
        addr = sock.getsockname()
        assert addr[0] == "127.0.0.1"
        assert addr[1] > 0
    finally:
        sock.close()


async def test_callback_server_receives_code():
    sock = _bind_callback_socket()
    port = sock.getsockname()[1]
    server = _CallbackServer(sock, timeout=5.0)
    await server.start()

    async def _send_callback():
        await asyncio.sleep(0.1)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            f"GET /callback?code=test-code&state=test-state HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n\r\n".encode()
        )
        await writer.drain()
        await reader.read(4096)
        writer.close()
        await writer.wait_closed()

    results = await asyncio.gather(server.wait_for_code(), _send_callback())
    assert results[0] == ("test-code", "test-state")


async def test_callback_server_timeout():
    sock = _bind_callback_socket()
    server = _CallbackServer(sock, timeout=0.1)
    await server.start()
    with pytest.raises(asyncio.TimeoutError):
        await server.wait_for_code()


# --- oauth_auth factory ---


def test_oauth_auth_with_url_returns_provider():
    auth = oauth_auth("https://mcp.example.com/mcp")
    assert isinstance(auth, OAuthClientProvider)


def test_oauth_auth_without_url_returns_lazy():
    auth = oauth_auth()
    assert isinstance(auth, _LazyOAuthAuth)


def test_lazy_auth_resolve():
    auth = oauth_auth()
    assert isinstance(auth, _LazyOAuthAuth)
    auth.resolve("https://mcp.example.com/mcp")
    assert auth._inner is not None
    assert isinstance(auth._inner, OAuthClientProvider)


async def test_lazy_auth_raises_if_not_resolved():
    auth = _LazyOAuthAuth()
    request = httpx.Request("GET", "https://example.com")
    with pytest.raises(RuntimeError, match="not resolved"):
        async for _ in auth.async_auth_flow(request):
            pass


def test_oauth_auth_custom_token_dir(tmp_path: Path):
    auth = oauth_auth("https://mcp.example.com/mcp", token_dir=tmp_path / "custom")
    assert isinstance(auth, OAuthClientProvider)
    # Verify the storage was configured with our custom dir
    storage = auth.context.storage
    assert isinstance(storage, FileTokenStorage)
    assert storage._dir == tmp_path / "custom"
