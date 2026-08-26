"""Tests for MCP protocol-revision negotiation.

MCP has two negotiation eras. Everything up to 2025-11-25 is reached through
the `initialize` handshake; 2026-07-28 is reached through a `server/discover`
probe instead. A client that only ever calls `initialize` therefore stays on a
handshake revision no matter what the server supports, which is what these
tests pin down.
"""

import sys

import pytest

from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import create_session, negotiate_protocol
from tests.servers.protocol_server import create_protocol_server
from tests.utils import run_streamable_http

MODERN = "2026-07-28"
HANDSHAKE = "2025-11-25"
PORT = 8250


def _http(port: int, **extra) -> dict:
    return {"url": f"http://localhost:{port}/mcp", "transport": "http", **extra}


async def test_auto_reaches_modern_over_http(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT):
        client = MultiServerMCPClient({"s": _http(PORT)})
        info = await client.get_server_info()

    assert info["s"].protocol_version == MODERN


async def test_auto_reaches_modern_over_stdio() -> None:
    client = MultiServerMCPClient(
        {
            "s": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    "-c",
                    "from tests.servers.protocol_server import "
                    "create_protocol_server; "
                    "create_protocol_server().run(transport='stdio')",
                ],
            }
        }
    )
    info = await client.get_server_info()

    assert info["s"].protocol_version == MODERN


async def test_legacy_stays_on_the_handshake(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 1):
        client = MultiServerMCPClient({"s": _http(PORT + 1, protocol="legacy")})
        info = await client.get_server_info()

    assert info["s"].protocol_version == HANDSHAKE
    assert info["s"].server_info.name == "protocol-server"


async def test_connection_overrides_the_client_default(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 2):
        client = MultiServerMCPClient(
            {
                "pinned": _http(PORT + 2, protocol="legacy"),
                "inherited": _http(PORT + 2),
            },
            protocol="auto",
        )
        info = await client.get_server_info()

    assert info["pinned"].protocol_version == HANDSHAKE
    assert info["inherited"].protocol_version == MODERN


async def test_tools_work_on_both_eras(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 3):
        for protocol, expected in (("auto", MODERN), ("legacy", HANDSHAKE)):
            client = MultiServerMCPClient({"s": _http(PORT + 3, protocol=protocol)})
            echo = next(t for t in await client.get_tools() if t.name == "echo")
            result = await echo.ainvoke(
                {"args": {"text": "hi"}, "id": "1", "type": "tool_call"}
            )

            assert "hi" in str(result.content)
            assert (await client.get_server_info())["s"].protocol_version == expected


async def test_unknown_protocol_is_rejected(socket_enabled) -> None:
    """A typo must not silently leave the connection on a handshake revision."""
    with run_streamable_http(create_protocol_server, PORT + 4):
        async with create_session(_http(PORT + 4)) as session:
            with pytest.raises(ValueError, match="Unsupported protocol"):
                await negotiate_protocol(session, "v2")


async def test_protocol_key_does_not_reach_the_transport(socket_enabled) -> None:
    """`create_session` must strip `protocol` before splatting into a transport."""
    with run_streamable_http(create_protocol_server, PORT + 5):
        # Reaching the negotiation at all means no unexpected-kwarg TypeError.
        async with create_session(_http(PORT + 5, protocol="auto")) as session:
            await negotiate_protocol(session)
            assert session.protocol_version == MODERN


async def test_logging_callback_survives_the_move_to_modern(socket_enabled) -> None:
    """SEP-2577: modern servers only log for requests that opt in.

    Without the opt-in a callback that worked on a handshake connection goes
    silent, with no error to explain it.
    """
    messages: list[str] = []

    async def on_logging_message(params, context) -> None:
        messages.append(str(params.data))

    with run_streamable_http(create_protocol_server, PORT + 6):
        client = MultiServerMCPClient(
            {"s": _http(PORT + 6)},
            callbacks=Callbacks(on_logging_message=on_logging_message),
        )
        noisy = next(t for t in await client.get_tools() if t.name == "noisy")
        await noisy.ainvoke({"args": {"text": "hi"}, "id": "1", "type": "tool_call"})

        assert (await client.get_server_info())["s"].protocol_version == MODERN

    assert any("handling hi" in m for m in messages)


async def test_progress_callback_survives_the_move_to_modern(socket_enabled) -> None:
    """Server-to-client progress is unaffected; only the reverse leg is gone."""
    updates: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress, total, message, context) -> None:
        updates.append((progress, total, message))

    with run_streamable_http(create_protocol_server, PORT + 7):
        client = MultiServerMCPClient(
            {"s": _http(PORT + 7)}, callbacks=Callbacks(on_progress=on_progress)
        )
        noisy = next(t for t in await client.get_tools() if t.name == "noisy")
        await noisy.ainvoke({"args": {"text": "hi"}, "id": "1", "type": "tool_call"})

    assert updates == [(1.0, 1.0, "done")]
