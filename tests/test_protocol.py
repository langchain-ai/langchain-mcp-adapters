"""Tests for MCP protocol-revision negotiation.

The distinction under test is between the two negotiation eras:

- Handshake era (up to 2025-11-25), reached via `initialize`.
- 2026-07-28 and later, reached via a `server/discover` probe. `initialize`
  cannot reach it, so a client that only ever calls `initialize` stays on a
  handshake revision no matter what the server supports.
"""

import sys

import pytest
from mcp import ClientSession

from langchain_mcp_adapters.callbacks import DEFAULT_LOG_LEVEL, Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    DEFAULT_PROTOCOL,
    create_session,
    negotiate_protocol,
    resolve_protocol,
)
from tests.servers.protocol_server import create_protocol_server
from tests.utils import run_streamable_http

MODERN = "2026-07-28"
PORT = 8250


def _http(port: int, **extra) -> dict:
    return {"url": f"http://localhost:{port}/mcp", "transport": "http", **extra}


def _stdio(**extra) -> dict:
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [
            "-c",
            "from tests.servers.protocol_server import create_protocol_server; "
            "create_protocol_server().run(transport='stdio')",
        ],
        **extra,
    }


# --- resolve_protocol -------------------------------------------------------


def test_default_protocol_is_auto() -> None:
    assert DEFAULT_PROTOCOL == "auto"
    assert resolve_protocol(None) == "auto"
    assert resolve_protocol({"transport": "stdio"}) == "auto"


def test_connection_protocol_wins_over_default() -> None:
    assert resolve_protocol({"protocol": "legacy"}, default=MODERN) == "legacy"
    assert resolve_protocol({}, default="legacy") == "legacy"


@pytest.mark.parametrize("bad", ["v2", "2025-06-18", "", "AUTO"])
def test_unknown_protocol_is_rejected(bad) -> None:
    """A typo must not silently leave the connection on a handshake revision."""
    with pytest.raises(ValueError, match="Unsupported protocol"):
        resolve_protocol({"protocol": bad})
    with pytest.raises(ValueError, match="Unsupported protocol"):
        MultiServerMCPClient({}, protocol=bad)


def test_handshake_era_version_is_rejected_as_a_pin() -> None:
    """Handshake revisions are reached with 'legacy', not by pinning them."""
    with pytest.raises(ValueError, match="Unsupported protocol"):
        resolve_protocol({"protocol": "2025-11-25"})


# --- what actually gets negotiated ------------------------------------------


async def test_auto_reaches_modern_over_http(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT):
        client = MultiServerMCPClient({"s": _http(PORT)})
        info = await client.get_server_info()

    assert info["s"].protocol_version == MODERN
    assert info["s"].is_modern


async def test_auto_reaches_modern_over_stdio() -> None:
    client = MultiServerMCPClient({"s": _stdio()})
    info = await client.get_server_info()

    assert info["s"].protocol_version == MODERN
    assert info["s"].is_modern


async def test_legacy_stays_on_the_handshake(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 1):
        client = MultiServerMCPClient({"s": _http(PORT + 1, protocol="legacy")})
        info = await client.get_server_info()

    assert info["s"].protocol_version == "2025-11-25"
    assert not info["s"].is_modern
    # The handshake result carries the server identity unconditionally.
    assert info["s"].server_info is not None
    assert info["s"].server_info.name == "protocol-server"


async def test_pinned_modern_version_skips_the_probe(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 2):
        client = MultiServerMCPClient({"s": _http(PORT + 2, protocol=MODERN)})
        info = await client.get_server_info()
        tools = await client.get_tools()

    assert info["s"].protocol_version == MODERN
    assert {t.name for t in tools} == {"echo", "noisy"}


async def test_client_wide_default_applies_to_every_server(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 3):
        client = MultiServerMCPClient(
            {"a": _http(PORT + 3), "b": _http(PORT + 3)}, protocol="legacy"
        )
        info = await client.get_server_info()

    assert [i.protocol_version for i in info.values()] == ["2025-11-25"] * 2


async def test_connection_overrides_client_wide_default(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 4):
        client = MultiServerMCPClient(
            {"pinned": _http(PORT + 4, protocol=MODERN), "inherited": _http(PORT + 4)},
            protocol="legacy",
        )
        info = await client.get_server_info()

    assert info["pinned"].protocol_version == MODERN
    assert info["inherited"].protocol_version == "2025-11-25"


async def test_session_protocol_argument_overrides_everything(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 5):
        client = MultiServerMCPClient(
            {"s": _http(PORT + 5, protocol="legacy")}, protocol="legacy"
        )
        async with client.session("s", protocol=MODERN) as session:
            assert session.protocol_version == MODERN


async def test_tools_work_on_both_eras(socket_enabled) -> None:
    with run_streamable_http(create_protocol_server, PORT + 6):
        for protocol, expected in (("auto", MODERN), ("legacy", "2025-11-25")):
            client = MultiServerMCPClient({"s": _http(PORT + 6, protocol=protocol)})
            tools = await client.get_tools()
            echo = next(t for t in tools if t.name == "echo")
            result = await echo.ainvoke(
                {"args": {"text": "hi"}, "id": "1", "type": "tool_call"}
            )
            assert "hi" in str(result.content)
            assert (await client.get_server_info())["s"].protocol_version == expected


# --- the protocol key is negotiation policy, not transport config -----------


async def test_protocol_key_does_not_reach_the_transport(socket_enabled) -> None:
    """`create_session` must strip `protocol` before splatting into a transport."""
    with run_streamable_http(create_protocol_server, PORT + 7):
        async with create_session(_http(PORT + 7, protocol=MODERN)) as session:
            # Reaching here at all means no unexpected-kwarg TypeError.
            assert isinstance(session, ClientSession)
            await negotiate_protocol(session, protocol=MODERN)
            assert session.protocol_version == MODERN


# --- SEP-2577 logging opt-in ------------------------------------------------


async def test_logging_callback_fires_on_modern_protocol(socket_enabled) -> None:
    """A registered logging callback must survive the move to 2026-07-28.

    Modern servers emit `notifications/message` only for requests that opt in
    via `_meta`. Without that opt-in a callback that worked on a handshake
    connection goes silent with no error.
    """
    messages: list[str] = []

    async def on_logging_message(params, context) -> None:
        messages.append(str(params.data))

    with run_streamable_http(create_protocol_server, PORT + 8):
        client = MultiServerMCPClient(
            {"s": _http(PORT + 8)},
            callbacks=Callbacks(on_logging_message=on_logging_message),
        )
        tools = await client.get_tools()
        noisy = next(t for t in tools if t.name == "noisy")
        await noisy.ainvoke({"args": {"text": "hi"}, "id": "1", "type": "tool_call"})
        assert (await client.get_server_info())["s"].protocol_version == MODERN

    assert any("handling hi" in m for m in messages)


async def test_default_log_level_is_the_permissive_one() -> None:
    """Matches handshake-era behavior, where an unset level means "send it all"."""
    assert DEFAULT_LOG_LEVEL == "debug"
    assert Callbacks().log_level == "debug"


async def test_progress_callback_fires_on_modern_protocol(socket_enabled) -> None:
    """Server-to-client progress survives; only the client-to-server leg is gone."""
    updates: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress, total, message, context) -> None:
        updates.append((progress, total, message))

    with run_streamable_http(create_protocol_server, PORT + 9):
        client = MultiServerMCPClient(
            {"s": _http(PORT + 9)}, callbacks=Callbacks(on_progress=on_progress)
        )
        tools = await client.get_tools()
        noisy = next(t for t in tools if t.name == "noisy")
        await noisy.ainvoke({"args": {"text": "hi"}, "id": "1", "type": "tool_call"})

    assert updates == [(1.0, 1.0, "done")]
