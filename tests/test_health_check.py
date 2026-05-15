import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
from anyio import BrokenResourceError, ClosedResourceError

from langchain_mcp_adapters.client import LongLivedMultiServerMCPClient


def _make_client(**kwargs: object) -> LongLivedMultiServerMCPClient:
    return LongLivedMultiServerMCPClient(connections={}, **kwargs)  # type: ignore[arg-type]


class TestHealthCheck:
    """Tests for LongLivedMultiServerMCPClient.health_check() — raw ping API."""

    async def test_healthy_sessions_return_true(self):
        client = _make_client()
        s1, s2 = MagicMock(), MagicMock()
        s1.send_ping = AsyncMock(return_value=None)
        s2.send_ping = AsyncMock(return_value=None)
        client.sessions = {"server_a": s1, "server_b": s2}

        result = await client.health_check()
        assert result == {"server_a": True, "server_b": True}

    async def test_dead_session_returns_false(self):
        client = _make_client()
        healthy = MagicMock()
        healthy.send_ping = AsyncMock(return_value=None)
        dead = MagicMock()
        dead.send_ping = AsyncMock(side_effect=ClosedResourceError())
        client.sessions = {"alive": healthy, "dead_server": dead}

        result = await client.health_check()
        assert result == {"alive": True, "dead_server": False}

    async def test_empty_sessions_returns_empty_dict(self):
        client = _make_client()
        client.sessions = {}
        result = await client.health_check()
        assert result == {}

    async def test_timeout_marks_slow_sessions_unhealthy(self):
        client = _make_client()
        slow = MagicMock()

        async def _slow_ping() -> None:
            await asyncio.sleep(10)

        slow.send_ping = _slow_ping
        client.sessions = {"slow_server": slow}

        result = await client.health_check(timeout=0.1)
        assert result == {"slow_server": False}

    async def test_broken_resource_error_returns_false(self):
        client = _make_client()
        broken = MagicMock()
        broken.send_ping = AsyncMock(side_effect=BrokenResourceError())
        client.sessions = {"broken": broken}

        result = await client.health_check()
        assert result == {"broken": False}

    async def test_httpx_transport_error_returns_false(self):
        client = _make_client()
        http_dead = MagicMock()
        http_dead.send_ping = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client.sessions = {"http_dead": http_dead}

        result = await client.health_check()
        assert result == {"http_dead": False}

    async def test_uses_instance_timeout_by_default(self):
        client = _make_client(health_check_timeout=1.0)
        fast = MagicMock()
        fast.send_ping = AsyncMock(return_value=None)
        client.sessions = {"fast": fast}

        result = await client.health_check()
        assert result == {"fast": True}

    async def test_timeout_arg_overrides_instance_default(self):
        client = _make_client(health_check_timeout=60.0)
        slow = MagicMock()

        async def _slow_ping() -> None:
            await asyncio.sleep(10)

        slow.send_ping = _slow_ping
        client.sessions = {"slow": slow}

        result = await client.health_check(timeout=0.1)
        assert result == {"slow": False}


class TestSessionsHealthy:
    """Tests for LongLivedMultiServerMCPClient.sessions_healthy() — debounced API."""

    async def test_healthy_returns_true(self):
        client = _make_client()
        s1 = MagicMock()
        s1.send_ping = AsyncMock(return_value=None)
        client.sessions = {"server_a": s1}

        assert await client.sessions_healthy() is True

    async def test_dead_session_returns_false(self):
        client = _make_client()
        dead = MagicMock()
        dead.send_ping = AsyncMock(side_effect=ClosedResourceError())
        client.sessions = {"dead": dead}

        assert await client.sessions_healthy() is False

    async def test_empty_sessions_returns_true(self):
        client = _make_client()
        client.sessions = {}
        assert await client.sessions_healthy() is True

    async def test_debounce_skips_ping_within_interval(self):
        client = _make_client(health_check_interval=30.0)
        s1 = MagicMock()
        s1.send_ping = AsyncMock(return_value=None)
        client.sessions = {"server_a": s1}

        assert await client.sessions_healthy() is True
        assert s1.send_ping.call_count == 1

        # Second call — within 30s interval, should skip ping
        assert await client.sessions_healthy() is True
        assert s1.send_ping.call_count == 1

    async def test_debounce_expired_pings_again(self):
        client = _make_client(health_check_interval=30.0)
        s1 = MagicMock()
        s1.send_ping = AsyncMock(return_value=None)
        client.sessions = {"server_a": s1}

        assert await client.sessions_healthy() is True
        assert s1.send_ping.call_count == 1

        # Force debounce to expire
        client._last_healthy_ping = 0.0

        assert await client.sessions_healthy() is True
        assert s1.send_ping.call_count == 2

    async def test_no_interval_pings_every_call(self):
        client = _make_client(health_check_interval=None)
        s1 = MagicMock()
        s1.send_ping = AsyncMock(return_value=None)
        client.sessions = {"server_a": s1}

        await client.sessions_healthy()
        await client.sessions_healthy()
        await client.sessions_healthy()
        assert s1.send_ping.call_count == 3

    async def test_unhealthy_does_not_update_last_ping(self):
        client = _make_client(health_check_interval=30.0)
        dead = MagicMock()
        dead.send_ping = AsyncMock(side_effect=ClosedResourceError())
        client.sessions = {"dead": dead}

        assert await client.sessions_healthy() is False
        assert client._last_healthy_ping == 0.0

    async def test_healthy_updates_last_ping(self):
        client = _make_client(health_check_interval=30.0)
        s1 = MagicMock()
        s1.send_ping = AsyncMock(return_value=None)
        client.sessions = {"server_a": s1}

        assert client._last_healthy_ping == 0.0
        assert await client.sessions_healthy() is True
        assert client._last_healthy_ping > 0.0


class TestStartResetsLastHealthyPing:
    """Verify that start() resets _last_healthy_ping after opening sessions."""

    async def test_start_resets_last_healthy_ping(self, monkeypatch):
        client = LongLivedMultiServerMCPClient(
            {
                "math": {
                    "command": "python3",
                    "args": ["unused"],
                    "transport": "stdio",
                }
            }
        )
        assert client._last_healthy_ping == 0.0

        fake_session = AsyncMock()

        @asynccontextmanager
        async def fake_session_cm(server_name, *, auto_initialize=True):
            yield fake_session

        monkeypatch.setattr(client, "session", fake_session_cm)

        await client.start()
        assert client._last_healthy_ping > 0.0
        await client.stop()
