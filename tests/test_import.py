import asyncio


def test_import() -> None:
    """Test that the code can be imported"""
    from langchain_mcp_adapters import (  # noqa: F401, PLC0415
        callbacks,
        client,
        interceptors,
        prompts,
        resources,
        server_info,
        sessions,
        tools,
    )


async def test_asyncio_event_loop_starts_under_socket_guard() -> None:
    """Windows ProactorEventLoop must start when sockets are restricted.

    ``make test`` passes ``--disable-socket --allow-unix-socket``. Linux
    asyncio uses an AF_UNIX self-pipe (allowed). Windows uses an AF_INET
    ``socketpair()`` to 127.0.0.1, which that combination used to block so
    every async test failed during loop setup.
    """
    await asyncio.sleep(0)
