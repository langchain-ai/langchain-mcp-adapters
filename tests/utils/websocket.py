import time

import uvicorn
from mcp.server.websocket import websocket_server
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

from tests.servers.time_server import mcp as time_mcp


def make_server_app() -> Starlette:
    server = time_mcp._mcp_server

    async def handle_ws(websocket):
        async with websocket_server(websocket.scope, websocket.receive, websocket.send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    app = Starlette(
        routes=[
            WebSocketRoute("/ws", endpoint=handle_ws),
        ]
    )

    return app


def run_server(server_port: int) -> None:
    app = make_server_app()
    server = uvicorn.Server(
        config=uvicorn.Config(app=app, host="127.0.0.1", port=server_port, log_level="error")
    )
    server.run()

    # Give server time to start
    while not server.started:
        time.sleep(0.5)
