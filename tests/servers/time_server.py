import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time")


@mcp.tool()
def get_time() -> str:
    """Get current time"""
    return str(datetime.datetime.now().time())


if __name__ == "__main__":
    mcp.run()
