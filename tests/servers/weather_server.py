from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return f"It's always sunny in {location}"


@mcp.prompt()
def weather_briefing_style(tone: str) -> list[dict]:
    """How to phrase weather answers for end users."""
    return [
        {
            "role": "assistant",
            "content": (
                f"You report weather in a {tone} tone. "
                "Keep answers short unless the user asks for detail."
            ),
        },
    ]


@mcp.resource("weather://forecast")
def get_weather_forecast() -> str:
    """Get weather forecast."""
    return "Sunny with a chance of clouds"


if __name__ == "__main__":
    mcp.run(transport="stdio")
