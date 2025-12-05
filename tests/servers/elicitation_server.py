"""Simple MCP server with elicitation support for testing."""

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel

mcp = FastMCP("Elicitation Test Server")

# Track how many times code before elicit runs (should be exactly once per call)
_pre_elicit_call_count = 0


class UserDetails(BaseModel):
    email: str
    age: int


@mcp.tool()
async def create_profile(name: str, ctx: Context) -> str:
    """Create a user profile. Asks for additional details via elicitation."""
    global _pre_elicit_call_count
    # This code should only run once, not be re-executed after elicitation
    _pre_elicit_call_count += 1

    result = await ctx.elicit(
        message=f"Please provide details for {name}'s profile:",
        schema=UserDetails,
    )

    if result.action == "accept" and result.data:
        return (
            f"Created profile for {name}: "
            f"email={result.data.email}, age={result.data.age}, "
            f"pre_elicit_calls={_pre_elicit_call_count}"
        )
    if result.action == "decline":
        return f"User declined. Created minimal profile for {name}. pre_elicit_calls={_pre_elicit_call_count}"
    return f"Profile creation cancelled. pre_elicit_calls={_pre_elicit_call_count}"


if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "streamable-http"
    mcp.run(transport=transport)
