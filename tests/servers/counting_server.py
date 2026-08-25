"""Elicit server that counts `tools/call` rounds, for round-trip assertions.

The count lives in the resolver rather than the tool body: the body only runs
once the inputs resolve, but every `tools/call` round re-runs the resolver. So
`rounds_seen` is exactly the number of round-trips `create_profile` cost.
"""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.resolve import Elicit, Resolve
from pydantic import BaseModel

_ROUNDS: list[str] = []


class UserDetails(BaseModel):
    email: str
    age: int


def ask_details(name: str) -> Elicit[UserDetails]:
    _ROUNDS.append(name)
    return Elicit(f"Please provide details for {name}'s profile:", UserDetails)


def create_counting_server() -> MCPServer:
    server = MCPServer("counting-server")

    @server.tool()
    async def create_profile(
        name: str,
        details: Annotated[UserDetails, Resolve(ask_details)],
    ) -> str:
        """Create a user profile, eliciting the details from the client."""
        return f"Created profile for {name}: email={details.email}, age={details.age}"

    @server.tool()
    async def rounds_seen() -> int:
        """How many create_profile round-trips the server has handled."""
        return len(_ROUNDS)

    return server
