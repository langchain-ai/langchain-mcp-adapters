from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@mcp.prompt()
def configure_assistant(skills: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": (
                f"You are a helpful assistant. You have these skills: {skills}. "
                "Always use only one tool at a time."
            ),
        },
    ]


@mcp.resource("math://randomformulas/{name}")
def get_random_formulas(name: str) -> str:
    """Get random formulas."""
    formulas = {
        "energy": "E = mc^2",
        "area": "A = πr^2",
        "volume": "V = lwh",
    }
    return formulas.get(name, "Unknown formula")


if __name__ == "__main__":
    mcp.run(transport="stdio")
