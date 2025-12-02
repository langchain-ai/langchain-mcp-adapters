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


@mcp.prompt()
def math_problem_solver(problem_type: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": (
                f"You are a math expert specializing in {problem_type}. "
                "Provide step-by-step solutions to problems."
            ),
        },
    ]


@mcp.prompt()
def calculation_guide() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": (
                "You are a calculation guide. "
                "Help users understand how to perform "
                "mathematical operations correctly."
            ),
        },
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
