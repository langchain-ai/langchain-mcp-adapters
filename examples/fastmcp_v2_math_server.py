"""
Example FastMCP 2.0 SDK Math Server

This is an example server using the independent FastMCP 2.0 SDK
(https://github.com/jlowin/fastmcp) that provides mathematical operations.

This demonstrates how to create a FastMCP 2.0 server that can be integrated
with LangGraph using the new FastMCP 2.0 adapters.
"""

try:
    # FastMCP 2.0 SDK imports
    from fastmcp import FastMCP
    FASTMCP_V2_AVAILABLE = True
except ImportError:
    FASTMCP_V2_AVAILABLE = False
    print("FastMCP 2.0 SDK not available. Please install with: pip install fastmcp")
    exit(1)

# Create FastMCP 2.0 server
mcp = FastMCP("FastMCP 2.0 Math Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract second number from first number.
    
    Args:
        a: First number (minuend)
        b: Second number (subtrahend)
        
    Returns:
        Difference of a and b
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide first number by second number.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Quotient of a divided by b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent.
    
    Args:
        base: Base number
        exponent: Exponent
        
    Returns:
        base raised to the power of exponent
    """
    return base ** exponent


@mcp.tool()
def square_root(x: float) -> float:
    """Calculate square root of a number.
    
    Args:
        x: Number to find square root of
        
    Returns:
        Square root of x
        
    Raises:
        ValueError: If x is negative
    """
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return x ** 0.5


@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@mcp.tool()
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence (0-indexed)
        
    Returns:
        nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative indices")
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


@mcp.tool()
def is_prime(n: int) -> bool:
    """Check if a number is prime.
    
    Args:
        n: Number to check
        
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


@mcp.tool()
def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Greatest common divisor of a and b
    """
    while b:
        a, b = b, a % b
    return abs(a)


@mcp.tool()
def lcm(a: int, b: int) -> int:
    """Calculate least common multiple of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Least common multiple of a and b
    """
    return abs(a * b) // gcd(a, b) if a and b else 0


@mcp.tool()
async def advanced_calculation(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the expression as a string
        
    Raises:
        ValueError: If expression is invalid or unsafe
    """
    # Simple expression evaluator (for demo purposes)
    # In production, you'd want a more robust and secure evaluator
    import re
    
    # Only allow numbers, basic operators, and parentheses
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        raise ValueError("Expression contains invalid characters")
    
    try:
        # Use eval with restricted globals for demo (not recommended for production)
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


# Additional FastMCP 2.0 features demonstration
@mcp.resource("math://constants")
def get_math_constants():
    """Get common mathematical constants."""
    import math
    return {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "golden_ratio": (1 + 5**0.5) / 2
    }


@mcp.prompt()
def math_tutor_prompt(topic: str) -> str:
    """Generate a math tutoring prompt for a given topic.
    
    Args:
        topic: Mathematical topic to create a prompt for
        
    Returns:
        A tutoring prompt for the topic
    """
    return f"""You are a helpful math tutor. Please explain the concept of {topic} in simple terms, 
provide examples, and suggest practice problems. Make sure to:

1. Start with a clear definition
2. Provide step-by-step examples
3. Suggest 2-3 practice problems
4. Offer tips for remembering the concept

Topic: {topic}
"""


if __name__ == "__main__":
    # Run the FastMCP 2.0 server
    print("Starting FastMCP 2.0 Math Server...")
    print("Available tools:", [tool.name for tool in mcp._tools.values()])
    
    # Run with stdio transport (default)
    mcp.run(transport="stdio")