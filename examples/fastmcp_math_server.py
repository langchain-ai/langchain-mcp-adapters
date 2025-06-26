"""
Example FastMCP Math Server

This is an example FastMCP server that provides mathematical operations.
It demonstrates how to create a FastMCP server that can be integrated with LangGraph.
"""

# This example works with the official MCP Python SDK's FastMCP
from mcp.server.fastmcp import FastMCP

# Create FastMCP server
mcp = FastMCP("FastMCP Math Server")


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


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")