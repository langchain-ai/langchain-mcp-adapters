"""Example demonstrating MCP tool call interceptors and callbacks.

This example shows how to use interceptors to:
1. Modify tool arguments before execution
2. Transform tool results after execution
3. Add authentication headers
4. Log tool calls and progress
5. Handle different result formats
"""

import asyncio
import time
from typing import Any

from langchain_mcp_adapters import (
    MultiServerMCPClient,
    InterceptorConfig,
    BeforeToolCallHook,
    AfterToolCallHook,
    OnMessageCallback,
    OnProgressCallback,
)
from langchain_mcp_adapters.interceptors import (
    ToolCallArgs,
    ToolCallResult,
    MessageEvent,
    ProgressEvent,
)
from mcp.types import CallToolResult


# Example 1: Basic authentication and logging interceptor
def auth_before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
    """Add authentication and user context to tool calls."""
    print(f"🔐 Adding auth to tool call: {tool_call['name']}")

    # Add user context
    tool_call["arguments"]["user_id"] = "user_123"
    tool_call["arguments"]["session_id"] = "session_abc"

    # Add authentication headers (Note: header support would need MCP SDK updates)
    tool_call["headers"] = {
        "Authorization": "Bearer your-api-token",
        "X-Request-ID": f"req_{int(time.time())}",
    }

    return tool_call


def logging_after_hook(tool_call: ToolCallArgs, result: CallToolResult) -> ToolCallResult:
    """Log and optionally transform tool results."""
    print(f"📝 Tool {tool_call['name']} completed with {len(result.content)} content items")

    # Extract text content
    text_content = []
    for content in result.content:
        if hasattr(content, 'text'):
            text_content.append(content.text)

    # Return enhanced result with metadata
    return {
        "content": f"[Logged] {' '.join(text_content)}",
        "artifacts": [
            {
                "tool_name": tool_call["name"],
                "execution_time": time.time(),
                "user_id": tool_call["arguments"].get("user_id"),
            }
        ],
    }


# Example 2: Async interceptors with validation
async def async_validation_before_hook(tool_call: ToolCallArgs) -> ToolCallArgs:
    """Async hook that validates and enriches tool calls."""
    print(f"⚡ Async validating tool call: {tool_call['name']}")

    # Simulate async validation (e.g., checking permissions)
    await asyncio.sleep(0.1)

    # Add validation metadata
    tool_call["arguments"]["validated"] = True
    tool_call["arguments"]["validation_timestamp"] = time.time()

    return tool_call


async def async_formatting_after_hook(
    tool_call: ToolCallArgs, result: CallToolResult
) -> tuple[str, list[Any]]:
    """Async hook that formats results with external data."""
    print(f"🎨 Async formatting result for: {tool_call['name']}")

    # Simulate async formatting (e.g., fetching additional context)
    await asyncio.sleep(0.05)

    # Extract text
    text_parts = []
    for content in result.content:
        if hasattr(content, 'text'):
            text_parts.append(content.text)

    formatted_content = f"""
    📊 Tool Execution Report
    ========================
    Tool: {tool_call['name']}
    User: {tool_call['arguments'].get('user_id', 'unknown')}
    Result: {' '.join(text_parts)}
    Processed at: {time.strftime('%Y-%m-%d %H:%M:%S')}
    """

    return formatted_content.strip(), [{"format": "execution_report"}]


# Example 3: Notification callbacks
def message_callback(event: MessageEvent) -> None:
    """Handle server messages."""
    level = event.get("level", "info")
    message = event["message"]
    source = event.get("source", "unknown")

    emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(level, "📢")
    print(f"{emoji} [{source}] {message}")


def progress_callback(event: ProgressEvent) -> None:
    """Handle progress updates."""
    progress = event.get("progress", 0)
    source = event.get("source", "tool")
    total = event.get("total")
    completed = event.get("completed")

    if total and completed:
        print(f"🔄 {source}: {completed}/{total} ({progress:.1f}%)")
    else:
        print(f"🔄 {source}: {progress:.1f}%")


async def tools_list_changed_callback(tools: list[str]) -> None:
    """Handle tools list changes."""
    print(f"🔧 Available tools updated: {len(tools)} tools")
    print(f"   Tools: {', '.join(tools[:5])}{'...' if len(tools) > 5 else ''}")


# Example 4: Different hook configurations
def create_debug_config() -> InterceptorConfig:
    """Create interceptor config for debugging."""
    return {
        "before_tool_call": lambda tc: {
            **tc,
            "arguments": {**tc["arguments"], "debug": True}
        },
        "after_tool_call": lambda tc, result: f"DEBUG: {result.content[0].text if result.content else 'no content'}",
        "on_message": lambda event: print(f"🐛 DEBUG: {event}"),
    }


def create_production_config() -> InterceptorConfig:
    """Create interceptor config for production."""
    return {
        "before_tool_call": auth_before_hook,
        "after_tool_call": logging_after_hook,
        "on_message": message_callback,
        "on_progress": progress_callback,
        "on_tools_list_changed": tools_list_changed_callback,
    }


def create_async_config() -> InterceptorConfig:
    """Create interceptor config with async hooks."""
    return {
        "before_tool_call": async_validation_before_hook,
        "after_tool_call": async_formatting_after_hook,
        "on_message": message_callback,
        "on_progress": progress_callback,
    }


# Example usage function
async def main():
    """Demonstrate different interceptor configurations."""

    # Example connection (replace with your actual MCP server)
    connections = {
        "math": {
            "command": "python",
            "args": ["/path/to/your/math_server.py"],
            "transport": "stdio",
        }
    }

    print("=" * 60)
    print("🚀 MCP Interceptors Example")
    print("=" * 60)

    # Example 1: Basic interceptors
    print("\n1️⃣ Basic Authentication & Logging Interceptors")
    client1 = MultiServerMCPClient(
        connections=connections,
        interceptors=create_production_config(),
    )

    try:
        tools1 = await client1.get_tools()
        print(f"✅ Loaded {len(tools1)} tools with basic interceptors")

        # Example tool call (if tools are available)
        if tools1:
            tool = tools1[0]
            print(f"🔧 Calling tool: {tool.name}")
            # result = await tool.ainvoke({"value": 42})
            # print(f"📤 Result: {result}")
    except Exception as e:
        print(f"❌ Error with basic interceptors: {e}")

    # Example 2: Async interceptors
    print("\n2️⃣ Async Validation & Formatting Interceptors")
    client2 = MultiServerMCPClient(
        connections=connections,
        interceptors=create_async_config(),
    )

    try:
        tools2 = await client2.get_tools()
        print(f"✅ Loaded {len(tools2)} tools with async interceptors")
    except Exception as e:
        print(f"❌ Error with async interceptors: {e}")

    # Example 3: Debug configuration
    print("\n3️⃣ Debug Configuration")
    client3 = MultiServerMCPClient(
        connections=connections,
        interceptors=create_debug_config(),
    )

    try:
        tools3 = await client3.get_tools()
        print(f"✅ Loaded {len(tools3)} tools with debug config")
    except Exception as e:
        print(f"❌ Error with debug config: {e}")

    print("\n" + "=" * 60)
    print("✨ Interceptors example completed!")
    print("=" * 60)


# Example of using interceptors with specific session
async def session_example():
    """Example using interceptors with explicit session management."""

    connections = {
        "math": {
            "command": "python",
            "args": ["/path/to/your/math_server.py"],
            "transport": "stdio",
        }
    }

    client = MultiServerMCPClient(
        connections=connections,
        interceptors={
            "before_tool_call": lambda tc: {**tc, "arguments": {**tc["arguments"], "session_managed": True}},
            "on_message": lambda event: print(f"📨 Session message: {event}"),
        }
    )

    # Using explicit session management
    async with client.session("math") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools

        # Load tools with interceptors applied
        tools = await load_mcp_tools(session, interceptors=client.interceptors)
        print(f"📦 Loaded {len(tools)} tools in managed session")


if __name__ == "__main__":
    # Note: Update the connection paths before running
    print("📝 This is an example file. Update connection paths before running.")
    print("🔧 Uncomment the line below to run the example:")
    # asyncio.run(main())