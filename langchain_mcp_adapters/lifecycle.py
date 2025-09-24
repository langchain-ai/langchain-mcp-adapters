"""Lifecycle management for MCP client callbacks and hooks.

This module provides callback and hook interfaces for intercepting and extending
MCP client behavior, similar to the LangChain JS implementation.
"""

# Re-export from the specialized modules for backward compatibility
from langchain_mcp_adapters.callbacks import (
    CallbackContext,
    Callbacks,
    OnLoggingMessageCallback,
    OnProgressNotificationCallback,
)
from langchain_mcp_adapters.hooks import (
    AfterToolCallHook,
    BeforeToolCallHook,
    BeforeToolCallResult,
    HookContext,
    Hooks,
)

__all__ = [
    "AfterToolCallHook",
    "BeforeToolCallHook",
    "BeforeToolCallResult",
    "CallbackContext",
    "Callbacks",
    "HookContext",
    "Hooks",
    "OnLoggingMessageCallback",
    "OnProgressNotificationCallback",
]
