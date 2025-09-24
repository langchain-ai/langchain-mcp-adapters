"""Test hooks and callbacks functionality."""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.lifecycle import (
    Callbacks,
    Hooks,
)


class TestMultiServerMCPClientIntegration:
    """Test MultiServerMCPClient integration with hooks and callbacks."""

    def test_client_init_with_hooks_and_callbacks(self):
        """Test MultiServerMCPClient initialization with hooks and callbacks."""

        async def mock_callback(notification, context):
            pass

        async def mock_hook(request, context):
            return None

        callbacks = Callbacks(on_logging_message=mock_callback)
        hooks = Hooks(before_tool_call=mock_hook)

        client = MultiServerMCPClient(
            connections={"test": {"command": "echo", "transport": "stdio"}},
            callbacks=callbacks,
            hooks=hooks,
        )

        assert client.callbacks == callbacks
        assert client.hooks == hooks

    def test_client_init_without_hooks_and_callbacks(self):
        """Test MultiServerMCPClient initialization without hooks and callbacks."""
        client = MultiServerMCPClient(
            connections={"test": {"command": "echo", "transport": "stdio"}}
        )

        assert client.callbacks is not None
        assert client.hooks is not None
