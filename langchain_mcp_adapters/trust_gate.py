"""Trust gate interceptor for pre-call MCP server trust verification.

Checks behavioral trust scores from a configurable trust oracle before
allowing tool calls. Servers below the configured threshold are blocked
with a clear explanation.

The default trust oracle is Dominion Observatory
(https://dominion-observatory.sgdata.workers.dev), which monitors 14,820+
MCP servers. You can substitute any endpoint that returns a JSON object
with a ``trust_score`` field (0-100).

Example::

    from langchain_mcp_adapters.trust_gate import TrustGateInterceptor

    interceptor = TrustGateInterceptor(min_trust_score=60)
    client = MultiServerMCPClient(interceptors=[interceptor], ...)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage

from langchain_mcp_adapters.interceptors import MCPToolCallRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_mcp_adapters.interceptors import MCPToolCallResult


@dataclass
class TrustGateInterceptor:
    """Intercepts MCP tool calls and checks server trust before execution.

    Queries a trust oracle (default: Dominion Observatory) to retrieve
    a behavioral trust score for the target MCP server. If the score is
    below ``min_trust_score``, the call is blocked and a ``ToolMessage``
    with an explanation is returned instead.

    Trust scores are cached for ``cache_ttl_seconds`` to avoid excessive
    API calls.

    Attributes:
        observatory_url: Base URL of the trust oracle API.
        min_trust_score: Minimum trust score (0-100) required to proceed.
        api_key: Optional API key for paid tiers (enables audit receipts).
        cache_ttl_seconds: How long to cache trust scores (default 300s).
    """

    observatory_url: str = "https://dominion-observatory.sgdata.workers.dev"
    min_trust_score: float = 60.0
    api_key: str | None = None
    cache_ttl_seconds: int = 300
    _cache: dict[str, tuple[float, float]] = field(
        default_factory=dict, repr=False
    )

    async def _get_trust_score(self, server_name: str) -> dict[str, Any]:
        """Fetch trust score from the observatory, with caching."""
        now = time.time()
        cached = self._cache.get(server_name)
        if cached and (now - cached[1]) < self.cache_ttl_seconds:
            return {"trust_score": cached[0], "cached": True}

        import httpx

        url = f"{self.observatory_url}/benchmark/{server_name}"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    score = data.get("trust_score")
                    if score is not None:
                        self._cache[server_name] = (float(score), now)
                    return data
                return {"trust_score": None, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"trust_score": None, "error": str(e)}

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        """Check trust score before allowing tool execution.

        Args:
            request: The MCP tool call request.
            handler: The next handler in the interceptor chain.

        Returns:
            Tool result if trusted, or a ToolMessage explaining the block.
        """
        server_name = request.server_name
        trust_data = await self._get_trust_score(server_name)
        score = trust_data.get("trust_score")

        if score is not None and score >= self.min_trust_score:
            return await handler(request)

        # Server didn't meet trust threshold — block the call
        if score is None:
            reason = (
                f"Could not verify trust for MCP server '{server_name}'. "
                f"Trust oracle returned: {trust_data.get('error', 'unknown error')}. "
                f"The tool call was blocked for safety."
            )
        else:
            reason = (
                f"MCP server '{server_name}' has a trust score of {score:.1f}, "
                f"which is below the minimum threshold of {self.min_trust_score}. "
                f"The tool call '{request.name}' was blocked. "
                f"Check {self.observatory_url}/benchmark/{server_name} for details."
            )

        return ToolMessage(
            content=reason,
            tool_call_id=request.name,
            status="error",
        )
