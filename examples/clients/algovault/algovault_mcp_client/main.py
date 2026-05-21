"""AlgoVault MCP client demo.

Connects to the public AlgoVault MCP server at ``https://api.algovault.com/mcp``
via ``MultiServerMCPClient``, loads tools, and invokes ``get_trade_call`` for a
single coin + timeframe. Prints the parsed ``{call, confidence, indicators,
regime}`` JSON to stdout.

No LLM API key required. The optional ReAct-agent variant is documented in
README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

ALGOVAULT_MCP_URL = "https://api.algovault.com/mcp"

# The four documented contract keys on ``get_trade_call``'s payload.
TRADE_CALL_KEYS = ("call", "confidence", "indicators", "regime")


def _extract_payload(raw: Any) -> dict:
    """Parse the JSON string out of a ``langchain-mcp-adapters`` tool response.

    ``langchain-mcp-adapters==0.2.2`` returns tool output as one of:
    - a ``str`` (already-extracted text), or
    - a ``list`` of dicts shaped ``{"type": "text", "text": "<json>", ...}``, or
    - a ``list`` of objects with a ``.text`` attribute.

    This helper handles all three shapes so the example survives minor
    upstream-shape changes.
    """
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            text = first.get("text", "")
        else:
            text = getattr(first, "text", "")
    else:
        text = ""

    if not text:
        raise ValueError(f"unexpected tool-response shape: {raw!r}")
    return json.loads(text)


async def fetch_trade_call(coin: str, timeframe: str) -> dict:
    """Fetch a composite trade call for one coin + timeframe.

    Args:
        coin: e.g. ``"BTC"``, ``"ETH"``, ``"SOL"``.
        timeframe: e.g. ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``.

    Returns:
        ``{"call": "BUY"|"SELL"|"HOLD", "confidence": int,
           "indicators": {...}, "regime": str}``.
    """
    client = MultiServerMCPClient(
        {
            "algovault": {
                "url": ALGOVAULT_MCP_URL,
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    by_name = {t.name: t for t in tools}
    if "get_trade_call" not in by_name:
        raise RuntimeError(
            f"get_trade_call not found in tools; got {sorted(by_name)!r}"
        )

    raw = await by_name["get_trade_call"].ainvoke(
        {"coin": coin, "timeframe": timeframe}
    )
    payload = _extract_payload(raw)
    return {key: payload[key] for key in TRADE_CALL_KEYS}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="algovault-mcp-client",
        description=(
            "Fetch a composite trade call from AlgoVault MCP via "
            "langchain-mcp-adapters."
        ),
    )
    parser.add_argument(
        "coin",
        nargs="?",
        default="BTC",
        help="Coin symbol (default: BTC)",
    )
    parser.add_argument(
        "timeframe",
        nargs="?",
        default="4h",
        help="Timeframe: 15m, 1h, 4h, 1d (default: 4h)",
    )
    return parser.parse_args(argv)


async def main(coin: str = "BTC", timeframe: str = "4h") -> int:
    result = await fetch_trade_call(coin, timeframe)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cli() -> None:
    args = _parse_args()
    sys.exit(asyncio.run(main(args.coin, args.timeframe)))


if __name__ == "__main__":
    cli()
