"""
sources/polymarket.py
Polymarket public API client.

Polymarket is a prediction market focused on geopolitics, crypto, and current events.
API is public and unauthenticated for read operations.

Docs: https://docs.polymarket.com
Gamma API: https://gamma-api.polymarket.com
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


class PolymarketClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Search active markets by keyword.
        Returns a list of market dicts with probability, volume, and metadata.
        """
        params = {
            "search": query,
            "active": "true",
            "closed": "false",
            "limit": limit,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{GAMMA_BASE}/markets", params=params)
                resp.raise_for_status()
                markets = resp.json()
        except Exception as exc:
            logger.warning("Polymarket search failed for %r: %s", query, exc)
            return []

        return [self._normalise(m) for m in (markets if isinstance(markets, list) else [])]

    async def get_market(self, market_id: str) -> dict[str, Any] | None:
        """Fetch a single market by its condition ID or slug."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{GAMMA_BASE}/markets/{market_id}")
                resp.raise_for_status()
                return self._normalise(resp.json())
        except Exception as exc:
            logger.warning("Polymarket get_market failed for %r: %s", market_id, exc)
            return None

    @staticmethod
    def _normalise(m: dict) -> dict[str, Any]:
        """Flatten Polymarket market into a consistent schema."""
        # Outcome prices come as a JSON string e.g. '["0.73", "0.27"]'
        import json as _json
        raw_prices = m.get("outcomePrices") or m.get("outcome_prices", "[]")
        try:
            prices = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            yes_prob = float(prices[0]) if prices else None
        except Exception:
            yes_prob = None

        return {
            "platform": "polymarket",
            "id": m.get("id") or m.get("conditionId", ""),
            "question": m.get("question", ""),
            "description": (m.get("description") or "")[:500],
            "yes_probability": yes_prob,
            "volume_usd": float(m.get("volume", 0) or 0),
            "liquidity_usd": float(m.get("liquidity", 0) or 0),
            "end_date": m.get("endDate") or m.get("end_date", ""),
            "active": m.get("active", True),
            "url": f"https://polymarket.com/event/{m.get('slug', m.get('id', ''))}",
            # Derived signal text for The Debate
            "text": (
                f"{m.get('question', '')} "
                f"[Polymarket: {round(yes_prob * 100)}% Yes, "
                f"${round(float(m.get('volume', 0) or 0) / 1000)}k volume]"
                if yes_prob is not None else m.get("question", "")
            ),
        }
