"""
sources/polymarket.py
Polymarket public API client.

Polymarket is the world's largest prediction market (crypto/geopolitics/current events).
All read endpoints are fully public — no authentication required.

APIs used:
  Gamma API  https://gamma-api.polymarket.com
    GET /public-search     Search markets, events by keyword → primary entry point
    GET /events            Browse events by tag / category
    GET /markets/{id}      Full market detail (resolution criteria, volume, prob)

  CLOB API   https://clob.polymarket.com
    GET /prices-history    Probability drift over time (interval=1w) — more
                           valuable than the current snapshot alone because it
                           shows whether smart money is becoming more or less
                           confident.

Docs: https://docs.polymarket.com/api-reference/introduction
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GAMMA  = "https://gamma-api.polymarket.com"
CLOB   = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    # ── Primary: keyword search ────────────────────────────────────────────────

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Search markets and events by keyword via GET /public-search.
        Returns normalised market dicts with current probability, volume,
        and 1-week price drift from the CLOB.
        """
        params = {
            "q": query,
            "limit_per_type": limit,
            "events_status": "open",   # only active markets
            "sort": "volume24hr",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{GAMMA}/public-search", params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Polymarket /public-search failed for %r: %s", query, exc)
            return []

        # /public-search returns {events: [...], tags: [...], profiles: [...]}
        # Each event contains its nested markets list
        events: list[dict] = data.get("events") or []
        markets: list[dict] = []
        for event in events:
            for m in event.get("markets") or []:
                markets.append({**m, "_event": event})

        if not markets:
            return []

        # Enrich top-N markets with price drift from CLOB (parallelised)
        top = sorted(markets, key=lambda m: float(m.get("volume") or 0), reverse=True)[:8]
        drift_tasks = [self._price_drift(m) for m in top]
        drifts = await asyncio.gather(*drift_tasks, return_exceptions=True)

        results = []
        for m, drift in zip(top, drifts):
            drift_val = drift if not isinstance(drift, Exception) else None
            results.append(self._normalise(m, drift_val))

        return results

    # ── Event browser ──────────────────────────────────────────────────────────

    async def get_events(
        self, tag: str = "", limit: int = 20, active: bool = True
    ) -> list[dict[str, Any]]:
        """
        Browse events via GET /events, optionally filtered by tag slug.
        Useful when MarketQuery specifies a thematic area rather than keywords.
        """
        params: dict = {"limit": limit}
        if tag:
            params["tag"] = tag
        if active:
            params["active"] = "true"
            params["closed"] = "false"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{GAMMA}/events", params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Polymarket /events failed (tag=%r): %s", tag, exc)
            return []

        events = data if isinstance(data, list) else data.get("results", [])
        out = []
        for event in events:
            for m in event.get("markets") or []:
                out.append(self._normalise({**m, "_event": event}, None))
        return out

    # ── CLOB: probability drift ────────────────────────────────────────────────

    async def _price_drift(self, market: dict) -> float | None:
        """
        Fetch 1-week price history for the YES token from the CLOB.
        Returns the change in probability over the period (positive = rising,
        negative = falling). None if the market has no CLOB token.
        """
        # clobTokenIds is a JSON-encoded array e.g. '["0xabc...", "0xdef..."]'
        raw_ids = market.get("clobTokenIds") or market.get("clob_token_ids", "[]")
        try:
            token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            yes_token = token_ids[0] if token_ids else None
        except Exception:
            yes_token = None

        if not yes_token:
            return None

        params = {
            "market": yes_token,
            "interval": "1w",
            "fidelity": 60,   # hourly data points (60-minute resolution)
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{CLOB}/prices-history", params=params)
                resp.raise_for_status()
                history = resp.json().get("history", [])
        except Exception as exc:
            logger.debug("CLOB prices-history failed for token %s: %s", yes_token, exc)
            return None

        if len(history) < 2:
            return None

        # history items: {t: unix_timestamp, p: price (0-1)}
        first_p = float(history[0]["p"])
        last_p  = float(history[-1]["p"])
        return round(last_p - first_p, 4)   # positive = odds rising

    # ── Normaliser ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(m: dict, drift: float | None) -> dict[str, Any]:
        """Flatten a Gamma market + optional CLOB drift into a consistent schema."""
        event: dict = m.pop("_event", {}) if "_event" in m else {}

        # Outcome prices: JSON string e.g. '["0.73", "0.27"]' (YES / NO)
        raw_prices = m.get("outcomePrices") or "[]"
        try:
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            yes_prob = float(prices[0]) if prices else None
        except Exception:
            yes_prob = None

        slug = m.get("slug") or event.get("slug", "")
        question = m.get("question") or event.get("title", "")
        volume = float(m.get("volume") or event.get("volume") or 0)
        end_date = m.get("endDate") or event.get("endDate", "")

        # Build a signal string: what does this market tell The Debate?
        prob_str = f"{round(yes_prob * 100)}% Yes" if yes_prob is not None else "prob unknown"
        drift_str = ""
        if drift is not None:
            direction = "↑" if drift > 0.01 else ("↓" if drift < -0.01 else "~")
            drift_str = f", {direction}{abs(round(drift * 100, 1))}pp/wk"
        vol_str = f"${round(volume / 1000)}k vol" if volume >= 1000 else f"${round(volume)} vol"

        return {
            "platform": "polymarket",
            "id": m.get("conditionId") or m.get("id", ""),
            "question": question,
            "description": (m.get("description") or event.get("description", ""))[:500],
            "resolution_source": m.get("resolutionSource") or event.get("resolutionSource", ""),
            "yes_probability": yes_prob,
            "drift_1w": drift,             # prob change over past week
            "volume_usd": volume,
            "end_date": end_date,
            "active": m.get("active", True),
            "url": f"https://polymarket.com/event/{slug}" if slug else "",
            # Derived signal text for The Debate
            "text": f"{question} [Polymarket: {prob_str}{drift_str}, {vol_str}]",
        }
