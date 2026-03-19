"""
sources/metaculus.py
Metaculus public API client.

Metaculus is a forecasting platform focused on science, technology, and policy.
API is public; optional token for higher rate limits.

Docs: https://www.metaculus.com/api/
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

METACULUS_BASE = "https://www.metaculus.com/api2"


class MetaculusClient:
    def __init__(self, api_token: str | None = None, timeout: int = 15):
        self.timeout = timeout
        token = api_token or os.environ.get("METACULUS_API_TOKEN", "")
        self._headers = {"Authorization": f"Token {token}"} if token else {}

    async def search(
        self,
        query: str,
        limit: int = 20,
        question_type: str = "forecast",   # forecast | binary | numeric | date | multiple_choice
    ) -> list[dict[str, Any]]:
        """
        Search questions by keyword. Returns questions with community forecast.
        """
        params = {
            "search": query,
            "limit": limit,
            "order_by": "-activity",
            "status": "open",
        }
        if question_type:
            params["type"] = question_type

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{METACULUS_BASE}/questions/",
                    params=params,
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Metaculus search failed for %r: %s", query, exc)
            return []

        results = data.get("results", data) if isinstance(data, dict) else data
        return [self._normalise(q) for q in results if isinstance(q, dict)]

    @staticmethod
    def _normalise(q: dict) -> dict[str, Any]:
        """Flatten a Metaculus question into a consistent schema."""
        # Community prediction is nested differently by question type
        cp = q.get("community_prediction") or {}
        if isinstance(cp, dict):
            probability = cp.get("full", {}).get("q2") or cp.get("q2")
        else:
            probability = None

        resolution_criteria = (q.get("resolution_criteria") or "")[:400]
        title = q.get("title", "")
        url = f"https://www.metaculus.com{q.get('page_url', '')}" if q.get("page_url") else ""

        prob_str = f"{round(float(probability) * 100)}% probability" if probability else "no forecast yet"

        return {
            "platform": "metaculus",
            "id": str(q.get("id", "")),
            "question": title,
            "description": (q.get("description") or "")[:500],
            "resolution_criteria": resolution_criteria,
            "probability": float(probability) if probability else None,
            "close_time": q.get("close_time", ""),
            "resolve_time": q.get("resolve_time", ""),
            "num_predictions": q.get("number_of_predictions", 0),
            "url": url,
            # Derived signal text
            "text": f"{title} [{prob_str}, {q.get('number_of_predictions', 0)} forecasters, Metaculus]",
        }
