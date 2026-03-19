"""
sources/metaculus.py
Metaculus API v2 client.

Metaculus is a forecasting platform focused on science, technology, and policy.

API reference: https://metaculus-metaculus.mintlify.app/api/overview.md

Key facts about the current API:
  - Base URL : https://www.metaculus.com/api/
  - Auth     : required — Authorization: Token <token>
                 (unauthenticated requests return 401)
  - Search   : GET /api/posts/?search=<query>&statuses=open&with_cp=true
  - Posts wrap questions — each result is a post that contains a `question` object
  - Community prediction lives at question.aggregations.recency_weighted.latest.means[100]
    (index 100 of the 201-point CDF for binary questions → probability at 50% centile)

Get your token at: https://www.metaculus.com/accounts/settings/account/#api-access
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

METACULUS_BASE = "https://www.metaculus.com/api"


class MetaculusClient:
    def __init__(self, api_token: str | None = None, timeout: int = 20):
        self.timeout = timeout
        token = api_token or os.environ.get("METACULUS_API_TOKEN", "")
        if not token:
            logger.warning(
                "METACULUS_API_TOKEN not set — Metaculus API requires authentication. "
                "Get your token at https://www.metaculus.com/accounts/settings/account/#api-access"
            )
        self._headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        limit: int = 20,
        statuses: list[str] | None = None,
        forecast_type: str | None = None,
        order_by: str = "-hotness",
    ) -> list[dict[str, Any]]:
        """
        Search Metaculus posts (which contain questions) by keyword.

        Args:
            query:         Keyword search string
            limit:         Max results (default 20, API max 100)
            statuses:      Filter by status — ["open"], ["open", "closed"], etc.
            forecast_type: Filter by question type — "binary", "numeric", "date", "multiple_choice"
            order_by:      Sort order — "-hotness" (default), "-published_at", "-forecasts_count"
        """
        params: dict[str, Any] = {
            "search": query,
            "limit": min(limit, 100),
            "order_by": order_by,
            "with_cp": "true",      # include community predictions
            "include_description": "false",
        }

        for status in (statuses or ["open"]):
            params.setdefault("statuses", [])
            if isinstance(params["statuses"], list):
                params["statuses"].append(status)
            else:
                params["statuses"] = [params["statuses"], status]

        if forecast_type:
            params["forecast_type"] = forecast_type

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{METACULUS_BASE}/posts/",
                    params=params,
                    headers=self._headers,
                )
                if resp.status_code == 401:
                    logger.error(
                        "Metaculus API returned 401 — check METACULUS_API_TOKEN in .env"
                    )
                    return []
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Metaculus HTTP error for %r: %s", query, exc)
            return []
        except Exception as exc:
            logger.warning("Metaculus search failed for %r: %s", query, exc)
            return []

        results = data.get("results", [])
        normalised = []
        for post in results:
            item = self._normalise_post(post)
            if item:
                normalised.append(item)
        return normalised

    @staticmethod
    def _extract_probability(question: dict) -> float | None:
        """
        Extract community probability from a question's aggregations.

        The API returns a 201-point CDF for continuous questions and a single
        probability float for binary questions under:
          question.aggregations.recency_weighted.latest

        For binary: latest.means is a single float.
        For numeric/date: latest.means is a 201-element array; index 100 is the median.
        """
        try:
            agg = question.get("aggregations") or {}
            rw = agg.get("recency_weighted") or {}
            latest = rw.get("latest") or {}
            means = latest.get("means")

            if means is None:
                return None
            if isinstance(means, (int, float)):
                return float(means)
            if isinstance(means, list) and len(means) >= 101:
                # Numeric/date CDF — index 100 is the median (50th percentile)
                return float(means[100])
        except Exception:
            pass
        return None

    @staticmethod
    def _build_url(post: dict) -> str:
        slug = post.get("slug") or ""
        post_id = post.get("id") or ""
        if slug:
            return f"https://www.metaculus.com/questions/{slug}/"
        if post_id:
            return f"https://www.metaculus.com/questions/{post_id}/"
        return ""

    @classmethod
    def _normalise_post(cls, post: dict) -> dict[str, Any] | None:
        """Flatten a Metaculus post+question into a consistent schema."""
        question = post.get("question")
        if not question:
            return None  # notebooks, conditional parents etc.

        title = post.get("title") or question.get("title", "")
        url = cls._build_url(post)
        probability = cls._extract_probability(question)

        nr_forecasters = post.get("nr_forecasters", 0)
        forecasts_count = post.get("forecasts_count", 0)
        close_time = question.get("scheduled_close_time", "")
        resolve_time = question.get("scheduled_resolve_time", "")

        prob_str = (
            f"{round(probability * 100)}% probability"
            if probability is not None
            else "no forecast yet"
        )

        return {
            "platform":        "metaculus",
            "id":              str(post.get("id", "")),
            "question":        title,
            "type":            question.get("type", ""),
            "status":          question.get("status", ""),
            "probability":     probability,
            "close_time":      close_time,
            "resolve_time":    resolve_time,
            "nr_forecasters":  nr_forecasters,
            "forecasts_count": forecasts_count,
            "url":             url,
            "text": (
                f"{title} "
                f"[{prob_str}, {nr_forecasters} forecasters, "
                f"closes {close_time[:10] if close_time else 'n/d'}, Metaculus]"
            ),
        }
