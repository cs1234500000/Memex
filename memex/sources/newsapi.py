"""
sources/newsapi.py
Async wrapper around NewsAPI.org /v2/everything.

Provides two usage modes:
  1. Simple query search         → NewsAPIClient.search(query)
     Used by the query-driven NewsAgent.
  2. Per-interest subscription   → NewsAPIClient.fetch_items_for_subscriptions(subs)
     Used by the passive IngestPipeline; mirrors newsApiSource.js logic.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone

import httpx

BASE_URL = "https://newsapi.org/v2/everything"


def _dedupe_by_url(articles: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for a in articles:
        key = a.get("url") or a.get("id", "")
        if key and key not in seen:
            seen[key] = a
    return list(seen.values())


def _map_article(article: dict) -> dict | None:
    """Canonical item shape shared with the rest of the pipeline."""
    url = str(article.get("url") or "").strip()
    title = str(article.get("title") or "").strip()
    description = str(article.get("description") or "").strip()
    text = ". ".join(filter(None, [title, description]))
    if not text:
        return None

    id_seed = url or f"{title}-{article.get('publishedAt', '')}"
    item_id = hashlib.sha1(id_seed.encode()).hexdigest()
    source_name = (article.get("source") or {}).get("name") or "newsapi"

    return {
        "id": item_id,
        "text": text,
        "author": str(article.get("author") or source_name),
        "handle": source_name.lower().replace(" ", "_"),
        "createdAt": article.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
        "metrics": {"likeCount": 0, "repostCount": 0, "replyCount": 0},
        "url": url,
        "source": "newsapi",
    }


class NewsAPIClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWSAPI_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Mode 1: query-driven (used by NewsAgent)                           #
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        page_size: int = 10,
        days_back: int = 7,
        language: str = "en",
    ) -> list[dict]:
        """Search NewsAPI by query string. Returns raw article dicts."""
        if not self.api_key:
            raise RuntimeError("NEWSAPI_KEY is not set.")

        from_date = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "pageSize": page_size,
            "language": language,
            "sortBy": "relevancy",
            "apiKey": self.api_key,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json().get("articles", [])

    # ------------------------------------------------------------------ #
    # Mode 2: subscription-driven (used by IngestPipeline)              #
    # ------------------------------------------------------------------ #

    async def fetch_items_for_subscriptions(self, subscriptions: dict) -> dict:
        """
        Fetch items for all interests in a subscriptions config.
        Mirrors fetchNewsApiItems() from newsApiSource.js.
        Returns {"items": [...], "warnings": [...]}.
        """
        if not self.api_key:
            raise RuntimeError("NEWSAPI_KEY / NEWSAPI_API_KEY is not set.")

        config = (subscriptions.get("sources") or {}).get("newsapi", {})
        interests = subscriptions.get("interests", [])
        max_items = int(subscriptions.get("max_items_per_run", 60) or 60)

        if not interests:
            return {"items": [], "warnings": []}

        per_interest_max = max(5, max_items // len(interests))
        all_articles: list[dict] = []
        warnings: list[str] = []

        for interest in interests:
            try:
                articles = await self._fetch_for_interest(
                    interest, per_interest_max, config
                )
                all_articles.extend(articles)
            except Exception as exc:
                warnings.append(
                    f"newsapi fetch failed for interest \"{interest.get('label')}\": {exc}"
                )

        # Dedupe → map to canonical shape → sort by date → trim
        items = [
            mapped
            for a in _dedupe_by_url(all_articles)
            if (mapped := _map_article(a)) is not None
        ]
        items.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
        items = items[:max_items]

        return {"items": items, "warnings": warnings}

    async def _fetch_for_interest(
        self, interest: dict, page_size: int, config: dict
    ) -> list[dict]:
        """Build keyword OR query and fetch articles for one interest."""
        keywords = [kw for kw in (interest.get("keywords") or [])[:8] if kw]
        if not keywords:
            return []

        query = " OR ".join(f'"{kw}"' for kw in keywords)
        language = config.get("language") or config.get("lang", "en")
        sort_by = config.get("sort_by") or config.get("sortBy", "publishedAt")

        params: dict = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": str(min(page_size, 100)),
            "searchIn": "title,description,content",
            "apiKey": self.api_key,
        }

        domains = config.get("domains", [])
        source_ids = config.get("source_ids") or config.get("sourceIds", [])
        if domains:
            params["domains"] = ",".join(domains)
        if source_ids:
            params["sources"] = ",".join(source_ids)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json().get("articles", [])
