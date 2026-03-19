"""
hackernews.py
Wrapper around the Algolia HN Search API (hn.algolia.com/api).
No auth required.
"""

from __future__ import annotations

import httpx


HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"


class HackerNewsClient:
    async def search(
        self,
        query: str,
        limit: int = 10,
        tags: str = "story",
        days_back: int = 30,
    ) -> list[dict]:
        """Return HN stories matching the query."""
        import time
        since = int(time.time()) - days_back * 86400

        params = {
            "query": query,
            "tags": tags,
            "numericFilters": f"created_at_i>{since}",
            "hitsPerPage": limit,
        }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(HN_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        return data.get("hits", [])

    async def get_item(self, item_id: int) -> dict:
        """Fetch a single HN item (story or comment) by ID."""
        url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
