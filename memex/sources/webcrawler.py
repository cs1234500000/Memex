"""
sources/webcrawler.py
General-purpose web crawl source powered by Tavily.

Tavily is an AI-native search API that combines web search and full-content
extraction in a single call — no separate reader step needed.

Strategy:
  1. Call Tavily Search with `include_raw_content=True` to get the top-N
     results, each with its full extracted page text already included.
  2. Map each result to the canonical item dict used by the rest of the
     pipeline.

Required env vars:
  TAVILY_API_KEY   — get a free key at https://app.tavily.com/home
                     (free tier: 1 000 API credits / month)

Docs: https://docs.tavily.com/welcome
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone

from tavily import AsyncTavilyClient

logger = logging.getLogger(__name__)


def _extract_domain(url: str) -> str:
    parts = url.split("/")
    return parts[2] if len(parts) > 2 else url


def _make_item(result: dict) -> dict:
    """Map a Tavily search result to the canonical pipeline item shape."""
    url = result.get("url", "")
    title = result.get("title", "")
    # raw_content is the full extracted text; fall back to the AI snippet
    text = (result.get("raw_content") or result.get("content") or "").strip()
    domain = _extract_domain(url)
    item_id = hashlib.sha1(url.encode()).hexdigest()
    return {
        "id": item_id,
        "title": title,
        "text": text,
        "author": domain,
        "handle": domain.replace(".", "_"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "metrics": {"likeCount": 0, "repostCount": 0, "replyCount": 0},
        "url": url,
        "source": "webcrawler",
        "domain": domain,
        "score": result.get("score"),
    }


class WebCrawlerClient:
    """
    Tavily-powered web crawler source.

    Searches the web and returns full-text results in one API call.

    Usage::

        client = WebCrawlerClient()
        items = await client.search("AI safety research 2025", top_n=10)
        # items[0] → {"id": ..., "title": ..., "text": "<full page text>", ...}
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")

    def _client(self) -> AsyncTavilyClient:
        if not self._api_key:
            raise RuntimeError(
                "WebCrawlerClient: TAVILY_API_KEY is not set. "
                "Get a free key at https://app.tavily.com/home"
            )
        return AsyncTavilyClient(api_key=self._api_key)

    MAX_RESULTS = 20

    async def search(
        self,
        query: str,
        top_n: int = 10,
        search_depth: str = "advanced",
        topic: str = "general",
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[dict]:
        """
        Search the web for `query` and return full-text results.

        Tavily fetches and extracts page content server-side, so results
        arrive with full text included — no secondary read step needed.

        Args:
            query:           Natural-language search query.
            top_n:           Number of results to return (capped at MAX_RESULTS=20).
            search_depth:    "basic" (fast) or "advanced" (thorough, uses more
                             credits).
            topic:           "general" | "news" — use "news" for current events.
            time_range:      Optional recency filter: "day" | "week" | "month" |
                             "year".
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude these domains from results.

        Returns:
            List of canonical item dicts ordered by Tavily relevance score.
        """
        top_n = min(top_n, self.MAX_RESULTS)
        logger.info("WebCrawler: Tavily search for %r (top %d)", query, top_n)

        kwargs: dict = {
            "query": query,
            "max_results": top_n,
            "search_depth": search_depth,
            "topic": topic,
            "include_raw_content": True,
        }
        if time_range:
            kwargs["time_range"] = time_range
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains

        response = await self._client().search(**kwargs)
        raw_results = response.get("results", [])

        if not raw_results:
            logger.warning("WebCrawler: no Tavily results for %r", query)
            return []

        items = [_make_item(r) for r in raw_results]
        logger.info("WebCrawler: %d items ready for %r", len(items), query)
        return items
