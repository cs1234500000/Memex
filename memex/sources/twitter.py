"""
sources/twitter.py
Twitter API v2 recent-search wrapper.
Migrated from src/news/sources/twitterApiSource.js

Searches tweets from a configured list of handles.
Returns items in the canonical shape used by the ingest pipeline.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import httpx

TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


def _build_query(handles: list[str]) -> str:
    """Mirrors buildSearchQuery() — searches tweets from specified handles."""
    from_parts = [f"from:{h}" for h in handles]
    return f"({' OR '.join(from_parts)}) -is:retweet lang:en"


def _build_user_map(includes: dict) -> dict[str, dict]:
    users = (includes or {}).get("users", [])
    return {u["id"]: u for u in users}


def _map_tweet(tweet: dict, user_map: dict[str, dict]) -> dict:
    """Mirrors mapTweet() from twitterApiSource.js."""
    user = user_map.get(tweet.get("author_id", ""), {})
    handle = user.get("username", "unknown")
    metrics = tweet.get("public_metrics", {})
    return {
        "id": str(tweet["id"]),
        "text": tweet.get("text", ""),
        "author": user.get("name", handle),
        "handle": handle,
        "createdAt": tweet.get("created_at", datetime.now(timezone.utc).isoformat()),
        "metrics": {
            "likeCount": metrics.get("like_count", 0),
            "repostCount": metrics.get("retweet_count", 0),
            "replyCount": metrics.get("reply_count", 0),
        },
        "url": f"https://x.com/{handle}/status/{tweet['id']}",
        "source": "twitter",
    }


class TwitterClient:
    def __init__(self, bearer_token: str | None = None):
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN", "")

    async def search(
        self,
        handles: list[str],
        max_results: int = 60,
    ) -> list[dict]:
        """
        Search recent tweets from the given handles.
        Returns a list of canonical item dicts.
        """
        if not self.bearer_token:
            raise RuntimeError("TWITTER_BEARER_TOKEN is not set.")
        if not handles:
            return []

        query = _build_query(handles)
        limit = min(max_results, 100)

        params = {
            "query": query,
            "max_results": str(limit),
            "tweet.fields": "author_id,created_at,public_metrics",
            "expansions": "author_id",
            "user.fields": "name,username",
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                TWITTER_SEARCH_URL,
                params=params,
                headers={"Authorization": f"Bearer {self.bearer_token}"},
            )
            resp.raise_for_status()
            payload = resp.json()

        user_map = _build_user_map(payload.get("includes", {}))
        return [
            _map_tweet(t, user_map)
            for t in (payload.get("data") or [])
        ][:max_results]

    async def fetch_items(self, subscriptions: dict) -> dict:
        """
        Adapter used by IngestPipeline.
        Mirrors fetchTwitterApiItems() from twitterApiSource.js.
        """
        handles = [
            h.lstrip("@").strip()
            for h in subscriptions.get("twitter_handles", [])
            if h
        ]
        max_items = min(
            int(subscriptions.get("max_tweets_per_run", 60) or 60), 100
        )
        try:
            items = await self.search(handles, max_results=max_items)
            return {"items": items, "warnings": []}
        except Exception as exc:
            return {"items": [], "warnings": [str(exc)]}
