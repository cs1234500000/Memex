"""
reddit.py
Async wrapper around PRAW (Python Reddit API Wrapper).
PRAW is synchronous; we run it in a thread executor to avoid blocking.
"""

from __future__ import annotations

import asyncio
import os
from functools import partial

import praw


class RedditClient:
    def __init__(self):
        self._client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        self._client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        self._reddit = None

    def _get_reddit(self):
        if not self._client_id or not self._client_secret:
            raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set.")
        if self._reddit is None:
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent="memex-research-bot/1.0",
            )
        return self._reddit

    async def search(
        self,
        query: str,
        subreddits: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search Reddit and return serialised post dicts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self._sync_search, query, subreddits, limit)
        )

    def _sync_search(
        self,
        query: str,
        subreddits: list[str] | None,
        limit: int,
    ) -> list[dict]:
        target = "+".join(subreddits) if subreddits else "all"
        sub = self._get_reddit().subreddit(target)
        results = []
        for post in sub.search(query, sort="relevance", time_filter="week", limit=limit):
            post.comments.replace_more(limit=0)
            top_comments = " | ".join(
                c.body[:200] for c in post.comments.list()[:5]
            )
            results.append(
                {
                    "title": post.title,
                    "url": f"https://reddit.com{post.permalink}",
                    "selftext": post.selftext[:1000],
                    "top_comments": top_comments,
                    "subreddit": post.subreddit.display_name,
                    "score": post.score,
                    "num_comments": post.num_comments,
                }
            )
        return results
