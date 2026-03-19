"""
agents/social_agent.py
SocialAgent — tracks grassroots reaction, sentiment, and emerging discourse.

Sources:
  1. Reddit via PRAW  — subreddit search + top-level comments on hot threads
  2. Hacker News via Algolia API — top stories + discussion thread text

Strengths: real-time public sentiment, contrarian views, practitioner reactions.
Known bias: selection bias toward vocal minorities; HN skews technical/VC; Reddit
skews young English-speaking demographics. Upvote mechanics surface outrage
and strong opinions over moderate nuance.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex.forum.base_agent import BaseAgent, ContentItem, AgentReport
from memex.forum.decomposer import SocialQuery
from memex.sources.reddit import RedditClient
from memex.sources.hackernews import HackerNewsClient


# Subreddits likely to have substantive discussion on research/tech/geopolitics
DEFAULT_SUBREDDITS = [
    "worldnews", "technology", "MachineLearning", "artificial",
    "geopolitics", "Economics", "Futurology",
]


class SocialAgent(BaseAgent):
    agent_type = "social"
    agent_label = "SocialAgent"
    caveats_text = (
        "Selection bias toward vocal minorities. HN skews technical/VC community; "
        "Reddit demographics are young, English-speaking, and Western. Upvote "
        "mechanics amplify emotional or contrarian takes. May miss expert consensus "
        "entirely in favour of popular narratives."
    )

    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        super().__init__(client, model)
        self.reddit = RedditClient()
        self.hn = HackerNewsClient()

    async def run_decomposed(self, query: SocialQuery) -> AgentReport:
        reddit_tasks = [
            self._fetch_reddit(term, subreddits=query.subreddits, min_score=query.min_score)
            for term in (query.search_terms or [""])
        ]
        results = await asyncio.gather(
            *reddit_tasks,
            self._fetch_hn(query.hn_query),
            return_exceptions=True,
        )

        hn_result = results[-1]
        reddit_results = results[:-1]

        items: list[ContentItem] = []
        seen_urls: set[str] = set()
        for result in reddit_results:
            if isinstance(result, Exception):
                continue
            for item in result:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    items.append(item)
        if not isinstance(hn_result, Exception):
            for item in hn_result:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    items.append(item)

        scoring_query = query.search_terms[0] if query.search_terms else query.hn_query
        for item in items:
            item.relevance_score = await self.score(item, scoring_query)
        summary, findings = await self._produce_report(query.search_terms, items)
        return AgentReport(
            agent_type=self.agent_type, agent_label=self.agent_label,
            sub_queries=query.search_terms, items=items,
            summary=summary, findings=findings, caveats=self.caveats_text,
        )

    async def fetch(self, sub_query: str) -> list[ContentItem]:
        reddit_items, hn_items = await asyncio.gather(
            self._fetch_reddit(sub_query),
            self._fetch_hn(sub_query),
            return_exceptions=True,
        )

        items: list[ContentItem] = []
        if not isinstance(reddit_items, Exception):
            items.extend(reddit_items)
        if not isinstance(hn_items, Exception):
            items.extend(hn_items)
        return items

    async def _fetch_reddit(
        self, query: str, subreddits: list[str] | None = None, min_score: int = 25
    ) -> list[ContentItem]:
        posts = await self.reddit.search(query, subreddits=subreddits, limit=15)
        items: list[ContentItem] = []
        for post in posts:
            if post.get("score", 0) < min_score:
                continue
            selftext = (post.get("selftext") or "").strip()
            comments = post.get("top_comments") or ""
            if isinstance(comments, list):
                comments = "\n".join(str(c) for c in comments[:5])
            body = "\n\n".join(filter(None, [selftext, comments]))[:2000]
            if not body and not post.get("title"):
                continue
            items.append(ContentItem(
                title=post.get("title", ""),
                url=post.get("url", ""),
                body=body or post.get("title", ""),
                source="reddit",
                published_at=str(post.get("created_utc", "")),
                metadata={
                    "subreddit": post.get("subreddit"),
                    "upvotes": post.get("score"),
                    "num_comments": post.get("num_comments"),
                },
            ))
        return items

    async def _fetch_hn(self, query: str) -> list[ContentItem]:
        stories = await self.hn.search(query, limit=15, days_back=30)
        items: list[ContentItem] = []
        for story in stories:
            # Prefer story_text (self-post) > comment_text > title
            body = (
                story.get("story_text")
                or story.get("comment_text")
                or story.get("title", "")
            )[:2000]
            hn_url = f"https://news.ycombinator.com/item?id={story.get('objectID', '')}"
            items.append(ContentItem(
                title=story.get("title") or story.get("story_title", ""),
                url=story.get("url") or hn_url,
                body=body,
                source="hackernews",
                published_at=story.get("created_at", ""),
                metadata={
                    "points": story.get("points"),
                    "num_comments": story.get("num_comments"),
                    "author": story.get("author"),
                    "hn_discussion": hn_url,
                },
            ))
        return items
