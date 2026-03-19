"""
agents/news_agent.py
NewsAgent — monitors breaking news, press releases, and mainstream media.

Sources (in priority order):
  1. NewsAPI.org     — broad article search, past 30 days
  2. RSS feeds       — Reuters, AP, BBC, FT (real-time wire copy)
  3. Jina Reader     — full-text fetch for top articles

Strengths: facts, timelines, official narratives.
Known bias: recency bias; favours official sources; corporate PR can slip through.
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex.forum.base_agent import BaseAgent, ContentItem, AgentReport
from memex.forum.decomposer import NewsQuery
from memex.sources.newsapi import NewsAPIClient
from memex.sources.rss import RSSClient
from memex.sources.jina import JinaClient
from memex.ingest.enricher import is_fetchable

RSS_FEEDS = {
    "reuters":  "https://feeds.reuters.com/reuters/topNews",
    "ap":       "https://feeds.apnews.com/rss/apf-topnews",
    "bbc":      "http://feeds.bbci.co.uk/news/world/rss.xml",
    "ft":       "https://www.ft.com/rss/home",
}


class NewsAgent(BaseAgent):
    agent_type = "news"
    agent_label = "NewsAgent"
    caveats_text = (
        "Prone to recency bias and official narratives. Corporate press releases "
        "and government statements may be over-represented. RSS wire copy may lack "
        "depth; Jina-fetched full text may hit paywalls."
    )

    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        super().__init__(client, model)
        self.newsapi = NewsAPIClient(api_key=os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWSAPI_API_KEY", ""))
        self.rss = RSSClient()
        self.jina = JinaClient()

    async def run_decomposed(self, query: NewsQuery) -> AgentReport:
        items = await self._fetch_with_context(
            search=query.search_string,
            days_back=query.time_window_days,
        )
        for item in items:
            item.relevance_score = await self.score(item, query.search_string)
        summary, findings = await self._produce_report([query.search_string], items)
        return AgentReport(
            agent_type=self.agent_type, agent_label=self.agent_label,
            sub_queries=[query.search_string], items=items,
            summary=summary, findings=findings, caveats=self.caveats_text,
        )

    async def fetch(self, sub_query: str) -> list[ContentItem]:
        return await self._fetch_with_context(search=sub_query, days_back=14)

    async def _fetch_with_context(self, search: str, days_back: int = 14) -> list[ContentItem]:
        # 1. Parallel fetch from NewsAPI and RSS
        newsapi_results, rss_results = await asyncio.gather(
            self.newsapi.search(search, page_size=10, days_back=days_back),
            self.rss.search(search, feeds=RSS_FEEDS, limit_per_feed=5),
            return_exceptions=True,
        )

        raw_articles: list[dict] = []
        if not isinstance(newsapi_results, Exception):
            raw_articles.extend(newsapi_results)
        if not isinstance(rss_results, Exception):
            raw_articles.extend(rss_results)

        # 2. Enrich top 6 fetchable articles with Jina Reader
        fetchable = [a for a in raw_articles if is_fetchable(a.get("url", ""))[0]]
        full_texts = await asyncio.gather(
            *[self._read_full_text(a["url"]) for a in fetchable[:6]],
            return_exceptions=True,
        )
        url_to_body: dict[str, str] = {}
        for i, ft in enumerate(full_texts):
            if not isinstance(ft, Exception) and ft:
                url_to_body[fetchable[i]["url"]] = ft

        # 3. Build ContentItems
        items: list[ContentItem] = []
        for art in raw_articles:
            url = art.get("url", "")
            body = (
                url_to_body.get(url)
                or art.get("description") or art.get("text", "")
            )
            if not body:
                continue
            items.append(ContentItem(
                title=art.get("title", ""),
                url=url,
                body=body[:2000],
                source=art.get("source") or "newsapi",
                published_at=art.get("publishedAt") or art.get("published_at", ""),
                metadata={"source_name": art.get("source", {}).get("name") if isinstance(art.get("source"), dict) else art.get("source")},
            ))

        return items

    async def _read_full_text(self, url: str) -> str | None:
        try:
            result = await self.jina.read(url)
            if len(result.content) > 300:
                return result.content[:3000]
        except Exception:
            pass
        return None
