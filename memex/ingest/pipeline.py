"""
ingest/pipeline.py
Passive ingestion pipeline: fetch news → filter → per-item analysis
→ topic summarization → persist to DB.

This is the scheduled / background mode that runs against a user's
subscriptions config rather than an interactive query.
It mirrors the original pipeline.js + summarizer.js + itemInsights.js logic.

Usage:
    from memex.ingest.pipeline import IngestPipeline
    pipeline = IngestPipeline(client=openai_client, db_conn=conn)
    report = await pipeline.run(user_id="alice", subscriptions={...})
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from memex.ingest.filter import (
    FilteredItem,
    filter_items,
    group_by_topic,
)
from memex.ingest.summarizer import summarize_topic_group
from memex.sources.newsapi import NewsAPIClient
from memex.sources.twitter import TwitterClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Default subscriptions (mirrors defaults.js)                         #
# ------------------------------------------------------------------ #

DEFAULT_SUBSCRIPTIONS: dict[str, Any] = {
    "source": "newsapi",
    "twitter_handles": [
        "OpenAI", "AnthropicAI", "perplexity_ai", "ycombinator", "a16z", "sama"
    ],
    "sources": {
        "newsapi": {"language": "en", "sort_by": "publishedAt", "domains": [], "source_ids": []}
    },
    "interests": [
        {
            "id": "ai_startups",
            "label": "AI startup companies",
            "keywords": [
                "ai startup", "seed round", "series a", "series b",
                "ai agent", "llm startup", "language model", "ai funding",
                "foundation model", "enterprise ai", "generative ai",
            ],
        }
    ],
    "max_items_per_run": 60,
    "max_tweets_per_run": 60,
}


# ------------------------------------------------------------------ #
# Report data structures                                               #
# ------------------------------------------------------------------ #

@dataclass
class ItemAnalysis:
    summary: str
    insights: list[str]


@dataclass
class TopicReport:
    topic_id: str
    topic_label: str
    matched_count: int
    summary: str
    insights: list[str]
    top_items: list[dict]
    all_items: list[dict] = field(default_factory=list)


@dataclass
class IngestReport:
    id: str
    generated_at: str
    source: str
    subscriptions: dict
    total_fetched_items: int
    warnings: list[str]
    topics: list[TopicReport]
    user_id: str = "default"


# ------------------------------------------------------------------ #
# Per-item analysis (mirrors itemInsights.js)                         #
# ------------------------------------------------------------------ #

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _summarize_text(text: str) -> str:
    sentences = _split_sentences(text)
    base = sentences[0] if sentences else text.strip()
    if not base:
        return "No concise summary available."
    return base[:220] + "..." if len(base) > 220 else base


def _matched_keywords(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    return [kw for kw in keywords if kw and kw.lower() in lower]


def analyze_item(item: dict, topic_label: str, keywords: list[str]) -> ItemAnalysis:
    matched = _matched_keywords(item.get("text", ""), keywords)
    published_at = item.get("createdAt", "")
    handle = item.get("handle", "unknown")
    source = item.get("source", "unknown")
    url = item.get("url", "")

    insights = [
        f"Topic: {topic_label}.",
        f"Source: {source} via {handle}; published at {published_at or 'unknown time'}.",
        (
            f"Matched keywords: {', '.join(matched[:8])}."
            if matched
            else "No direct keyword match — passed LLM filter on semantic grounds."
        ),
        (
            "Original link available for full context and verification."
            if url
            else "No canonical link available in source payload."
        ),
    ]

    return ItemAnalysis(
        summary=_summarize_text(item.get("text", "")),
        insights=[line for line in insights if line],
    )


# ------------------------------------------------------------------ #
# Pipeline                                                            #
# ------------------------------------------------------------------ #

def _build_report_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")


class IngestPipeline:
    def __init__(self, client: AsyncOpenAI, db_conn=None):
        self.client = client
        self.db = db_conn
        self.newsapi = NewsAPIClient()
        self.twitter = TwitterClient()

    async def run(
        self,
        user_id: str = "default",
        subscriptions: dict | None = None,
    ) -> IngestReport:
        subs = subscriptions or DEFAULT_SUBSCRIPTIONS

        # 1. Fetch raw items
        source = subs.get("source", "newsapi")
        if source == "twitter":
            fetch_result = await self.twitter.fetch_items(subs)
        else:
            fetch_result = await self.newsapi.fetch_items_for_subscriptions(subs)

        items = fetch_result["items"]
        warnings = fetch_result.get("warnings", [])
        logger.info("Fetched %d items from %s", len(items), source)

        # 2. Filter items with LLM (primary) or keyword/semantic (fallback)
        interests = subs.get("interests", [])
        # Build a user_query string from the first interest label + keywords
        primary = interests[0] if interests else {}
        user_query = primary.get("label", "recent news")

        filtered_items = await filter_items(
            items, user_query, interests, client=self.client
        )
        grouped = group_by_topic(filtered_items)

        # 3. Per-item analysis + topic summarization
        topic_reports: list[TopicReport] = []

        for interest in interests:
            group = grouped.get(interest["id"], {"items": []})
            kept: list[FilteredItem] = group["items"]  # already in relevance order

            enriched = []
            for fi in kept:
                analysis = analyze_item(
                    fi.item,
                    topic_label=interest["label"],
                    keywords=interest.get("keywords", []),
                )
                enriched.append(
                    {
                        "item": fi.item,
                        "summary": analysis.summary,
                        "insights": analysis.insights,
                        "reason": fi.reason,
                    }
                )

            brief = await summarize_topic_group(
                kept, interest["label"], client=self.client
            )

            topic_reports.append(
                TopicReport(
                    topic_id=interest["id"],
                    topic_label=interest["label"],
                    matched_count=len(enriched),
                    summary=brief.get("executive_summary", ""),
                    insights=brief.get("emerging_signals", []),
                    top_items=enriched[:5],
                    all_items=enriched,
                )
            )

        report = IngestReport(
            id=_build_report_id(),
            generated_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            subscriptions=subs,
            total_fetched_items=len(items),
            warnings=warnings,
            topics=topic_reports,
            user_id=user_id,
        )

        # 4. Persist to DB
        if self.db:
            await self._persist(report)

        return report

    async def _persist(self, report: IngestReport) -> None:
        """Write run + items to DB. Implemented when DB is wired up."""
        # TODO: asyncpg upserts mirroring knowledgeStore.js mergeUserKnowledgeFromReport
        pass
