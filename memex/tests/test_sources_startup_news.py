"""
tests/test_sources_startup_news.py

Live integration test — hits real APIs to fetch startup news.

IMPORTANT — two-layer architecture:
  Layer 1 (tested here): Raw source fetch — NewsAPI/HN/Jina return whatever
                         matches their keyword query. No relevance filtering.
  Layer 2 (pipeline):    memex/ingest/filter.py passes items through an LLM
                         filter (gpt-4o-mini) or semantic/keyword fallback.
                         Off-topic results you see below (Packers article,
                         TSA story) would be dropped there before reaching the DB.

Run:
    cd /Users/shuchen/Projects/Memex
    .venv/bin/python -m pytest memex/tests/test_sources_startup_news.py -v -s --asyncio-mode=auto

Requires in .env:
    NEWSAPI_KEY / NEWSAPI_API_KEY   → NewsAPI
    JINA_API_KEY                    → Jina Search (s.jina.ai requires auth)
    HackerNews                      → no key needed
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

QUERY = "AI startup funding"

STARTUP_INTEREST = {
    "id": "ai_startups",
    "label": "AI startup companies",
    # Use multi-word phrases to avoid false positives from single tokens
    # like "agent" (free agent) or "funding" (TSA funding).
    # Semantic embeddings (NEWS_USE_LOCAL_EMBEDDINGS=true) handle single words
    # correctly; keyword-only fallback needs explicit phrases.
    "keywords": [
        "ai startup", "seed round", "series a", "series b",
        "ai agent", "llm startup", "language model", "ai funding",
        "foundation model", "enterprise ai", "generative ai",
    ],
}


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _print_items(label: str, items: list, title_key="title", url_key="url", limit=5):
    print(f"\n{'─'*60}")
    print(f"  {label}  ({len(items)} results)")
    print(f"{'─'*60}")
    for i, item in enumerate(items[:limit], 1):
        title = item.get(title_key) or item.get("story_title") or item.get("text", "")[:80]
        url = item.get(url_key) or ""
        print(f"  {i}. {title}")
        if url:
            print(f"     {url}")


# ------------------------------------------------------------------ #
# NewsAPI                                                              #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_newsapi_startup_news():
    from memex.sources.newsapi import NewsAPIClient

    key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWSAPI_API_KEY", "")
    if not key:
        pytest.skip("NEWSAPI_KEY not set")

    client = NewsAPIClient(api_key=key)
    articles = await client.search(QUERY, page_size=5, days_back=7)

    _print_items("NewsAPI  [raw — no filter applied]", articles)

    assert isinstance(articles, list)
    assert len(articles) > 0, "Got zero articles — check API key or quota"
    assert articles[0].get("title")


# ------------------------------------------------------------------ #
# Hacker News                                                          #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_hackernews_startup_news():
    from memex.sources.hackernews import HackerNewsClient

    client = HackerNewsClient()
    stories = await client.search(QUERY, limit=5, days_back=30)

    _print_items("HackerNews  [raw — no filter applied]", stories)

    assert isinstance(stories, list)
    assert len(stories) > 0, "Got zero stories"
    assert stories[0].get("title") or stories[0].get("story_title")


# ------------------------------------------------------------------ #
# Jina Search                                                          #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_jina_search_startup_news():
    from memex.sources.jina import JinaClient

    if not os.environ.get("JINA_API_KEY"):
        pytest.skip("JINA_API_KEY not set — s.jina.ai requires auth")

    client = JinaClient()
    results = await client.search(QUERY, num_results=5)

    _print_items("Jina Search  [raw — no filter applied]", results)

    assert isinstance(results, list)
    assert len(results) > 0, "Got zero results"
    assert results[0].get("title") or results[0].get("url")


# ------------------------------------------------------------------ #
# Enricher — full-text fetch via memex/ingest/enricher.py            #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_enricher_full_text():
    """
    The enricher (memex/ingest/enricher.py) is the proper place for URL→full text.
    It handles:
      - skipping discussion/social URLs (HN item pages, Reddit, Twitter)
      - detecting error/paywall pages
      - fetching full article text via Jina Reader

    This test pulls live HN stories and runs them through the enricher,
    verifying at least one article was successfully fetched.
    """
    from memex.sources.hackernews import HackerNewsClient
    from memex.ingest.enricher import enrich_items, is_fetchable

    hn = HackerNewsClient()
    stories = await hn.search(QUERY, limit=10, days_back=30)
    enriched = await enrich_items(stories, concurrency=3)

    print(f"\n{'─'*60}")
    print(f"  Enricher results  ({len(enriched)} items)")
    print(f"{'─'*60}")
    for item in enriched:
        status = item.get("fetch_status", "?")
        chars = len(item.get("full_text", ""))
        title = item.get("title") or item.get("story_title", "")
        url = item.get("url", "")
        fetchable, reason = is_fetchable(url)
        print(f"  [{status:20s}] {chars:6d} chars  {title[:60]}")
        if not fetchable:
            print(f"    → skipped: {reason}")

    successful = [it for it in enriched if it.get("fetch_status") == "ok"]
    assert len(successful) > 0, (
        "Enricher fetched zero articles successfully — check Jina API key / network"
    )
    # Verify the fetched content is real article text, not an error page
    for item in successful[:3]:
        assert len(item["full_text"]) >= 300, f"Full text too short for {item.get('url')}"


# ------------------------------------------------------------------ #
# Relevance filter — shows what gets KEPT after scoring               #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_filter_on_newsapi():
    """
    Demonstrates the two-layer design:
      - Raw fetch (NewsAPI) returns broad results
      - filter.filter_by_interest() keeps only on-topic ones

    NOTE: sentence-transformers model is downloaded on first run (~90 MB).
    Set NEWS_USE_LOCAL_EMBEDDINGS=false to use keyword-only fallback instead.
    """
    from memex.sources.newsapi import NewsAPIClient
    from memex.ingest.filter import filter_by_interest

    key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWSAPI_API_KEY", "")
    if not key:
        pytest.skip("NEWSAPI_KEY not set")

    client = NewsAPIClient(api_key=key)
    subs = {
        "interests": [STARTUP_INTEREST],
        "sources": {"newsapi": {"language": "en", "sort_by": "publishedAt"}},
        "max_items_per_run": 20,
    }
    result = await client.fetch_items_for_subscriptions(subs)
    raw_items = result["items"]

    kept = filter_by_interest(raw_items, [STARTUP_INTEREST])

    print(f"\n{'─'*60}")
    print(f"  Filter: {len(raw_items)} raw → {len(kept)} kept")
    print(f"{'─'*60}")
    for i, fi in enumerate(kept[:8], 1):
        print(f"  {i}. {fi.item.get('text','')[:80]}")
        print(f"       {fi.item.get('url','')}")

    assert isinstance(kept, list)
    if kept:
        top_text = kept[0].item.get("text", "").lower()
        print(f"\n  Top result: {top_text[:120]}")


# ------------------------------------------------------------------ #
# All sources combined (raw)                                          #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_all_sources_combined():
    """Fetch from all available sources in parallel. Raw results, no filter."""
    from memex.sources.newsapi import NewsAPIClient
    from memex.sources.hackernews import HackerNewsClient
    from memex.sources.jina import JinaClient

    key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWSAPI_API_KEY", "")

    tasks: dict[str, object] = {
        "HackerNews": HackerNewsClient().search(QUERY, limit=5),
    }
    if os.environ.get("JINA_API_KEY"):
        tasks["Jina Search"] = JinaClient().search(QUERY, num_results=5)
    if key:
        tasks["NewsAPI"] = NewsAPIClient(api_key=key).search(QUERY, page_size=5)

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    total = 0
    for source, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"\n  ✗ {source}: {result}")
        else:
            _print_items(f"{source}  [raw]", result)
            total += len(result)

    assert total > 0, "All sources returned zero results"
