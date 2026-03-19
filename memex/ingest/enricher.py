"""
ingest/enricher.py

Enriches raw source items with full article text fetched via Jina Reader.

Role in the pipeline:
  sources (fetch) → enricher (full text) → filter (LLM/semantic) → store (persist)

The enricher is intentionally optional: items that can't be fetched (paywalls,
bot-blocks, social/discussion URLs) are returned with empty `full_text` and a
`fetch_status` explaining why, so the pipeline can still score them on their
title/snippet alone.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Domains whose pages are discussion threads / social feeds rather than
# articles. Jina Reader would return comment dumps, not article text.
DISCUSSION_DOMAINS: frozenset[str] = frozenset(
    {
        "news.ycombinator.com",  # HN item pages (comments)
        "reddit.com",
        "www.reddit.com",
        "twitter.com",
        "x.com",
        "mobile.twitter.com",
        "linkedin.com",
        "www.linkedin.com",
        "facebook.com",
        "www.facebook.com",
        "t.me",  # Telegram
    }
)

# Known paywall / login-wall / error signals in page title + first 500 chars
_ERROR_SIGNALS = [
    "page not found",
    "404 not found",
    "access denied",
    "403 forbidden",
    "just a moment",          # Cloudflare challenge
    "enable javascript",
    "are you a robot",
    "please enable cookies",
    "subscribe to read",
    "sign in to read",
    "log in to read",
]


def _domain(url: str) -> str:
    """Extract the netloc from a URL without importing urllib."""
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1).lower() if m else ""


def is_fetchable(url: str) -> tuple[bool, str]:
    """
    Return (True, "") if the URL is worth fetching as an article,
    or (False, reason) if it should be skipped.
    """
    if not url or not url.startswith(("http://", "https://")):
        return False, "no url"
    dom = _domain(url)
    if dom in DISCUSSION_DOMAINS:
        return False, f"discussion platform ({dom})"
    return True, ""


def is_error_page(title: str, content: str) -> bool:
    """Return True if the fetched page looks like a 404 / paywall / bot-block."""
    combined = (title + " " + content[:500]).lower()
    return any(s in combined for s in _ERROR_SIGNALS)


async def enrich_items(
    items: list[dict[str, Any]],
    *,
    concurrency: int = 5,
    min_content_chars: int = 300,
) -> list[dict[str, Any]]:
    """
    Fetch full article text for each item and add it as ``full_text``.

    Items that cannot be fetched keep their original fields and get:
      - ``full_text``   → "" (empty)
      - ``fetch_status`` → human-readable reason

    Args:
        items: Raw dicts from any source (must have a "url" key).
        concurrency: Max simultaneous Jina Reader requests.
        min_content_chars: Minimum chars to consider a fetch successful.

    Returns:
        Same list with ``full_text`` and ``fetch_status`` added in-place.
    """
    from memex.sources.jina import JinaClient  # local import avoids circular

    client = JinaClient()
    sem = asyncio.Semaphore(concurrency)

    async def _fetch_one(item: dict[str, Any]) -> dict[str, Any]:
        url = item.get("url", "")
        ok, reason = is_fetchable(url)
        if not ok:
            return {**item, "full_text": "", "fetch_status": reason}

        async with sem:
            try:
                result = await client.read(url)
            except Exception as exc:
                logger.debug("Jina Reader failed for %s: %s", url, exc)
                return {**item, "full_text": "", "fetch_status": f"fetch error: {exc}"}

        if is_error_page(result.title, result.content):
            return {**item, "full_text": "", "fetch_status": "error/paywall page"}

        if len(result.content) < min_content_chars:
            return {**item, "full_text": "", "fetch_status": "content too short"}

        return {
            **item,
            "full_text": result.content,
            "fetch_status": "ok",
            # Prefer the fetched title when the source had none
            "title": item.get("title") or result.title,
        }

    enriched = await asyncio.gather(*[_fetch_one(it) for it in items])
    return list(enriched)
