"""
sources/rss.py
Async RSS/Atom feed reader. Returns a flat list of item dicts.

No external deps beyond httpx (already in requirements). Parses both RSS 2.0
and Atom 1.0 with stdlib xml.etree.ElementTree.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Major wire/broadcast feeds with reliable RSS
DEFAULT_FEEDS: dict[str, str] = {
    "reuters":  "https://feeds.reuters.com/reuters/topNews",
    "ap":       "https://feeds.apnews.com/ApNewsAlert",
    "bbc":      "http://feeds.bbci.co.uk/news/world/rss.xml",
    "ft":       "https://www.ft.com/rss/home",
    "guardian": "https://www.theguardian.com/world/rss",
}

# Atom namespace
_ATOM = "http://www.w3.org/2005/Atom"


def _parse_rss(text: str, source: str) -> list[dict[str, Any]]:
    """Parse RSS 2.0 XML into a list of item dicts."""
    items = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        logger.warning("RSS parse error (%s): %s", source, exc)
        return []

    # Handle both RSS 2.0 (<item>) and Atom (<entry>)
    is_atom = root.tag == f"{{{_ATOM}}}feed" or "atom" in root.tag.lower()

    if is_atom:
        ns = {"a": _ATOM}
        entries = root.findall("a:entry", ns)
        for e in entries:
            title = (e.findtext("a:title", namespaces=ns) or "").strip()
            link_el = e.find("a:link", ns)
            url = (link_el.get("href") if link_el is not None else "") or ""
            summary = (e.findtext("a:summary", namespaces=ns) or "").strip()
            published = (e.findtext("a:published", namespaces=ns) or "").strip()
            items.append({
                "title": title, "url": url, "text": summary,
                "published_at": published, "source": source,
            })
    else:
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            description = (item.findtext("description") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            items.append({
                "title": title, "url": url, "text": description,
                "published_at": pub_date, "source": source,
            })

    return items


class RSSClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    async def fetch_feed(self, url: str, source_name: str) -> list[dict[str, Any]]:
        """Fetch and parse a single RSS/Atom feed URL."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "memex-research-bot/1.0 (+https://withmemex.com)"},
                )
                resp.raise_for_status()
            return _parse_rss(resp.text, source_name)
        except Exception as exc:
            logger.warning("RSS fetch failed for %s (%s): %s", source_name, url, exc)
            return []

    async def search(
        self,
        query: str,
        feeds: dict[str, str] | None = None,
        limit_per_feed: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Fetch all feeds, filter items whose title or description contains any
        query word, return up to limit_per_feed items per feed.
        """
        import asyncio
        resolved = feeds or DEFAULT_FEEDS
        results = await asyncio.gather(
            *[self.fetch_feed(url, name) for name, url in resolved.items()],
            return_exceptions=True,
        )

        query_words = {w.lower() for w in query.split() if len(w) > 3}
        matched: list[dict[str, Any]] = []
        for batch in results:
            if isinstance(batch, Exception):
                continue
            for item in batch:
                text = (item["title"] + " " + item["text"]).lower()
                if any(w in text for w in query_words):
                    matched.append(item)

        # Sort by feed order (already recent-first from RSS) and cap total
        return matched[:limit_per_feed * len(resolved)]
