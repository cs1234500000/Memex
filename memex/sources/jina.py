"""
jina.py
Wrappers for two Jina AI services:
  - r.jina.ai  → Jina Reader: converts any URL to clean markdown
  - s.jina.ai  → Jina Search: web search returning structured results
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


JINA_READER_BASE = "https://r.jina.ai/"
JINA_SEARCH_BASE = "https://s.jina.ai/"


@dataclass
class ReaderResult:
    url: str
    title: str
    content: str


class JinaClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self._headers = {
            "Accept": "application/json",
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
        }

    async def read(self, url: str) -> ReaderResult:
        """
        Fetch and convert a web page to clean markdown via Jina Reader.
        Returns a ReaderResult with title + full markdown body.
        """
        reader_url = JINA_READER_BASE + url
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(reader_url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()

        return ReaderResult(
            url=url,
            title=data.get("data", {}).get("title", ""),
            content=data.get("data", {}).get("content", ""),
        )

    async def search(self, query: str, num_results: int = 5) -> list[dict]:
        """
        Run a web search via Jina Search and return structured result dicts.
        Each dict has: title, url, snippet, domain.
        """
        search_url = JINA_SEARCH_BASE + query
        params = {"count": num_results}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(search_url, headers=self._headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("data", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "domain": item.get("url", "").split("/")[2] if item.get("url") else "",
                }
            )
        return results
