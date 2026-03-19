"""
forum/market_agent.py
MarketAgent — reads collective probability estimates from prediction markets.

Sources:
  1. Polymarket — geopolitics, crypto, current events (binary markets)
  2. Metaculus  — science, technology, policy (calibrated community forecasts)

What prediction market data adds to The Debate:
  - Crowd-aggregated probability on specific, resolvable outcomes
  - Liquidity / participation signals (how much people are willing to bet)
  - Market movement over time (are odds shifting, and why?)
  - Explicit falsifiability: markets force precise question framing

Strengths: aggregated wisdom of crowds with real financial stakes;
forces precise, falsifiable framing; captures uncertainty quantitatively.
Known bias: Polymarket skews crypto-native and English-speaking traders;
Metaculus community is technically educated but small; both miss tail
risks that have never been explicitly questioned; markets can be thin
on niche topics.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex.forum.base_agent import BaseAgent, ContentItem, AgentReport
from memex.forum.decomposer import MarketQuery
from memex.sources.polymarket import PolymarketClient
from memex.sources.metaculus import MetaculusClient


class MarketAgent(BaseAgent):
    agent_type = "market"
    agent_label = "MarketAgent"
    caveats_text = (
        "Polymarket skews crypto-native English-speaking traders; Metaculus community "
        "is small and technically educated. Both miss tail risks never explicitly "
        "questioned. Markets can be illiquid on niche topics — low volume means "
        "prices are unreliable. Prediction markets reflect expected value, not certainty."
    )

    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        super().__init__(client, model)
        self.polymarket = PolymarketClient()
        self.metaculus = MetaculusClient()

    # ── Typed interface (primary) ──────────────────────────────────────────────

    async def run_decomposed(self, query: MarketQuery) -> AgentReport:
        """Fetch prediction market data using the typed MarketQuery."""
        items = await self._fetch_markets(query)
        summary, findings = await self._produce_report(query.search_terms, items)
        return AgentReport(
            agent_type=self.agent_type,
            agent_label=self.agent_label,
            sub_queries=query.search_terms,
            items=items,
            summary=summary,
            findings=findings,
            caveats=self.caveats_text,
        )

    # ── Generic interface (backwards compat) ───────────────────────────────────

    async def fetch(self, sub_query: str) -> list[ContentItem]:
        q = MarketQuery(
            platforms=["polymarket", "metaculus"],
            search_terms=[sub_query],
            resolvable_questions=[],
        )
        return await self._fetch_markets(q)

    # ── Implementation ─────────────────────────────────────────────────────────

    async def _fetch_markets(self, query: MarketQuery) -> list[ContentItem]:
        platforms = {p.lower() for p in query.platforms}
        search_terms = query.search_terms or [""]

        tasks = []
        labels = []

        for term in search_terms[:3]:   # cap to 3 terms to avoid rate limits
            if "polymarket" in platforms:
                tasks.append(self.polymarket.search(term, limit=10))
                labels.append(("polymarket", term))
            if "metaculus" in platforms:
                tasks.append(self.metaculus.search(term, limit=10))
                labels.append(("metaculus", term))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[ContentItem] = []
        seen_ids: set[str] = set()

        for (platform, term), result in zip(labels, results):
            if isinstance(result, Exception):
                self.log.warning("%s search failed for %r: %s", platform, term, result)
                continue
            for market in result:
                uid = f"{platform}:{market.get('id', market.get('question', ''))}"
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                prob = market.get("yes_probability") or market.get("probability")
                items.append(ContentItem(
                    title=market.get("question", ""),
                    url=market.get("url", ""),
                    body=market.get("text", market.get("question", "")),
                    source=platform,
                    published_at=market.get("close_time") or market.get("end_date", ""),
                    metadata={
                        "platform": platform,
                        "probability": prob,
                        "volume_usd": market.get("volume_usd"),
                        "num_predictions": market.get("num_predictions"),
                        "resolution_criteria": market.get("resolution_criteria", ""),
                    },
                ))

        # Sort by trading signal strength: high volume (Polymarket) or many forecasters (Metaculus)
        items.sort(
            key=lambda x: x.metadata.get("volume_usd") or x.metadata.get("num_predictions") or 0,
            reverse=True,
        )
        return items[:20]
