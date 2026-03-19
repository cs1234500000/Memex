"""
agents/base_agent.py
Shared base for all research agents (NewsAgent, SocialAgent, ExpertAgent, KnowledgeAgent).

Each agent follows a reflection loop:
  1. Decompose the query into targeted sub-queries
  2. Fetch raw content for each sub-query (agent-specific sources)
  3. Score relevance; if too low, refine the sub-query and retry
  4. Produce a structured AgentReport consumed by The Debate

AgentReport shape is intentionally richer than a plain summary so The Debate
can do claim-level auditing, not just paragraph-level comparison.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from memex.config import EXPERT_MODEL
from memex import llm

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RELEVANCE_THRESHOLD = 0.6


# ── Structured output schemas (Pydantic) ───────────────────────────────────────

class Finding(BaseModel):
    claim: str
    source_title: str
    url: str
    confidence: Literal["high", "medium", "low"]


class AgentOutput(BaseModel):
    """Schema the LLM must return from _produce_report."""
    summary: str
    findings: list[Finding]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ContentItem:
    title: str
    url: str
    body: str
    source: str
    published_at: str = ""
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentReport:
    """Structured output produced by each research agent for The Debate."""
    agent_type: str
    agent_label: str               # human-readable (e.g. "NewsAgent")
    sub_queries: list[str]
    items: list[ContentItem]
    # Narrative summary of findings (what this agent found)
    summary: str = ""
    # Specific factual claims extracted from items, with attribution
    findings: list[dict] = field(default_factory=list)
    # Known biases / blind spots of this agent's source set
    caveats: str = ""


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Subclasses must implement `fetch()`.
    Optionally override `caveats_text` (class-level string) to describe
    the known biases of the source set.
    """

    agent_type: str = "base"
    agent_label: str = "BaseAgent"
    caveats_text: str = "No specific known biases documented."

    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        self.client = client
        self.model = model
        self.log = logging.getLogger(f"memex.forum.{self.agent_type}")

    # ── Public interface ───────────────────────────────────────────────────────

    async def run_decomposed(self, agent_query: Any) -> AgentReport:
        """
        Primary interface used by the Debate engine.
        Receives the typed sub-query for this agent (e.g. NewsQuery, SocialQuery).
        Subclasses must override this; the default extracts a search string and
        falls back to run() for backwards compatibility.
        """
        # Extract a search string from whatever typed query we received
        search = getattr(agent_query, "search_string", None) or getattr(agent_query, "abstract_pattern", None)
        if search:
            return await self.run([search])
        return await self.run([str(agent_query)])

    async def run(self, sub_queries: list[str]) -> AgentReport:
        """Execute the reflection loop for all sub-queries; return merged report."""
        all_items: list[ContentItem] = []
        for sq in sub_queries:
            items = await self._fetch_with_reflection(sq)
            all_items.extend(items)

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[ContentItem] = []
        for it in all_items:
            if it.url not in seen:
                seen.add(it.url)
                unique.append(it)

        summary, findings = await self._produce_report(sub_queries, unique)
        return AgentReport(
            agent_type=self.agent_type,
            agent_label=self.agent_label,
            sub_queries=sub_queries,
            items=unique,
            summary=summary,
            findings=findings,
            caveats=self.caveats_text,
        )

    # ── Abstract hook ──────────────────────────────────────────────────────────

    @abstractmethod
    async def fetch(self, sub_query: str) -> list[ContentItem]:
        """Retrieve raw content items for one sub-query from agent-specific sources."""
        ...

    # ── Overridable hooks ──────────────────────────────────────────────────────

    async def score(self, item: ContentItem, sub_query: str) -> float:
        """Rate how relevant an item is to the sub-query (0.0–1.0)."""
        prompt = (
            f'Rate relevance to "{sub_query}" on a scale 0.0–1.0. '
            f"Reply with ONLY a float.\n\nTitle: {item.title}\nExcerpt: {item.body[:400]}"
        )
        text = await llm.complete(
            self.client,
            [{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0,
            max_tokens=10,
        )
        try:
            return float(text.strip())
        except ValueError:
            return 0.5

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _fetch_with_reflection(
        self, sub_query: str, attempt: int = 0
    ) -> list[ContentItem]:
        items = await self.fetch(sub_query)
        if not items:
            self.log.warning("No items for %r (attempt %d)", sub_query, attempt)
            return []

        for item in items:
            item.relevance_score = await self.score(item, sub_query)

        good = [i for i in items if i.relevance_score >= RELEVANCE_THRESHOLD]

        if not good and attempt < MAX_RETRIES:
            refined = await self._refine_query(sub_query)
            self.log.info("Reflecting: %r → %r", sub_query, refined)
            return await self._fetch_with_reflection(refined, attempt + 1)

        return good or items  # return all if none pass threshold after retries

    async def _refine_query(self, sub_query: str) -> str:
        text = await llm.complete(
            self.client,
            [{
                "role": "user",
                "content": (
                    f'The query "{sub_query}" returned low-quality results. '
                    "Rewrite it to be more specific and likely to find high-quality content. "
                    "Reply with ONLY the new query string."
                ),
            }],
            model=self.model,
            temperature=0.4,
            max_tokens=80,
        )
        return text.strip().strip('"')

    async def _produce_report(
        self, sub_queries: list[str], items: list[ContentItem]
    ) -> tuple[str, list[dict]]:
        """
        Ask the LLM to produce a structured AgentOutput (summary + findings).
        Uses client.beta.chat.completions.parse() for type-safe structured output.
        """
        if not items:
            return f"[{self.agent_label}] No relevant content found.", []

        bullets = "\n".join(
            f"- [{i.source}] {i.title} ({i.published_at or 'n/d'}): {i.body[:300]}"
            for i in items[:20]
        )

        try:
            output = await llm.parse_structured(
                self.client,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are the {self.agent_label}. "
                            f"Known bias: {self.caveats_text}\n\n"
                            "Produce a structured report with:\n"
                            "  summary: 3-5 sentence narrative of what you found\n"
                            "  findings: up to 8 specific factual claims with attribution\n"
                            "Be concrete. Name entities. Include dates when available. "
                            "Flag speculation vs confirmed fact. "
                            'confidence must be exactly "high", "medium", or "low".'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Queries: {sub_queries}\n\nContent:\n{bullets}",
                    },
                ],
                response_format=AgentOutput,
                model=self.model,
                temperature=0.3,
            )
            if output:
                return output.summary, [f.model_dump() for f in output.findings]
        except Exception as exc:
            self.log.warning("_produce_report failed: %s", exc)

        return f"[{self.agent_label}] Report generation failed.", []
