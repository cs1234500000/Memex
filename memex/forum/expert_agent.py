"""
agents/expert_agent.py
ExpertAgent — retrieves deep analysis from specialist and academic sources.

Sources:
  1. Jina Search (s.jina.ai) — targeted queries against think tanks,
     Substack, arXiv, and independent analyst sites.
  2. Jina Reader (r.jina.ai) — full-text fetch of found articles.

Target domains: RAND, Brookings, CSIS, CFR, IISS, arXiv, Substack,
                NBER, Foreign Affairs, War on the Rocks, Lawfare.

Strengths: depth, analytical frameworks, primary research, longitudinal view.
Known bias: elite consensus; underrepresents non-Western think tanks;
Substack may mix rigorous analysis with hot takes; arXiv pre-prints are unreviewed.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex.forum.base_agent import BaseAgent, ContentItem, AgentReport
from memex.forum.decomposer import ExpertQuery
from memex.sources.jina import JinaClient
from memex.ingest.enricher import is_fetchable

# Each entry: (site_operator, friendly source label)
# Grouped by type so we can parallelize without spamming one domain.
EXPERT_TARGETS: list[tuple[str, str]] = [
    # Think tanks / policy
    ("site:rand.org", "RAND"),
    ("site:brookings.edu", "Brookings"),
    ("site:csis.org", "CSIS"),
    ("site:cfr.org", "CFR"),
    ("site:iiss.org", "IISS"),
    ("site:foreignaffairs.com", "Foreign Affairs"),
    ("site:warontherocks.com", "War on the Rocks"),
    ("site:lawfaremedia.org", "Lawfare"),
    # Academic / pre-print
    ("site:arxiv.org", "arXiv"),
    ("site:nber.org", "NBER"),
    # Long-form analysis
    ("site:substack.com", "Substack"),
    ("site:stratechery.com", "Stratechery"),
]

# Max simultaneous Jina Search calls to avoid rate limits
_SEARCH_CONCURRENCY = 4
# Top N search results per target to deep-fetch
_RESULTS_PER_TARGET = 2
# Top N articles to deep-fetch in total
_DEEP_FETCH_LIMIT = 10


class ExpertAgent(BaseAgent):
    agent_type = "expert"
    agent_label = "ExpertAgent"
    caveats_text = (
        "Prone to elite consensus blind spots. Overrepresents Anglophone, Western "
        "think-tank framing. Substack pieces vary wildly in rigour. arXiv pre-prints "
        "are unreviewed. RAND/Brookings may reflect funder interests. Depth of "
        "analysis does not guarantee correctness."
    )

    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        super().__init__(client, model)
        self.jina = JinaClient()

    async def run_decomposed(self, query: ExpertQuery) -> AgentReport:
        # Build site operators from target_sources when the decomposer names specific sources
        targets = self._resolve_targets(query.target_sources)
        items = await self._fetch_expert(query.search_string, targets)
        for item in items:
            item.relevance_score = await self.score(item, query.search_string)
        summary, findings = await self._produce_report([query.search_string], items)
        return AgentReport(
            agent_type=self.agent_type, agent_label=self.agent_label,
            sub_queries=[query.search_string], items=items,
            summary=summary, findings=findings, caveats=self.caveats_text,
        )

    @staticmethod
    def _resolve_targets(source_names: list[str]) -> list[tuple[str, str]]:
        """Map human-readable source names → (site_operator, label) tuples."""
        # Build a lookup from label → operator
        label_to_op = {lbl.lower(): (op, lbl) for op, lbl in EXPERT_TARGETS}
        resolved = []
        for name in source_names:
            key = name.lower().replace(" ", "").replace(".", "")
            # Try exact and partial match
            match = label_to_op.get(key) or next(
                (v for k, v in label_to_op.items() if key in k or k in key), None
            )
            if match:
                resolved.append(match)
        # Fall back to all targets if none resolved
        return resolved or EXPERT_TARGETS

    async def fetch(self, sub_query: str) -> list[ContentItem]:
        return await self._fetch_expert(sub_query, EXPERT_TARGETS)

    async def _fetch_expert(self, sub_query: str, targets: list[tuple[str, str]]) -> list[ContentItem]:
        # Search all targets concurrently (with semaphore for rate limiting)
        sem = asyncio.Semaphore(_SEARCH_CONCURRENCY)
        search_tasks = [
            self._search_target(sub_query, site_op, label, sem)
            for site_op, label in targets
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect candidates, preserving source label
        candidates: list[tuple[dict, str]] = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                continue
            label = targets[i][1]
            for r in result[:_RESULTS_PER_TARGET]:
                candidates.append((r, label))

        # 2. Deep-fetch full text for top fetchable candidates
        fetchable = [
            (c, lbl) for c, lbl in candidates
            if is_fetchable(c.get("url", ""))[0]
        ][:_DEEP_FETCH_LIMIT]

        full_texts = await asyncio.gather(
            *[self._deep_fetch(c["url"]) for c, _ in fetchable],
            return_exceptions=True,
        )

        items: list[ContentItem] = []
        for i, (candidate, label) in enumerate(fetchable):
            body = (
                (full_texts[i][:3000] if not isinstance(full_texts[i], Exception) and full_texts[i] else None)
                or candidate.get("snippet", "")
            )
            if not body:
                continue
            items.append(ContentItem(
                title=candidate.get("title", ""),
                url=candidate.get("url", ""),
                body=body,
                source=label,
                metadata={"domain": candidate.get("domain"), "search_label": label},
            ))

        return items

    async def _search_target(
        self, query: str, site_op: str, label: str, sem: asyncio.Semaphore
    ) -> list[dict]:
        async with sem:
            try:
                return await self.jina.search(f"{query} {site_op}", num_results=3)
            except Exception as exc:
                self.log.debug("Jina search failed for %s: %s", label, exc)
                return []

    async def _deep_fetch(self, url: str) -> str | None:
        try:
            result = await self.jina.read(url)
            if len(result.content) > 300:
                return result.content
        except Exception:
            pass
        return None
