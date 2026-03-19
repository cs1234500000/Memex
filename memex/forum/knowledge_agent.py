"""
agents/knowledge_agent.py
KnowledgeAgent — surfaces historical precedents and analytical frameworks
from the offline curated corpus stored in pgvector.

Corpus includes:
  Public domain (Gutenberg): Thucydides, Clausewitz, Sun Tzu, Machiavelli, Gibbon
  Curated modern:            Dalio (Principles), Taleb (Antifragile), Kuhn (Structure)

At query time:
  1. Embed the sub-query + extract scenario keywords
  2. pgvector similarity search (cosine) over concept chunks
  3. Filter: only return chunks whose stored scenario tags overlap with the query
  4. Challenge step: flag analogies that feel rhetorical (high similarity, low scenario overlap)

Strengths: pattern recognition, long-horizon perspective, structural frameworks.
Known bias: corpus is overwhelmingly Western and pre-digital. Analogies can be
superficially appealing but structurally weak — The Debate is expected to
challenge any analogy that lacks mechanistic similarity.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex.forum.base_agent import BaseAgent, ContentItem, AgentReport
from memex.forum.decomposer import KnowledgeQuery
from memex.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

# Scenario keywords that chunk metadata may contain.
# Used to filter chunks that are relevant to the current situation type.
SCENARIO_TYPES = [
    "war", "conflict", "uprising", "revolution", "collapse", "pandemic",
    "economic crisis", "technological disruption", "geopolitical shift",
    "election", "assassination", "coup", "trade war", "sanctions",
    "market crash", "innovation", "empire decline", "alliance breakdown",
]


def _extract_scenario_hints(query: str) -> list[str]:
    """Return scenario types that appear in the query string."""
    q = query.lower()
    return [s for s in SCENARIO_TYPES if s in q]


class KnowledgeAgent(BaseAgent):
    agent_type = "knowledge"
    agent_label = "KnowledgeAgent"
    caveats_text = (
        "Corpus is overwhelmingly Western, pre-digital, and male-authored. Historical "
        "analogies can be structurally shallow — surface similarity ≠ causal similarity. "
        "Must be challenged when analogies feel rhetorical rather than mechanistic. "
        "Dalio/Taleb modern works are copyrighted; only conceptual summaries stored."
    )

    def __init__(self, client: AsyncOpenAI, db_conn=None, model: str = EXPERT_MODEL):
        super().__init__(client, model)
        self.store = KnowledgeStore(db_conn, client)

    async def run_decomposed(self, query: KnowledgeQuery) -> AgentReport:
        items = await self._fetch_knowledge(
            pattern=query.abstract_pattern,
            scenarios=query.scenarios,
            min_similarity=query.min_similarity,
        )
        for item in items:
            item.relevance_score = await self.score(item, query.abstract_pattern)
        summary, findings = await self._produce_report([query.abstract_pattern], items)
        return AgentReport(
            agent_type=self.agent_type, agent_label=self.agent_label,
            sub_queries=[query.abstract_pattern], items=items,
            summary=summary, findings=findings, caveats=self.caveats_text,
        )

    async def fetch(self, sub_query: str) -> list[ContentItem]:
        return await self._fetch_knowledge(
            pattern=sub_query,
            scenarios=_extract_scenario_hints(sub_query),
            min_similarity=0.55,
        )

    async def _fetch_knowledge(
        self, pattern: str, scenarios: list[str], min_similarity: float
    ) -> list[ContentItem]:
        # Retrieve candidate chunks via semantic search
        scenario_hints = scenarios or _extract_scenario_hints(pattern)
        try:
            chunks = await self.store.similarity_search(pattern, top_k=15)
        except Exception as exc:
            logger.warning("Knowledge store similarity search failed: %s", exc)
            return []

        items: list[ContentItem] = []
        for chunk in chunks:
            chunk_scenarios = chunk.get("scenarios") or []
            if isinstance(chunk_scenarios, str):
                chunk_scenarios = [chunk_scenarios]

            similarity = float(chunk.get("similarity", 0.0))
            if similarity < min_similarity:
                continue

            # Analogy quality check: high semantic similarity + zero scenario overlap
            # → likely rhetorical, not structural. Flag it.
            scenario_overlap = (
                bool(set(scenario_hints) & set(chunk_scenarios))
                if scenario_hints else True
            )
            analogy_flag = (
                "⚠ ANALOGY CAUTION: high semantic similarity but no scenario-type overlap — "
                "verify this analogy is structurally sound, not merely rhetorical."
                if similarity > 0.75 and not scenario_overlap else ""
            )

            body = chunk.get("text", "")
            if analogy_flag:
                body = f"{analogy_flag}\n\n{body}"

            items.append(ContentItem(
                title=chunk.get("title", "Knowledge Chunk"),
                url=chunk.get("source_url", ""),
                body=body[:2000],
                source=f"knowledge:{chunk.get('concept', 'corpus')}",
                metadata={
                    "concept": chunk.get("concept"),
                    "author": chunk.get("author"),
                    "work": chunk.get("work"),
                    "scenarios": chunk_scenarios,
                    "similarity": similarity,
                    "analogy_flagged": bool(analogy_flag),
                },
            ))

        # Sort by similarity descending
        items.sort(key=lambda x: x.metadata.get("similarity", 0.0), reverse=True)
        return items[:10]
