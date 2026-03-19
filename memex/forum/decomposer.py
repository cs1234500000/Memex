"""
forum/decomposer.py
Translates one user query into 5 typed sub-queries — one per debate agent.

Each sub-query is written in the "dialect" that agent's sources respond to best:
  - NewsAgent    → headline-optimised search string + keyword list + time window
  - SocialAgent  → specific subreddits + conversational phrasing + HN terse query
  - ExpertAgent  → research-grade formulation + named think-tank targets
  - KnowledgeAgent → abstract structural pattern (no proper nouns) + scenario tags
  - MarketAgent  → resolvable probabilistic questions for prediction markets

The Debate engine instantiates one QueryDecomposer and calls it at the start
of every run, before dispatching agents.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from memex.config import EXPERT_MODEL
from memex import llm

logger = logging.getLogger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────

class NewsQuery(BaseModel):
    search_string: str = Field(description="Short scannable English string optimised for keyword matching against headlines")
    keywords: list[str] = Field(description="4-6 individual keywords for headline filtering")
    time_window_days: int = Field(description="How many days back is relevant for this topic")
    source_hints: list[str] = Field(description="Preferred news sources or wire services", default_factory=list)


class SocialQuery(BaseModel):
    subreddits: list[str] = Field(description="3-5 specific subreddit names (without r/ prefix)")
    search_terms: list[str] = Field(description="Conversational search strings matching how real people discuss this")
    hn_query: str = Field(description="Terse technical query matching HN's engineering/founder audience")
    min_score: int = Field(description="Minimum upvote score — higher for mature topics, lower for niche/emerging")


class ExpertQuery(BaseModel):
    search_string: str = Field(description="Research-grade formulation using expert community vocabulary")
    target_sources: list[str] = Field(description="3-5 specific named sources appropriate for this domain")
    depth: str = Field(description="brief | standard | deep")
    recency_days: int = Field(description="Tight (14-30) for fast-moving topics, loose (60-90) for structural ones")


class KnowledgeQuery(BaseModel):
    abstract_pattern: str = Field(description="Abstract structural dynamic — NO proper nouns from the original query")
    scenarios: list[str] = Field(description="2-4 scenario tags from the fixed list")
    min_similarity: float = Field(description="Cosine similarity threshold (0.0-1.0) for pgvector retrieval", default=0.55)


class MarketQuery(BaseModel):
    platforms: list[str] = Field(description="Which prediction market platforms to query: polymarket, metaculus, manifold")
    search_terms: list[str] = Field(description="Terms matching how prediction markets title their questions")
    resolvable_questions: list[str] = Field(description="2-3 example resolvable (binary/numeric, time-bound) questions that plausibly exist")


class DecomposedQuery(BaseModel):
    intent: str = Field(description="Clean English summary of what the user is actually asking")
    news: NewsQuery
    social: SocialQuery
    expert: ExpertQuery
    knowledge: KnowledgeQuery
    market: MarketQuery


# ── System prompt ──────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM_PROMPT = """
You are a query decomposition engine for a multi-agent intelligence \
analysis system. Your job is to take any user query — on any topic, \
in any language — and decompose it into 5 precisely targeted \
sub-queries, one per agent.

## The 5 agents and what they need

**NewsAgent** searches news APIs and RSS feeds.
- Needs: a short, scannable English search string optimized for \
keyword matching against headlines and article abstracts.
- Think like a journalist searching an archive: concrete nouns, \
named entities, action verbs. No abstract framing.
- Include a list of 4-6 individual keywords for headline filtering.
- Set a time window: how many days back is relevant for this topic?

**SocialAgent** searches Reddit and Hacker News.
- Needs: specific subreddit targets (3-5) + a conversational search \
string that matches how real people discuss this topic, not how \
journalists write about it.
- Always include at least one meta-community (r/geopolitics, \
r/investing, r/technology, r/worldnews, r/economics, r/stocks, \
r/MachineLearning, r/singularity — pick the most relevant).
- The HN query should be terse and technical, matching HN's \
engineering/founder audience.
- Set min_score: higher (100+) for mature topics with lots of \
content, lower (25+) for niche or emerging topics.

**ExpertAgent** retrieves long-form analysis from think tanks, \
newsletters, and academic sources.
- Needs: a research-grade formulation — precise, analytical, using \
the vocabulary of the relevant expert community.
- Name 3-5 specific target sources appropriate for this topic domain.
  Examples by domain:
  - Geopolitics: RAND, Brookings, CSIS, Foreign Affairs, War on the Rocks
  - Tech/AI: a16z, Stratechery, Import AI, The Diff, Andreessen Horowitz
  - Economics/Finance: IMF blogs, Peterson Institute, The Economist, \
    Bloomberg Opinion, FT Alphaville
  - Military: Lawfare, Modern War Institute, IISS, Belfer Center
  - General: Substack ecosystem, Axios, Politico
- Set recency: tight (14-30 days) for fast-moving topics, looser \
(60-90 days) for structural/slow-moving ones.

**KnowledgeAgent** does semantic search over an offline corpus of \
classical texts: Thucydides, Clausewitz, Sun Tzu, Machiavelli, \
Gibbon, Dalio, Taleb, Kuhn, Schumpeter, Arendt, Popper.
- Needs: an ABSTRACT structural pattern — NOT a restatement of the \
surface topic. Step back from the specifics and ask: what \
fundamental dynamic is this an instance of?
- Wrong: "AI assistant subscription models" (too specific, too modern)
- Right: "Dominant platform extracting rents as ecosystem matures, \
triggering fragmentation or substitution by challengers"
- Wrong: "Trump tariffs on China" (named entities, current events)
- Right: "Rising power and established power using economic coercion \
as substitute for direct military conflict"
- Select 2-4 scenario tags from this fixed list ONLY:
  [platform_disruption, geopolitical_rivalry, market_consolidation, \
  technology_commoditization, us_china_competition, military_conflict,\
  sanctions, financial_crisis, tech_war, regime_change, \
  economic_decoupling, institutional_decay, arms_race, \
  currency_war, supply_chain_disruption]

**MarketAgent** searches prediction markets (Polymarket, Metaculus, Manifold).
- Needs: search terms that match how prediction markets title their \
questions, plus 2-3 example resolvable questions.
- Prediction market questions are always resolvable: they have a \
clear binary or numeric outcome by a specific date.
- Wrong: "What will happen with AI regulation?" (not resolvable)
- Right: "Will the EU AI Act enforcement begin before end of 2025?"
- Right: "Will Anthropic reach $5B revenue by Q4 2025?"
- Choose platforms based on topic: Polymarket skews geopolitics and \
crypto, Metaculus skews science and policy, Manifold is broadest.

## Decomposition rules

1. **Language**: Always output in English regardless of input language. \
   The intent field should be a clean English translation/summary.

2. **Specificity gradient**: NewsAgent gets the most literal/surface \
   query. KnowledgeAgent gets the most abstract. The others sit \
   between. Do not give all agents the same keywords.

3. **Domain sensitivity**: Adapt source hints and subreddits to the \
   actual domain. A query about military conflict needs different \
   subreddits than a query about startup fundraising.

4. **Scope calibration**: If the query is narrow (a single company, \
   a single event), keep time windows tight and min_score high. \
   If the query is structural (an ongoing trend, a systemic shift), \
   widen the time window and lower the score threshold.

5. **No hallucinated markets**: For MarketQuery.resolvable_questions, \
   only write questions that plausibly exist on prediction markets — \
   verifiable, time-bound, binary or numeric. Do not fabricate \
   specific URLs or market IDs.

6. **KnowledgeAgent caveat**: The abstract_pattern must not contain \
   any proper noun from the original query (no company names, person \
   names, country names, product names). If you find yourself writing \
   one, you have not abstracted enough.

## Output

Return valid JSON matching the DecomposedQuery schema exactly.
No explanation, no preamble. JSON only.
"""


# ── QueryDecomposer ────────────────────────────────────────────────────────────

class QueryDecomposer:
    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        self.client = client
        self.model = model

    async def decompose(self, query: str) -> DecomposedQuery:
        """
        Translate a user query into 5 typed sub-queries via LLM.
        Uses client.beta.chat.completions.parse() so the SDK builds a
        spec-compliant strict schema (with additionalProperties:false on
        every nested object) automatically.
        """
        try:
            parsed = await llm.parse_structured(
                self.client,
                messages=[
                    {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user",   "content": f'User query: "{query}"'},
                ],
                response_format=DecomposedQuery,
                model=self.model,
                temperature=0.3,
            )
            if parsed is not None:
                return parsed
        except Exception as exc:
            logger.warning("DecomposedQuery parse failed (%s), using minimal fallback", exc)
        return _minimal_decompose(query)


def _minimal_decompose(query: str) -> DecomposedQuery:
    """Emergency fallback — build a bare-bones decomposition from the raw query."""
    words = query.split()[:6]
    return DecomposedQuery(
        intent=query,
        news=NewsQuery(search_string=query, keywords=words, time_window_days=14, source_hints=[]),
        social=SocialQuery(subreddits=["worldnews", "technology"], search_terms=[query], hn_query=query, min_score=25),
        expert=ExpertQuery(search_string=query, target_sources=["Substack", "arXiv"], depth="standard", recency_days=30),
        knowledge=KnowledgeQuery(abstract_pattern=query, scenarios=[], min_similarity=0.5),
        market=MarketQuery(platforms=["metaculus", "manifold"], search_terms=[query], resolvable_questions=[]),
    )
