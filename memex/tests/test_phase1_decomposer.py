"""
test_phase1_decomposer.py
Phase 1: Query decomposition via QueryDecomposer.

Tests that the decomposer produces a valid, typed DecomposedQuery with all
required sub-query fields populated — before any agent runs.

Run:
    pytest memex/tests/test_phase1_decomposer.py -s -v
"""

from __future__ import annotations

import time

import pytest

from memex.forum.decomposer import QueryDecomposer, DecomposedQuery
from memex.tests.conftest import TEST_QUERY, section, subsection


def _print_decomposed(d: DecomposedQuery) -> None:
    section("STEP 1 · QUERY DECOMPOSITION  (host: QueryDecomposer via Debate)")
    print(f"\n  Original query: {TEST_QUERY}\n")
    print(f"  Detected intent: {d.intent}\n")

    subsection("NewsAgent sub-query")
    print(f"  search:   {d.news.search_string}")
    print(f"  keywords: {d.news.keywords}")
    print(f"  window:   {d.news.time_window_days} days")

    subsection("SocialAgent sub-query")
    print(f"  subreddits:  {d.social.subreddits}")
    print(f"  search:      {d.social.search_terms}")
    print(f"  HN query:    {d.social.hn_query}")
    print(f"  min_score:   {d.social.min_score}")

    subsection("ExpertAgent sub-query")
    print(f"  search:   {d.expert.search_string}")
    print(f"  sources:  {d.expert.target_sources}")
    print(f"  recency:  {d.expert.recency_days} days")

    subsection("KnowledgeAgent sub-query")
    print(f"  abstract pattern: {d.knowledge.abstract_pattern}")
    print(f"  scenarios:        {d.knowledge.scenarios}")
    print(f"  min_similarity:   {d.knowledge.min_similarity}")

    subsection("MarketAgent sub-query")
    print(f"  platforms:   {d.market.platforms}")
    print(f"  search:      {d.market.search_terms}")
    print(f"  questions:   {d.market.resolvable_questions}")


@pytest.mark.asyncio
async def test_decomposer(openai_client):
    """
    Verify QueryDecomposer produces a valid typed DecomposedQuery.
    This is the first step every Debate run starts with.
    """
    decomposer = QueryDecomposer(openai_client)
    t0 = time.perf_counter()
    decomposed = await decomposer.decompose(TEST_QUERY)
    elapsed = time.perf_counter() - t0

    _print_decomposed(decomposed)
    print(f"\n  ✓ Decomposition completed in {elapsed:.1f}s")

    assert decomposed.intent, "intent should not be empty"
    assert len(decomposed.news.keywords) >= 2, "need at least 2 news keywords"
    assert decomposed.social.subreddits, "need at least one subreddit"
    assert decomposed.expert.target_sources, "need at least one expert source"
    assert decomposed.knowledge.abstract_pattern, "need an abstract pattern"
    assert not any(
        noun in decomposed.knowledge.abstract_pattern.lower()
        for noun in ["openai", "anthropic", "google", "microsoft"]
    ), "abstract_pattern must not contain proper nouns from the query"
    assert decomposed.market.resolvable_questions, "need at least one market question"
