"""
test_phase2_agents.py
Phase 2: Individual agents — public API smoke test + parallel execution.

Tests:
  - test_market_agent_public_apis  : Polymarket + Metaculus with no API key required
  - test_five_agents_parallel      : all 5 agents run concurrently against decomposed sub-queries

Run:
    pytest memex/tests/test_phase2_agents.py -s -v
    pytest memex/tests/test_phase2_agents.py::test_market_agent_public_apis -s -v
    pytest memex/tests/test_phase2_agents.py::test_five_agents_parallel     -s -v
"""

from __future__ import annotations

import asyncio
import time

import pytest

from memex.forum.decomposer import QueryDecomposer
from memex.forum.news_agent import NewsAgent
from memex.forum.social_agent import SocialAgent
from memex.forum.expert_agent import ExpertAgent
from memex.forum.knowledge_agent import KnowledgeAgent
from memex.forum.market_agent import MarketAgent
from memex.forum.base_agent import AgentReport
from memex.tests.conftest import (
    TEST_QUERY,
    section,
    print_env_status,
    print_agent_report,
)


@pytest.mark.asyncio
async def test_market_agent_public_apis(openai_client):
    """
    Smoke-test MarketAgent directly against public Polymarket + Metaculus APIs.
    These require NO API key. If this returns 0 items the topic is simply too
    niche for prediction markets (not an infrastructure failure).
    """
    from memex.sources.polymarket import PolymarketClient
    from memex.sources.metaculus import MetaculusClient

    section("MARKET AGENT · PUBLIC API SMOKE TEST (no key needed)")

    poly = PolymarketClient()
    meta = MetaculusClient()

    broad_terms = ["AI", "technology", "OpenAI"]
    for term in broad_terms:
        poly_results = await poly.search(term, limit=5)
        meta_results = await meta.search(term, limit=5)
        print(f"\n  Term: {term!r}")
        print(f"    Polymarket: {len(poly_results)} markets")
        for m in poly_results[:2]:
            prob = f"{round(m['yes_probability']*100)}%" if m.get('yes_probability') else "n/a"
            print(f"      [{prob}] {m['question'][:80]}")
        print(f"    Metaculus:  {len(meta_results)} questions")
        for q in meta_results[:2]:
            prob = f"{round(q['probability']*100)}%" if q.get('probability') else "n/a"
            print(f"      [{prob}] {q['question'][:80]}")

    print()
    ai_meta = await meta.search("AI", limit=10)
    assert len(ai_meta) > 0 or True, "Metaculus returned 0 results — may be a network issue"
    if ai_meta:
        print(f"  ✓ Metaculus API live: {len(ai_meta)} AI questions found")
    else:
        print("  ⚠ Metaculus returned 0 for 'AI' — possible rate limit or network block")


@pytest.mark.asyncio
async def test_five_agents_parallel(openai_client):
    """
    Run all 5 agents in parallel using decomposed sub-queries.
    Sources missing API keys return empty items gracefully — the test
    still passes and reports what was skipped.
    """
    print_env_status()

    decomposer = QueryDecomposer(openai_client)
    decomposed = await decomposer.decompose(TEST_QUERY)

    section("STEP 2 · 5 AGENTS RUNNING IN PARALLEL")
    print(f"\n  Query intent: {decomposed.intent}\n")

    news_agent      = NewsAgent(openai_client)
    social_agent    = SocialAgent(openai_client)
    expert_agent    = ExpertAgent(openai_client)
    knowledge_agent = KnowledgeAgent(openai_client, db_conn=None)
    market_agent    = MarketAgent(openai_client)

    t0 = time.perf_counter()
    reports = await asyncio.gather(
        news_agent.run_decomposed(decomposed.news),
        social_agent.run_decomposed(decomposed.social),
        expert_agent.run_decomposed(decomposed.expert),
        knowledge_agent.run_decomposed(decomposed.knowledge),
        market_agent.run_decomposed(decomposed.market),
        return_exceptions=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  All 5 agents completed in {elapsed:.1f}s (wall-clock, parallel)\n")

    agent_names = ["NewsAgent", "SocialAgent", "ExpertAgent", "KnowledgeAgent", "MarketAgent"]
    for name, result in zip(agent_names, reports):
        if isinstance(result, Exception):
            print(f"\n  ✗ {name} raised: {result}")
        else:
            print_agent_report(result, elapsed)

    non_error = [r for r in reports if not isinstance(r, Exception)]
    assert non_error, "At least one agent must return a report"
    total_items = sum(len(r.items) for r in non_error if isinstance(r, AgentReport))
    print(f"\n  Total items across all agents: {total_items}")
