"""
test_agentic_workflow.py
End-to-end integration test of the full Debate agentic pipeline.

Pipeline:
    User query
        ↓
    [Debate] QueryDecomposer  (GPT-4o) → 5 typed sub-queries
        ↓             ↓             ↓             ↓             ↓
    NewsAgent   SocialAgent   ExpertAgent  KnowledgeAgent  MarketAgent
    (NewsAPI    (Reddit        (Jina Search  (pgvector        (Polymarket
     + RSS       + HN)          → think       corpus)          + Metaculus)
     + Jina)                    tanks)
        ↓             ↓             ↓             ↓             ↓
    AgentReport   AgentReport   AgentReport  AgentReport   AgentReport
        ↓─────────────┴─────────────┴─────────────┴─────────────┘
    [Debate] Synthesis  (multi-round GPT-4o)
        ↓
    DebateReport  (trajectory + pressure questions + per-agent findings)

Host:       memex/forum/engine.py  →  class Debate
            Debate owns the decomposer and all 5 agents.
            base_agent.py is the shared abstract base class only.

Modular phase tests (run individually):
    pytest memex/tests/test_phase1_decomposer.py -s -v   # decomposition only
    pytest memex/tests/test_phase2_agents.py     -s -v   # individual agents

Run the full orchestrated pipeline:
    pytest memex/tests/test_agentic_workflow.py  -s -v

Run all phases together:
    pytest memex/tests/ -s -v

Or directly:
    python memex/tests/test_agentic_workflow.py

Notes:
    - Agents with missing API keys log a warning and return empty items;
      the Debate continues with whatever data is available.
    - Only OPENAI_API_KEY is strictly required.
    - Set NEWSAPI_KEY, JINA_API_KEY, REDDIT_CLIENT_ID/SECRET for richer results.
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

# ── env loading ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

# ── imports ───────────────────────────────────────────────────────────────────
from openai import AsyncOpenAI

from memex.forum.engine import Debate, DebateReport
from memex.tests.conftest import (
    TEST_QUERY,
    section,
    subsection,
    wrap,
    print_env_status,
    print_agent_report,
)


# ── debate-specific helpers ───────────────────────────────────────────────────

def _conf_bar(pct: int, width: int = 20) -> str:
    """Render a compact Unicode block bar for a confidence percentage."""
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _bullets(items: list[str], indent: int = 6, marker: str = "•") -> None:
    """Print a bullet list, each item text-wrapped at 88 chars."""
    prefix = " " * indent
    for item in items:
        lines = textwrap.wrap(item, width=88 - indent - 2)
        if not lines:
            continue
        print(f"{prefix}{marker} {lines[0]}")
        for cont in lines[1:]:
            print(f"{prefix}  {cont}")


def _print_trajectory(t: dict, idx: int, total: int) -> None:
    conf = t.get("confidence_pct", 0)
    label = t.get("label", f"Path {idx}")
    bar = _conf_bar(conf)

    # ── header ────────────────────────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print(f"  TRAJECTORY {idx} of {total}")
    print(f"  {label.upper()}")
    print(f"  {bar}  {conf}%")
    print(f"  {'─' * 68}")

    # ── narrative description ─────────────────────────────────────────────
    desc = t.get("description", "")
    if desc:
        print()
        for line in textwrap.wrap(desc, width=84, initial_indent="    ",
                                   subsequent_indent="    "):
            print(line)

    # ── FOR (supporting) ──────────────────────────────────────────────────
    supporting = t.get("supporting_evidence", [])
    if supporting:
        print(f"\n  FOR")
        _bullets(supporting)

    # ── AGAINST (countervailing) ──────────────────────────────────────────
    counter = t.get("countervailing_evidence", [])
    if counter:
        print(f"\n  AGAINST")
        _bullets(counter)

    # ── key variable + falsifiable test ──────────────────────────────────
    kv = t.get("key_variable", "")
    ft = t.get("falsifiable_test", "")
    if kv or ft:
        print()
    if kv:
        for line in textwrap.wrap(kv, width=78, initial_indent="  Key variable:    ",
                                   subsequent_indent="                   "):
            print(line)
    if ft:
        for line in textwrap.wrap(ft, width=78, initial_indent="  Watch for:       ",
                                   subsequent_indent="                   "):
            print(line)


def _print_debate_report(report: DebateReport) -> None:
    section(f"STEP 3 · DEBATE SYNTHESIS  ({len(report.rounds)} round(s))")

    for rnd in report.rounds:
        print(f"\n  {'═'*68}")
        print(f"  ROUND {rnd.round_num + 1}")
        print(f"  {'═'*68}")

        # Round verdict only — the analytical delta produced this round
        rv = rnd.round_verdict
        if rv:
            print(wrap(f"Established: {rv.get('established','')}", indent=4))
            print(wrap(f"Contested:   {rv.get('contested','')}", indent=4))
            print(f"\n    What changed this round:")
            print(wrap(rv.get("delta", "(none)"), indent=4))

    # ── Final comprehensive report ─────────────────────────────────────────
    fr = report.final_report
    if not fr:
        subsection("FINAL REPORT")
        print(wrap(report.final_trajectory or "(none)"))
        return

    section("FINAL INTELLIGENCE REPORT")

    subsection("EXECUTIVE SUMMARY")
    print(wrap(fr.executive_summary))

    if fr.validated_claims:
        subsection(f"WHAT WE KNOW  ({len(fr.validated_claims)})")
        print("  Findings that survived all rounds of scrutiny\n")
        _bullets(fr.validated_claims, indent=4)

    if fr.contested_claims:
        subsection(f"WHAT REMAINS UNCERTAIN  ({len(fr.contested_claims)})")
        _bullets(fr.contested_claims, indent=4)

    if fr.cross_agent_syntheses:
        subsection(f"KEY INSIGHTS  ({len(fr.cross_agent_syntheses)})")
        print("  Conclusions only visible by combining evidence across sources\n")
        for i, s in enumerate(fr.cross_agent_syntheses, 1):
            print(wrap(f"({i}) {s}"))

    if fr.trajectories:
        subsection(f"TRAJECTORIES  ({len(fr.trajectories)})")
        total = len(fr.trajectories)
        for idx, t in enumerate(fr.trajectories, 1):
            _print_trajectory(t, idx, total)

    if fr.critical_blindspots:
        subsection(f"CRITICAL BLINDSPOTS  ({len(fr.critical_blindspots)})")
        for i, b in enumerate(fr.critical_blindspots, 1):
            print(wrap(f"[{i}] {b}"))

    if fr.analyst_note:
        subsection("ANALYST NOTE  (watch next 30-90 days)")
        print(wrap(fr.analyst_note))


# ── test ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_debate_pipeline(openai_client):
    """
    Full end-to-end test:
      query → Debate (decompose + 5 agents + synthesis) → DebateReport

    The Debate class in forum/engine.py is the HOST. It owns:
      - QueryDecomposer  (one GPT-4o call per run)
      - All 5 agent instances
      - The multi-round synthesis loop

    base_agent.py is the shared abstract base class — NOT the host.
    """
    print_env_status()
    section("STEP 0 · PIPELINE OVERVIEW")
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  User query                                                         │
  │      ↓                                                              │
  │  [HOST: Debate / forum/engine.py]                                   │
  │      ↓                                                              │
  │  QueryDecomposer (GPT-4o) → 5 typed sub-queries                    │
  │      ↓         ↓         ↓         ↓         ↓                      │
  │  News      Social    Expert   Knowledge   Market   (parallel)       │
  │  NewsAPI   Reddit    Jina     pgvector    Polymarket                │
  │  RSS       HN        Search   corpus      Metaculus                 │
  │  Jina                                                               │
  │      ↓─────────┴─────────┴─────────┴─────────┘                     │
  │  Debate synthesis  (multi-round GPT-4o)                             │
  │      ↓                                                              │
  │  DebateReport  (trajectory + pressure Qs + per-agent findings)     │
  └─────────────────────────────────────────────────────────────────────┘
    """)

    print(f"  Query: {TEST_QUERY}\n")

    rounds = int(os.environ.get("DEBATE_ROUNDS", 2))
    print(f"  Debate rounds: {rounds}  (override with DEBATE_ROUNDS=N in .env)\n")
    debate = Debate(
        client=openai_client,
        db_conn=None,   # skip pgvector — KnowledgeAgent returns empty gracefully
        rounds=rounds,
    )

    t0 = time.perf_counter()
    report: DebateReport = await debate.run(TEST_QUERY)
    total = time.perf_counter() - t0

    section(f"STEP 2 · AGENT REPORTS  (from Debate.agent_reports)")
    for agent_type, agent_report in report.agent_reports.items():
        print_agent_report(agent_report, elapsed=total)

    _print_debate_report(report)

    section(f"PIPELINE COMPLETE  ·  Total wall-clock: {total:.1f}s")
    print(f"  Agents that returned items: "
          f"{[k for k, v in report.agent_reports.items() if v.items]}")
    total_items = sum(len(v.items) for v in report.agent_reports.values())
    print(f"  Total content items gathered: {total_items}")
    print(f"  Debate rounds completed: {len(report.rounds)}")
    print()

    assert report.query == TEST_QUERY
    assert len(report.rounds) >= 1, "at least one synthesis round expected"
    final = report.rounds[-1]
    assert final.round_verdict.get("delta"), "synthesis must produce a round delta"
    assert final.pressure_questions or final.round_verdict, "synthesis must produce structured output"
    assert "market" in report.agent_reports, "MarketAgent report must be present"


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run the full pipeline directly:
        python memex/tests/test_agentic_workflow.py
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=key)

    async def _main():
        from memex.forum.decomposer import QueryDecomposer
        from memex.tests.test_phase1_decomposer import _print_decomposed

        decomposer = QueryDecomposer(client)
        print(f"\nDecomposing: {TEST_QUERY!r} …")
        t0 = time.perf_counter()
        decomposed = await decomposer.decompose(TEST_QUERY)
        _print_decomposed(decomposed)
        print(f"\n  ✓ {time.perf_counter() - t0:.1f}s")

        rounds = int(os.environ.get("DEBATE_ROUNDS", 2))
        print(f"\nRunning Debate ({rounds} round(s)) …")
        debate = Debate(client=client, db_conn=None, rounds=rounds)
        t0 = time.perf_counter()
        report = await debate.run(TEST_QUERY)
        elapsed = time.perf_counter() - t0

        section("AGENT REPORTS")
        for agent_type, agent_report in report.agent_reports.items():
            print_agent_report(agent_report, elapsed=elapsed)

        _print_debate_report(report)
        section(f"DONE  ·  {elapsed:.1f}s")

    asyncio.run(_main())
