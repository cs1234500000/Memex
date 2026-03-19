"""
conftest.py
Shared fixtures, constants, and print helpers for all agentic-workflow tests.

Run any phase in isolation:
    pytest memex/tests/test_phase1_decomposer.py -s -v
    pytest memex/tests/test_phase2_agents.py     -s -v
    pytest memex/tests/test_agentic_workflow.py  -s -v   # full orchestrated

Run everything:
    pytest memex/tests/ -s -v
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

# ── path bootstrap ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from openai import AsyncOpenAI

from memex.forum.base_agent import AgentReport


# ── shared test query ─────────────────────────────────────────────────────────

TEST_QUERY = (
    "What is OpenClaw's effect on the AI agents ecosystem? "
    "Analyze its advantages and disadvantages for developers and incumbents."
)

# ── Agent → required env var mapping (for diagnostics) ───────────────────────

AGENT_KEYS: dict[str, list[str]] = {
    "NewsAgent":      ["NEWSAPI_KEY", "JINA_API_KEY"],
    "SocialAgent":    ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"],
    "ExpertAgent":    ["JINA_API_KEY"],
    "KnowledgeAgent": ["POSTGRES_DSN"],
    "MarketAgent":    [],
}


# ── print helpers ─────────────────────────────────────────────────────────────

def hr(char: str = "─", width: int = 72) -> str:
    return char * width


def section(title: str) -> None:
    print(f"\n{hr('═')}")
    print(f"  {title}")
    print(hr('═'))


def subsection(title: str) -> None:
    print(f"\n{hr('─')}")
    print(f"  {title}")
    print(hr('─'))


def wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=88, initial_indent=prefix, subsequent_indent=prefix)


def print_env_status() -> None:
    """Show which API keys are configured before running agents."""
    section("PRE-FLIGHT · API KEY STATUS")
    all_keys = {
        "OPENAI_API_KEY":          "OpenAI (required)",
        "NEWSAPI_KEY":             "NewsAPI.org → NewsAgent",
        "JINA_API_KEY":            "Jina Search/Reader → NewsAgent + ExpertAgent",
        "REDDIT_CLIENT_ID":        "Reddit PRAW → SocialAgent",
        "REDDIT_CLIENT_SECRET":    "Reddit PRAW → SocialAgent",
        "REDDIT_USER_AGENT":       "Reddit PRAW → SocialAgent",
        "POSTGRES_DSN":            "pgvector → KnowledgeAgent",
        "METACULUS_API_TOKEN":     "Metaculus (optional, higher rate limit)",
    }
    for key, desc in all_keys.items():
        val = os.environ.get(key, "")
        status = "✓ set" if val else "✗ not set"
        print(f"  {status:10s}  {key:<28s}  {desc}")
    print()
    print("  MarketAgent uses Polymarket + Metaculus public APIs — no key required.")
    print()


def print_agent_report(report: AgentReport, elapsed: float) -> None:
    subsection(f"{report.agent_label}  ({elapsed:.1f}s  ·  {len(report.items)} items)")

    if report.sub_queries:
        print(f"  sub-queries: {report.sub_queries}")

    if report.summary:
        print(wrap(report.summary))

    if report.findings:
        print(f"\n  Findings ({len(report.findings)}):")
        for f in report.findings:
            conf = f.get("confidence", "?").upper()
            claim = f.get("claim", "")
            src = f.get("source_title", "")
            url = f.get("url", "")
            print(f"    [{conf}] {claim}")
            if src:
                print(f"           ↳ {src}")
            if url:
                print(f"             {url}")

    if not report.items:
        required_keys = AGENT_KEYS.get(report.agent_label, [])
        missing = [k for k in required_keys if not os.environ.get(k)]
        if report.agent_label == "MarketAgent":
            print("  ⚠  Public APIs returned 0 markets — search terms likely too niche/technical")
            print("     for Polymarket (geopolitics/crypto) and Metaculus (science/policy).")
        elif missing:
            print(f"  ✗  Missing env vars → {', '.join(missing)}")
            print(f"     Add them to .env to enable {report.agent_label}.")
        elif report.agent_label == "KnowledgeAgent":
            print("  ⚠  db_conn=None in this test — KnowledgeAgent requires POSTGRES_DSN + pgvector.")
        else:
            print("  ⚠  No items returned (APIs responded but matched nothing, or a fetch error).")


# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def openai_client():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — cannot run agentic workflow test")
    return AsyncOpenAI(api_key=key)
