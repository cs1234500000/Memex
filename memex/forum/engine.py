"""
forum/engine.py
The Debate — synthesis intelligence at the centre of the multi-agent system.

Architecture
────────────
Round 0:
  All five agents (News, Social, Expert, Knowledge, Market) are decomposed
  into typed sub-queries by QueryDecomposer, then run in parallel and return
  AgentReports with structured findings.

Each subsequent round:
  The Debate synthesises the accumulated evidence, identifies tensions,
  and emits 2-3 pressure questions. Those questions become the sub-queries
  for the next round of agent research, targeting the most unresolved gaps.

Final output (DebateReport):
  - Full round log (RoundSynthesis per round)
  - Per-agent reports
  - Final trajectory assessment

Both The Debate and all research agents use EXPERT_MODEL (gpt-4o by default).
Model tier is defined in memex/config.py.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from memex import llm

from memex.forum.base_agent import AgentReport
from memex.forum.news_agent import NewsAgent
from memex.forum.social_agent import SocialAgent
from memex.forum.expert_agent import ExpertAgent
from memex.forum.knowledge_agent import KnowledgeAgent
from memex.forum.market_agent import MarketAgent
from memex.forum.decomposer import QueryDecomposer, DecomposedQuery
from memex.config import EXPERT_MODEL

logger = logging.getLogger(__name__)


# ── Structured output schema for Debate synthesis ─────────────────────────────

class RoundSynthesisOutput(BaseModel):
    """Schema the LLM must return each synthesis round."""
    situation: str
    tensions: str
    consensus_vs_divergence: str
    pressure_questions: list[str]
    counterintuitive: str
    trajectory: str


# ── Output data structures ─────────────────────────────────────────────────────

@dataclass
class RoundSynthesis:
    round_num: int
    situation: str                  # facts and timeline
    tensions: str                   # contradictions across agents
    consensus_vs_divergence: str    # where agents agree / disagree and why
    pressure_questions: list[str]   # sharp questions for the next round
    counterintuitive: str           # contrarian / historical challenge
    trajectory: str                 # 2-3 plausible paths + key variable
    raw_response: str = ""          # full LLM output for debugging


@dataclass
class DebateReport:
    query: str
    rounds: list[RoundSynthesis]
    agent_reports: dict[str, AgentReport]   # agent_type → AgentReport
    # Convenience: last round's trajectory is the final read
    final_trajectory: str = ""


# ── The Debate system prompt (user-provided) ─────────────────────────────────

def _build_system_prompt() -> str:
    return """You are the Debate — the synthesis intelligence at the \
center of a multi-agent analysis system. You do not merely summarize \
what agents report. Your role is to apply pressure: stress-test claims, \
expose contradictions, and forge a sharper picture than any single agent \
could produce alone.

## Your agents

- **NewsAgent**: Monitors breaking news, press releases, and mainstream \
media. Strong on facts and timelines. Prone to recency bias and \
official narratives.
  
- **SocialAgent**: Tracks Reddit, Hacker News, and public discourse. \
Strong on sentiment and emerging grassroots reaction. Prone to \
selection bias toward vocal minorities.
  
- **ExpertAgent**: Retrieves analysis from think tanks, Substack, \
academic sources, and specialist commentary. Strong on depth and \
frameworks. Prone to elite consensus blind spots.
  
- **KnowledgeAgent**: Surfaces historical precedents and analytical \
frameworks from a curated corpus (Thucydides, Clausewitz, Sun Tzu, \
Dalio, etc.). Strong on pattern recognition. Must be challenged when \
analogies are superficial.

- **MarketAgent**: Reads crowd-aggregated probabilities from prediction \
markets (Polymarket, Metaculus). Strong on quantified uncertainty and \
falsifiable framing. Prone to thin liquidity on niche topics; markets \
reflect trader priors, not ground truth.

## Your responsibilities each round

1. **Timeline reconstruction** — Extract the sequence of events across \
all agent reports. Surface what happened, in order, without \
editorializing yet.

2. **Claim audit** — Identify factual inconsistencies or logical \
contradictions between agents. Name them explicitly. Do not smooth \
them over.

3. **Perspective synthesis** — Map where agents agree, where they \
diverge, and why the divergence exists. Divergence is often more \
informative than consensus.

4. **Pressure questions** — Pose 2-3 sharp questions that none of the \
agents have adequately answered. These should push the next round \
toward the most unresolved tensions.

5. **Counterintuitive check** — Actively ask: what would a contrarian \
analyst say? What does the historical layer suggest that contradicts \
the current mainstream narrative?

6. **Trajectory assessment** — Based on accumulated evidence, what \
are the 2-3 most plausible trajectories? Assign rough confidence \
and name the key variable that determines which path materializes.

## Output format (strict JSON)

Return a JSON object with exactly these keys:
  "situation": string (facts and timeline, what happened in order)
  "tensions": string (contradictions and inconsistencies between agents)
  "consensus_vs_divergence": string (where agents agree vs diverge and why)
  "pressure_questions": array of 2-3 strings (sharp unresolved questions)
  "counterintuitive": string (contrarian view + historical challenge)
  "trajectory": string (2-3 plausible paths, rough confidence, key variable)

Keep each string under 300 words. Total output under 800 words.

## Operating principles

- Prioritize evidence over assertion. If a claim lacks a source, flag it.
- Maintain adversarial neutrality — your job is to find the truth, not \
  to validate any agent's framing.
- The historical layer (KnowledgeAgent) should sharpen analysis, not \
  decorate it. Challenge any historical analogy that feels rhetorical \
  rather than structural.
- Distinguish between what is known, what is inferred, and what is \
  speculated. Label each clearly.
- When the picture is genuinely unclear, say so. False confidence is \
  worse than acknowledged uncertainty."""


# ── Context builder ────────────────────────────────────────────────────────────

def _build_agent_context(reports: dict[str, AgentReport]) -> str:
    """Render all agent reports into a structured context block for The Debate."""
    parts: list[str] = []
    agent_order = ["news", "social", "expert", "knowledge", "market"]
    ordered = [(k, reports[k]) for k in agent_order if k in reports]
    ordered += [(k, v) for k, v in reports.items() if k not in agent_order]

    for _, report in ordered:
        parts.append(f"## {report.agent_label}")
        parts.append(f"**Known bias**: {report.caveats}\n")

        if report.summary:
            parts.append(f"**Summary**: {report.summary}\n")

        if report.findings:
            parts.append("**Findings**:")
            for f in report.findings[:8]:
                conf = f.get("confidence", "?")
                claim = f.get("claim", "")
                src = f.get("source_title", "")
                url = f.get("url", "")
                parts.append(f"  [{conf}] {claim}")
                if src:
                    parts.append(f"    Source: {src} — {url}")
            parts.append("")

    return "\n".join(parts)


def _build_round_prompt(
    query: str,
    agent_context: str,
    previous_rounds: list[RoundSynthesis],
    round_num: int,
) -> str:
    lines = [f"**Research query**: {query}", f"**Round**: {round_num + 1}\n"]

    if previous_rounds:
        lines.append("## Previous round synthesis")
        prev = previous_rounds[-1]
        lines.append(f"Situation: {prev.situation}")
        lines.append(f"Tensions: {prev.tensions}")
        lines.append(f"Pressure questions that agents just researched: {prev.pressure_questions}\n")

    lines.append("## Agent reports this round")
    lines.append(agent_context)
    return "\n".join(lines)


# ── The Debate ───────────────────────────────────────────────────────────────

class Debate:
    """
    Orchestrates the multi-agent research forum and produces a DebateReport.

    Usage:
        debate = Debate(client=openai_client, db_conn=pg_conn)
        report = await debate.run("What is driving the AI funding surge in 2026?")
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        db_conn=None,
        synthesis_model: str = EXPERT_MODEL,
        agent_model: str = EXPERT_MODEL,
        rounds: int = 2,
    ):
        self.client = client
        self.synthesis_model = synthesis_model
        self.rounds = rounds

        self.decomposer = QueryDecomposer(client, model=synthesis_model)

        self.news_agent      = NewsAgent(client, model=agent_model)
        self.social_agent    = SocialAgent(client, model=agent_model)
        self.expert_agent    = ExpertAgent(client, model=agent_model)
        self.knowledge_agent = KnowledgeAgent(client, db_conn=db_conn, model=agent_model)
        self.market_agent    = MarketAgent(client, model=agent_model)

    async def run(self, query: str) -> DebateReport:
        """
        Execute the full multi-round research + synthesis loop.

        Round 0: decompose query → run all 5 agents in parallel with typed sub-queries.
        Each subsequent round: pressure questions from the Debate become new sub-queries.
        Returns a DebateReport with all round syntheses and agent reports.
        """
        all_agent_reports: dict[str, AgentReport] = {}
        round_log: list[RoundSynthesis] = []
        logger.info("Debate starting: %r (%d rounds)", query, self.rounds)

        # Round 0: decompose and run with typed sub-queries
        logger.info("Decomposing query...")
        decomposed = await self.decomposer.decompose(query)
        logger.info("Intent: %s", decomposed.intent)

        for round_num in range(self.rounds):
            if round_num == 0:
                # First round: typed decomposed queries per agent
                new_reports = await self._run_agents_decomposed(decomposed)
            else:
                # Subsequent rounds: pressure questions as plain strings
                pressure_qs = round_log[-1].pressure_questions
                logger.info("Round %d — pressure questions: %s", round_num, pressure_qs)
                new_reports = await self._run_agents_followup(pressure_qs)

            # Merge into accumulated reports (later rounds extend earlier ones)
            for agent_type, report in new_reports.items():
                if agent_type in all_agent_reports:
                    # Merge findings and items
                    existing = all_agent_reports[agent_type]
                    existing.items.extend(report.items)
                    existing.findings.extend(report.findings)
                    existing.sub_queries.extend(report.sub_queries)
                else:
                    all_agent_reports[agent_type] = report

            # Synthesise this round
            agent_ctx = _build_agent_context(all_agent_reports)
            synthesis = await self._synthesise(query, agent_ctx, round_log, round_num)
            round_log.append(synthesis)

            logger.info(
                "Round %d synthesis done. Pressure questions: %s",
                round_num, synthesis.pressure_questions
            )

            # Pressure questions will drive the next round (consumed at round start)

        final_trajectory = round_log[-1].trajectory if round_log else ""
        return DebateReport(
            query=query,
            rounds=round_log,
            agent_reports=all_agent_reports,
            final_trajectory=final_trajectory,
        )

    async def _run_agents_decomposed(self, decomposed: DecomposedQuery) -> dict[str, AgentReport]:
        """Round 0: run all 5 agents with their typed, agent-specific sub-queries."""
        agent_calls = [
            ("news",      self.news_agent.run_decomposed(decomposed.news)),
            ("social",    self.social_agent.run_decomposed(decomposed.social)),
            ("expert",    self.expert_agent.run_decomposed(decomposed.expert)),
            ("knowledge", self.knowledge_agent.run_decomposed(decomposed.knowledge)),
            ("market",    self.market_agent.run_decomposed(decomposed.market)),
        ]
        return await self._gather_agents(agent_calls)

    async def _run_agents_followup(self, pressure_questions: list[str]) -> dict[str, AgentReport]:
        """Follow-up rounds: all agents research the pressure questions as plain strings."""
        agent_calls = [
            ("news",      self.news_agent.run(pressure_questions)),
            ("social",    self.social_agent.run(pressure_questions)),
            ("expert",    self.expert_agent.run(pressure_questions)),
            ("knowledge", self.knowledge_agent.run(pressure_questions)),
            ("market",    self.market_agent.run(pressure_questions)),
        ]
        return await self._gather_agents(agent_calls)

    async def _gather_agents(
        self, agent_calls: list[tuple[str, Any]]
    ) -> dict[str, AgentReport]:
        agent_types = [t for t, _ in agent_calls]
        results = await asyncio.gather(
            *[call for _, call in agent_calls], return_exceptions=True
        )
        reports: dict[str, AgentReport] = {}
        for agent_type, result in zip(agent_types, results):
            if isinstance(result, Exception):
                logger.warning("Agent %s failed: %s", agent_type, result)
                reports[agent_type] = AgentReport(
                    agent_type=agent_type,
                    agent_label=agent_type.capitalize() + "Agent",
                    sub_queries=[],
                    items=[],
                    summary=f"[Agent failed: {result}]",
                )
            else:
                reports[agent_type] = result
        return reports

    async def _synthesise(
        self,
        query: str,
        agent_context: str,
        previous_rounds: list[RoundSynthesis],
        round_num: int,
    ) -> RoundSynthesis:
        """Call The Debate LLM to synthesise the current round using structured output."""
        user_msg = _build_round_prompt(query, agent_context, previous_rounds, round_num)

        try:
            out = await llm.parse_structured(
                self.client,
                messages=[
                    {"role": "system", "content": _build_system_prompt()},
                    {"role": "user",   "content": user_msg},
                ],
                response_format=RoundSynthesisOutput,
                model=self.synthesis_model,
                temperature=0.4,
            )
            if out:
                return RoundSynthesis(
                    round_num=round_num,
                    situation=out.situation,
                    tensions=out.tensions,
                    consensus_vs_divergence=out.consensus_vs_divergence,
                    pressure_questions=out.pressure_questions,
                    counterintuitive=out.counterintuitive,
                    trajectory=out.trajectory,
                )
        except Exception as exc:
            logger.error("Debate synthesis failed: %s", exc)

        return RoundSynthesis(
            round_num=round_num,
            situation="[synthesis failed]",
            tensions="", consensus_vs_divergence="",
            pressure_questions=[], counterintuitive="", trajectory="",
        )


# ── Legacy alias (backwards compat with orchestrator.py that calls ForumEngine) ──
ForumEngine = Debate
