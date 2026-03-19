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
from typing import Any, Literal

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

class ClaimAudit(BaseModel):
    agent: str
    claim: str
    type: Literal["fact", "inference", "speculation"]
    sourced: bool
    flag: str | None = None


class Contradiction(BaseModel):
    claim_a: str
    agent_a: str
    claim_b: str
    agent_b: str
    resolution: str
    status: Literal["resolved", "unresolved"]


class PressureItem(BaseModel):
    agent: str
    claim: str
    challenge: str
    verdict: str


class PressureQuestion(BaseModel):
    question: str
    directed_to: str        # which agent should answer this next round
    why_unresolved: str     # what evidence is missing


class RoundVerdict(BaseModel):
    established: str    # what can now be treated as settled
    contested: str      # what remains genuinely uncertain
    delta: str          # what the debate revealed that was not visible at the start


class RoundSynthesisOutput(BaseModel):
    """Schema the LLM must return each synthesis round."""
    claim_audit: list[ClaimAudit]
    contradictions: list[Contradiction]
    pressure_applied: list[PressureItem]
    non_obvious_findings: list[str]
    pressure_questions: list[PressureQuestion]
    round_verdict: RoundVerdict


# ── Final report schema ────────────────────────────────────────────────────────

class TrajectoryPath(BaseModel):
    label: str              # short name, e.g. "Mainstream adoption"
    description: str        # detailed narrative of how this plays out (150+ words)
    confidence_pct: int     # rough probability 0-100
    supporting_evidence: list[str]   # analytical statements in your own words explaining WHY this evidence supports the path; optionally note the source in parentheses at the end — never copy-paste source titles verbatim
    countervailing_evidence: list[str]  # analytical statements in your own words explaining WHY this evidence cuts against the path; optionally note the source in parentheses — never copy-paste source titles verbatim
    key_variable: str       # the single observable that determines if this materializes
    falsifiable_test: str   # what would conclusively confirm or refute this path


class FinalReportOutput(BaseModel):
    """Schema the LLM must return for the final comprehensive report."""
    executive_summary: str              # 200+ word synthesis of the whole debate
    validated_claims: list[str]         # claims that survived all rounds of pressure with attribution
    contested_claims: list[str]         # claims that remained [UNRESOLVED] across rounds
    cross_agent_syntheses: list[str]    # insights only visible by combining ≥2 agents
    trajectories: list[TrajectoryPath]  # 2-3 richly evidenced plausible futures
    critical_blindspots: list[str]      # what the debate lacked data on / couldn't see
    analyst_note: str                   # what to watch next; specific, falsifiable, time-bound


# ── Output data structures ─────────────────────────────────────────────────────

@dataclass
class RoundSynthesis:
    round_num: int
    claim_audit: list[dict]             # per-claim type + sourcing audit
    contradictions: list[dict]          # inter-agent contradictions + resolutions
    pressure_applied: list[dict]        # targeted pressure per claim + verdict
    non_obvious_findings: list[str]     # cross-agent synthesis findings
    pressure_questions: list[dict]      # {question, directed_to, why_unresolved}
    round_verdict: dict                 # {established, contested, delta}
    raw_response: str = ""


@dataclass
class FinalReport:
    executive_summary: str
    validated_claims: list[str]
    contested_claims: list[str]
    cross_agent_syntheses: list[str]
    trajectories: list[dict]            # TrajectoryPath dicts
    critical_blindspots: list[str]
    analyst_note: str


@dataclass
class DebateReport:
    query: str
    rounds: list[RoundSynthesis]
    agent_reports: dict[str, AgentReport]   # agent_type → AgentReport
    final_report: FinalReport | None = None
    # Legacy convenience alias — points to executive_summary
    final_trajectory: str = ""


# ── The Debate system prompt ──────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """You are the debate forum host of a multi-agent \
intelligence analysis system. Your role is not to moderate \
politely — it is to conduct a rigorous adversarial process that \
forces genuine resolution. You apply pressure, expose weak \
reasoning, demand evidence, and refuse to let contradictions \
be smoothed over.

A valuable debate round ends with the analytical picture \
genuinely sharper than when it started. If every agent agrees \
and nothing is contested, you have failed.

---

## THE FIVE AGENTS AND THEIR BLIND SPOTS

Know not just what each agent contributes, but where each \
systematically misleads. Your pressure must target these \
failure modes precisely.

**NewsAgent**
Strength: named entities, verified events, official statements, \
chronological facts.
Failure modes:
- Recency bias — overweights the last 48 hours, misses slow \
  structural shifts.
- Source capture — launders press releases as facts. An official \
  statement is a claim, not a confirmation.
- False balance — "both sides said X" is not resolution when one \
  side is factually wrong.
When to challenge: whenever NewsAgent presents an official \
statement as settled fact, or ignores context older than a week.

**SocialAgent**
Strength: grassroots sentiment, emerging narratives before \
mainstream coverage, real human reaction.
Failure modes:
- Vocal minority problem — Reddit and HN skew young, technical, \
  Western, and male. High upvote count is not representative \
  of public opinion.
- Outrage amplification — emotionally charged content surfaces \
  more than measured analysis.
- Anecdote as data — one viral thread is not a trend.
When to challenge: whenever SocialAgent presents platform \
sentiment as public opinion, or treats volume of reaction as \
evidence of accuracy.

**ExpertAgent**
Strength: analytical depth, theoretical frameworks, specialist \
vocabulary, long-form reasoning.
Failure modes:
- Elite consensus capture — think tanks and Substack writers talk \
  to each other. Expert consensus can be collectively wrong.
- Framework obsession — applies a favored lens (geopolitical, \
  economic, technical) regardless of fit.
- Recency neglect — long-form analysis lags fast-moving events \
  by weeks.
When to challenge: whenever ExpertAgent cites consensus without \
noting who dissents, or applies a framework without arguing it \
fits this specific case.

**KnowledgeAgent**
Strength: structural pattern recognition, historical precedent, \
analytical frameworks that survive across centuries.
Failure modes:
- Superficial analogy — maps surface similarity onto structural \
  difference. "This is like X" without specifying what the \
  structural mechanism has in common.
- Determinism — historical patterns suggest tendencies, not \
  outcomes. History does not repeat; it rhymes imperfectly.
- Anachronism — applies pre-modern frameworks to situations \
  where technology, institutions, or information environments \
  have changed the causal structure.
When to challenge: whenever KnowledgeAgent offers a historical \
parallel without explicitly stating what is structurally \
different about the current case.

**MarketAgent**
Strength: quantified uncertainty, crowd-aggregated probability, \
falsifiable framing, real money behind claims.
Failure modes:
- Thin liquidity — on niche or emerging topics, markets may have \
  very few traders. A 73% probability based on $12K volume \
  means almost nothing.
- Trader prior bias — markets reflect what traders believe, which \
  may reflect media narratives rather than ground truth.
- Question framing dependency — the probability is only as good \
  as how the market question was written. A poorly framed \
  question produces a meaningless number.
When to challenge: whenever MarketAgent presents a probability \
without reporting volume, or when the market question does not \
cleanly map to the analytical question being asked.

---

## YOUR RESPONSIBILITIES EACH ROUND

### 1. Open with a claim audit
Before any synthesis, audit the inputs. For each agent's \
submission, ask:
- Is this a fact, an inference, or speculation? Label it.
- Is the source named? If not, flag it as unsourced.
- Does this contradict anything another agent reported?

State contradictions explicitly. Do not resolve them yet — \
surface them as open tensions that the debate must address.

### 2. Apply targeted pressure
For each significant claim in the round, apply pressure using \
the failure mode of the agent that produced it.

Do not apply generic skepticism. Apply specific pressure:
- To NewsAgent: "This is an official statement — what is the \
  incentive to say this? What does the behavior, not the \
  statement, show?"
- To SocialAgent: "This is Reddit sentiment — what is the \
  demographic of this subreddit? Does this represent the \
  affected population or a vocal subset?"
- To ExpertAgent: "This is the consensus view — who among \
  credible analysts dissents, and what is their argument?"
- To KnowledgeAgent: "This historical parallel assumes structural \
  similarity — what is critically different about the current \
  case that could invert the pattern?"
- To MarketAgent: "What is the trading volume behind this \
  probability? Does the market question cleanly map to the \
  analytical question?"

### 3. Force resolution on contradictions
When two agents contradict each other, do not present both views \
as equally valid. Force a resolution:
- Which claim has stronger sourcing?
- Which agent's failure mode is more likely to explain the \
  discrepancy?
- Is there a third interpretation that reconciles both?

State your resolution and defend it. If genuine uncertainty \
remains after applying all available evidence, say so explicitly \
and label it [UNRESOLVED] — but do not use uncertainty as an \
excuse to avoid taking a position.

### 4. Elevate the non-obvious
The most valuable output of a debate round is not a summary of \
what agents said — it is what the combination of agents reveals \
that none could see alone.

Actively look for:
- The claim that NewsAgent and MarketAgent agree on but \
  ExpertAgent contradicts — that tension is analytically rich.
- The historical pattern from KnowledgeAgent that reframes what \
  SocialAgent's sentiment data actually means.
- The prediction market probability that is wildly inconsistent \
  with expert consensus — someone is wrong, and that gap is \
  a finding.
- The second-order consequence that no agent mentioned but \
  follows logically from combining their outputs.

### 5. Issue pressure questions
End each round with 2-3 questions that target the most \
unresolved tensions. These questions must be:
- Specific, not general. Not "what will China do?" but "will \
  China accelerate SMIC investment following the October \
  restrictions, as it did after the 2019 entity list?"
- Directed. Each question should specify which agent is best \
  positioned to answer it in the next round.
- Falsifiable. Each question should have an observable answer \
  — not a philosophical inquiry.

### 6. Produce a round verdict
At the end of each round, state what has been established, \
what remains contested, and what the debate has revealed \
that was not visible at the start.

The round verdict is not a summary. It is a delta — what \
changed this round. If the picture did not sharpen, name \
what prevented resolution and what would be needed to \
resolve it.

---

## OUTPUT FORMAT

Return a JSON object with exactly these keys:

"claim_audit": array of objects, each with:
  - "agent": string
  - "claim": string
  - "type": "fact" | "inference" | "speculation"
  - "sourced": boolean
  - "flag": string or null (any concern about this claim)

"contradictions": array of objects, each with:
  - "claim_a": string (what one agent said)
  - "agent_a": string
  - "claim_b": string (what contradicts it)
  - "agent_b": string
  - "resolution": string (your reasoned resolution)
  - "status": "resolved" | "unresolved"

"pressure_applied": array of objects, each with:
  - "agent": string
  - "claim": string (the claim being pressured)
  - "challenge": string (the specific pressure you applied)
  - "verdict": string (does the claim hold, weaken, or collapse \
    under pressure?)

"non_obvious_findings": array of strings
  Each is a finding that emerges from combining agent outputs — \
  something no single agent could have produced alone. \
  Minimum 2, maximum 5. Each must name which agents' \
  outputs it synthesizes.

"pressure_questions": array of objects, each with:
  - "question": string
  - "directed_to": string (which agent should answer this)
  - "why_unresolved": string (what evidence is missing)

"round_verdict": object with:
  - "established": string (what can now be treated as settled)
  - "contested": string (what remains genuinely uncertain)
  - "delta": string (what the debate revealed that was not \
    visible at the start — this is the most important field)

---

## OPERATING PRINCIPLES

**Evidence hierarchy** — when claims conflict, rank sources:
1. Primary data (filings, prices, official records)
2. Specialist publications (think tanks, peer-reviewed)
3. Tier-1 journalism (Reuters, FT, Bloomberg, WSJ)
4. Expert commentary (Substack, analyst notes)
5. Social signal (Reddit, HN — sentiment only, not fact)
6. Historical analogy (pattern only, not prediction)

Higher-ranked evidence defeats lower-ranked evidence unless \
there is a specific, named reason it does not apply.

**Labeling discipline** — every claim in your output must \
carry one of:
[KNOWN] — directly evidenced by a named source
[INFERRED] — logical conclusion from known, named facts
[SPECULATED] — plausible but without direct evidence

Promotion is forbidden: you may not treat an inference as a \
known fact, and you may not treat speculation as an inference.

**Pressure is not cynicism** — the goal is not to tear down \
every claim. It is to determine which claims survive scrutiny \
and which do not. When a claim holds under pressure, say so \
clearly. Validated claims become the foundation of the \
final report.

**Resolve, do not defer** — it is easier to present both \
sides than to take a position. Do not do the easier thing. \
When evidence is sufficient to resolve a contradiction, \
resolve it. Reserve [UNRESOLVED] for cases where evidence \
is genuinely insufficient — not for cases where resolution \
is uncomfortable.

**The debate is the product** — the final report is built \
from what survives the debate. A weak debate produces a \
weak report regardless of how well the report is written. \
Hold the standard here."""


# ── Final report system prompt ────────────────────────────────────────────────

def _build_final_report_system_prompt() -> str:
    return """You are producing the final comprehensive intelligence \
report after a multi-round adversarial debate between five research \
agents. You have access to the full debate record: every claim \
audited, every contradiction resolved or flagged, every non-obvious \
finding, and every round verdict.

Your output is not a summary. It is the definitive analytical \
product of the debate — the distillation of what survived rigorous \
pressure testing across multiple rounds.

---

## WHAT YOU MUST DO

### Executive summary
Write 200-350 words synthesizing the complete debate. This is \
not a list of bullet points. It is a coherent analytical narrative \
that a senior analyst could read to understand the situation \
without reading anything else. It must:
- State what is now known with confidence and why
- Name the central unresolved tension
- Characterize the range of plausible futures and what drives them
- Acknowledge the limits of the evidence

### Validated claims
List only claims that:
1. Were not flagged [UNRESOLVED] in any round
2. Came from sources ranked above social signal in the evidence hierarchy
3. Were not successfully impeached by pressure from the debate host

Each validated claim must include the name of the publication, \
article, or report it came from. Do NOT mention agent names \
(NewsAgent, SocialAgent, etc.) — cite the source directly.

### Contested claims
List claims that remained [UNRESOLVED] across rounds. For each:
- State why it is contested (conflicting evidence, weak sourcing, \
  insufficient data)
- State what would be needed to resolve it
- Cite the source of the claim, not the agent that found it

### Cross-agent syntheses
List insights that emerged from combining evidence across multiple \
source types. These are the most valuable findings — they could not \
have been produced by any single source alone. Write them as \
analytical conclusions, citing the underlying sources (article titles, \
publications, data). Do NOT mention agent names.

### Trajectories
Produce exactly 2-3 trajectory paths. Each must be:

**Richly evidenced**: supporting_evidence and \
countervailing_evidence must be written in your own analytical \
words — explain WHY a piece of evidence supports or undermines \
the path. Never copy-paste a source title or claim verbatim. \
Bad: "Nvidia Announces NemoClaw — Nvidia Announces NemoClaw". \
Good: "Nvidia's NemoClaw initiative signals that a major \
incumbent is co-opting rather than blocking OpenClaw, which \
historically accelerates rather than slows adoption \
(Nvidia press release)." If two bullets say the same thing in \
different words, collapse them into one. Every bullet must add \
distinct analytical value.

**Internally consistent**: The evidence must logically lead to \
the described outcome. Name any assumption required to bridge \
from evidence to trajectory.

**Honestly weighted**: Assign confidence percentages that reflect \
the actual strength of evidence. If you cannot distinguish between \
two paths, explain why the key variable is genuinely uncertain. \
Confidence percentages must sum to 100.

**Falsifiable**: Each trajectory must include a specific, \
observable test — a named event, data release, or decision that \
would confirm or refute the path within a defined timeframe.

**Honest about countervailing evidence**: For each trajectory, \
name the strongest evidence that cuts against it. These should be \
different facts from the supporting evidence — not the same event \
reframed.

### Critical blindspots
Name what the debate could not see. What evidence was absent? \
Which sources were unavailable? What questions did the pressure \
questions surface that agents could not answer? These are not \
failures — they are the honest boundary of the analysis.

### Analyst note
One paragraph. Specific, falsifiable, time-bound. What should \
an analyst monitoring this situation watch in the next 30-90 days? \
Name specific events, reports, decisions, or data releases. Do not \
give generic advice.

---

## QUALITY STANDARDS

**Minimum length**: The executive_summary must be at least 200 words. \
Each trajectory description must be at least 100 words. \
This is not negotiable — brevity here is a quality failure.

**No fabrication**: Every claim must trace to an agent report. \
If you find yourself writing something that no agent said, label it \
[INFERRED] and state which known facts it follows from.

**No false resolution**: If the debate left something genuinely \
unresolved, the final report must say so. A confident-sounding \
report built on weak evidence is worse than an honest report that \
acknowledges uncertainty.

**The debate record is your only source**: Do not introduce new \
facts that were not in any agent report or debate round. Your \
job is synthesis and critical assessment, not new research."""


def _build_final_report_prompt(
    query: str,
    round_log: list[RoundSynthesis],
    agent_reports: dict[str, AgentReport],
) -> str:
    lines = [
        f"**Research query**: {query}",
        f"**Rounds completed**: {len(round_log)}",
        "",
    ]

    # Cumulative agent evidence
    lines.append("## ACCUMULATED AGENT EVIDENCE")
    lines.append(_build_agent_context(agent_reports))
    lines.append("")

    # Full round record
    lines.append("## DEBATE ROUND RECORD")
    for rnd in round_log:
        lines.append(f"### Round {rnd.round_num + 1}")

        if rnd.claim_audit:
            lines.append("**Claims audited:**")
            for c in rnd.claim_audit:
                flag = f" [FLAG: {c['flag']}]" if c.get("flag") else ""
                lines.append(
                    f"  [{c.get('type','?').upper()}|{'sourced' if c.get('sourced') else 'UNSOURCED'}] "
                    f"{c.get('agent','?')}: {c.get('claim','')}{flag}"
                )

        if rnd.contradictions:
            lines.append("**Contradictions:**")
            for ct in rnd.contradictions:
                lines.append(
                    f"  [{ct.get('status','?').upper()}] "
                    f"{ct.get('agent_a','?')} vs {ct.get('agent_b','?')}: "
                    f"{ct.get('resolution','')}"
                )

        if rnd.non_obvious_findings:
            lines.append("**Non-obvious findings:**")
            for f in rnd.non_obvious_findings:
                lines.append(f"  - {f}")

        rv = rnd.round_verdict
        if rv:
            lines.append(f"**Round verdict — established**: {rv.get('established','')}")
            lines.append(f"**Round verdict — contested**: {rv.get('contested','')}")
            lines.append(f"**Round verdict — delta**: {rv.get('delta','')}")

        lines.append("")

    return "\n".join(lines)


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
        prev = previous_rounds[-1]
        lines.append("## Previous round verdict")
        rv = prev.round_verdict
        lines.append(f"Established: {rv.get('established', '')}")
        lines.append(f"Contested:   {rv.get('contested', '')}")
        lines.append(f"Delta:       {rv.get('delta', '')}")
        lines.append("")
        lines.append("## Pressure questions agents just researched")
        for pq in prev.pressure_questions:
            lines.append(
                f"- [{pq.get('directed_to', '?')}] {pq.get('question', '')} "
                f"(was unresolved because: {pq.get('why_unresolved', '')})"
            )
        lines.append("")
        if prev.non_obvious_findings:
            lines.append("## Non-obvious findings from last round (carry forward)")
            for f in prev.non_obvious_findings:
                lines.append(f"- {f}")
            lines.append("")

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

    async def _emit(self, on_event, event_type: str, data: dict) -> None:
        """Fire a progress event to the optional streaming callback."""
        if on_event:
            try:
                await on_event(event_type, data)
            except Exception as exc:
                logger.debug("on_event callback raised: %s", exc)

    async def run(self, query: str, *, on_event=None) -> DebateReport:
        """
        Execute the full multi-round research + synthesis loop.

        Round 0: decompose query → run all 5 agents in parallel with typed sub-queries.
        Each subsequent round: pressure questions from the Debate become new sub-queries.
        Returns a DebateReport with all round syntheses and agent reports.

        on_event: optional async callable(event_type: str, data: dict) for streaming
                  progress to a UI. Events emitted: decompose_done, agent_start,
                  agent_done, round_start, round_done, final_done.
        """
        all_agent_reports: dict[str, AgentReport] = {}
        round_log: list[RoundSynthesis] = []
        logger.info("Debate starting: %r (%d rounds)", query, self.rounds)

        logger.info("Decomposing query...")
        decomposed = await self.decomposer.decompose(query)
        logger.info("Intent: %s", decomposed.intent)

        await self._emit(on_event, "decompose_done", {
            "intent": decomposed.intent,
            "news_query": decomposed.news.search_string,
            "expert_query": decomposed.expert.search_string,
            "knowledge_query": decomposed.knowledge.abstract_pattern,
            "market_queries": decomposed.market.search_terms,
            "social_subreddits": decomposed.social.subreddits,
        })

        for round_num in range(self.rounds):
            await self._emit(on_event, "round_start", {"round": round_num})

            if round_num == 0:
                new_reports = await self._run_agents_decomposed(decomposed, on_event=on_event)
            else:
                pressure_qs = [
                    pq.get("question", "") for pq in round_log[-1].pressure_questions
                    if pq.get("question")
                ]
                logger.info("Round %d — pressure questions: %s", round_num, pressure_qs)
                new_reports = await self._run_agents_followup(pressure_qs, on_event=on_event)

            for agent_type, report in new_reports.items():
                if agent_type in all_agent_reports:
                    existing = all_agent_reports[agent_type]
                    existing.items.extend(report.items)
                    existing.findings.extend(report.findings)
                    existing.sub_queries.extend(report.sub_queries)
                else:
                    all_agent_reports[agent_type] = report

            agent_ctx = _build_agent_context(all_agent_reports)
            synthesis = await self._synthesise(query, agent_ctx, round_log, round_num)
            round_log.append(synthesis)

            await self._emit(on_event, "round_done", {
                "round": round_num,
                "claim_audit": synthesis.claim_audit,
                "contradictions": synthesis.contradictions,
                "pressure_applied": synthesis.pressure_applied,
                "non_obvious_findings": synthesis.non_obvious_findings,
                "pressure_questions": synthesis.pressure_questions,
                "round_verdict": synthesis.round_verdict,
            })

            logger.info(
                "Round %d synthesis done. Pressure questions: %s",
                round_num, synthesis.pressure_questions
            )

        final_report = await self._produce_final_report(
            query, round_log, all_agent_reports
        )
        if final_report:
            await self._emit(on_event, "final_done", {
                "executive_summary": final_report.executive_summary,
                "validated_claims": final_report.validated_claims,
                "contested_claims": final_report.contested_claims,
                "cross_agent_syntheses": final_report.cross_agent_syntheses,
                "trajectories": final_report.trajectories,
                "critical_blindspots": final_report.critical_blindspots,
                "analyst_note": final_report.analyst_note,
            })

        return DebateReport(
            query=query,
            rounds=round_log,
            agent_reports=all_agent_reports,
            final_report=final_report,
            final_trajectory=final_report.executive_summary if final_report else (
                round_log[-1].round_verdict.get("delta", "") if round_log else ""
            ),
        )

    async def _produce_final_report(
        self,
        query: str,
        round_log: list[RoundSynthesis],
        agent_reports: dict[str, AgentReport],
    ) -> FinalReport | None:
        """
        Dedicated final synthesis pass — runs after all debate rounds.
        Produces a comprehensive, richly evidenced report with validated claims,
        contested terrain, trajectory paths, and analyst note.
        """
        if not round_log:
            return None

        prompt = _build_final_report_prompt(query, round_log, agent_reports)
        try:
            out = await llm.parse_structured(
                self.client,
                messages=[
                    {"role": "system", "content": _build_final_report_system_prompt()},
                    {"role": "user",   "content": prompt},
                ],
                response_format=FinalReportOutput,
                model=self.synthesis_model,
                temperature=0.3,
            )
            if out:
                return FinalReport(
                    executive_summary=out.executive_summary,
                    validated_claims=out.validated_claims,
                    contested_claims=out.contested_claims,
                    cross_agent_syntheses=out.cross_agent_syntheses,
                    trajectories=[t.model_dump() for t in out.trajectories],
                    critical_blindspots=out.critical_blindspots,
                    analyst_note=out.analyst_note,
                )
        except Exception as exc:
            logger.error("Final report generation failed: %s", exc)
        return None

    async def _run_agents_decomposed(self, decomposed: DecomposedQuery, on_event=None) -> dict[str, AgentReport]:
        """Round 0: run all 5 agents with their typed, agent-specific sub-queries."""
        agent_calls = [
            ("news",      self.news_agent.run_decomposed(decomposed.news)),
            ("social",    self.social_agent.run_decomposed(decomposed.social)),
            ("expert",    self.expert_agent.run_decomposed(decomposed.expert)),
            ("knowledge", self.knowledge_agent.run_decomposed(decomposed.knowledge)),
            ("market",    self.market_agent.run_decomposed(decomposed.market)),
        ]
        return await self._gather_agents(agent_calls, on_event=on_event)

    async def _run_agents_followup(self, pressure_questions: list[str], on_event=None) -> dict[str, AgentReport]:
        """Follow-up rounds: all agents research the pressure questions as plain strings."""
        agent_calls = [
            ("news",      self.news_agent.run(pressure_questions)),
            ("social",    self.social_agent.run(pressure_questions)),
            ("expert",    self.expert_agent.run(pressure_questions)),
            ("knowledge", self.knowledge_agent.run(pressure_questions)),
            ("market",    self.market_agent.run(pressure_questions)),
        ]
        return await self._gather_agents(agent_calls, on_event=on_event)

    async def _gather_agents(
        self, agent_calls: list[tuple[str, Any]], on_event=None
    ) -> dict[str, AgentReport]:
        async def _run_one(agent_type: str, coro) -> tuple[str, AgentReport]:
            await self._emit(on_event, "agent_start", {"agent": agent_type})
            try:
                result = await coro
                await self._emit(on_event, "agent_done", {
                    "agent": agent_type,
                    "summary": result.summary,
                    "item_count": len(result.items),
                    "findings": result.findings[:8],
                    "caveats": result.caveats,
                })
                return agent_type, result
            except Exception as exc:
                logger.warning("Agent %s failed: %s", agent_type, exc)
                report = AgentReport(
                    agent_type=agent_type,
                    agent_label=agent_type.capitalize() + "Agent",
                    sub_queries=[],
                    items=[],
                    summary=f"[Agent failed: {exc}]",
                )
                await self._emit(on_event, "agent_done", {
                    "agent": agent_type,
                    "summary": report.summary,
                    "item_count": 0,
                    "findings": [],
                    "caveats": "",
                    "error": str(exc),
                })
                return agent_type, report

        raw = await asyncio.gather(
            *[_run_one(t, c) for t, c in agent_calls],
            return_exceptions=True,
        )
        reports: dict[str, AgentReport] = {}
        for result in raw:
            if isinstance(result, Exception):
                logger.error("_run_one raised unexpectedly: %s", result)
                continue
            agent_type, report = result
            reports[agent_type] = report
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
                    claim_audit=[c.model_dump() for c in out.claim_audit],
                    contradictions=[c.model_dump() for c in out.contradictions],
                    pressure_applied=[p.model_dump() for p in out.pressure_applied],
                    non_obvious_findings=out.non_obvious_findings,
                    pressure_questions=[pq.model_dump() for pq in out.pressure_questions],
                    round_verdict=out.round_verdict.model_dump(),
                )
        except Exception as exc:
            logger.error("Debate synthesis failed: %s", exc)

        return RoundSynthesis(
            round_num=round_num,
            claim_audit=[],
            contradictions=[],
            pressure_applied=[],
            non_obvious_findings=[],
            pressure_questions=[],
            round_verdict={"established": "", "contested": "", "delta": "[synthesis failed]"},
        )


# ── Legacy alias (backwards compat with orchestrator.py that calls ForumEngine) ──
ForumEngine = Debate
