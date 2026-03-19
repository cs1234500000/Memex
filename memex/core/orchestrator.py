"""
orchestrator.py
Dispatches decomposed sub-queries to the four agents in parallel,
feeds their outputs into the forum engine, then hands off to the report agent.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL

from memex.forum.news_agent import NewsAgent
from memex.forum.social_agent import SocialAgent
from memex.forum.expert_agent import ExpertAgent
from memex.forum.knowledge_agent import KnowledgeAgent
from memex.forum.market_agent import MarketAgent
from memex.forum.decomposer import DecomposedQuery, QueryDecomposer
from memex.forum.engine import ForumEngine
from memex.report.agent import ReportAgent

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    query: str
    started_at: datetime
    finished_at: datetime | None = None
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    forum_log: list[dict] = field(default_factory=list)
    report_html: str = ""
    report_pdf_path: str | None = None


class Orchestrator:
    """
    Top-level controller:
      1. Decompose query into per-agent sub-queries
      2. Dispatch all four agents in parallel (asyncio.gather)
      3. Run forum engine (debate + synthesis)
      4. Generate final report
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        db_conn=None,
        model: str = EXPERT_MODEL,
    ):
        self.client = client
        self.db = db_conn
        self.decomposer = QueryDecomposer(client, model)
        self.forum = ForumEngine(client, model)
        self.reporter = ReportAgent(client, model)

        self.news_agent = NewsAgent(client, model)
        self.social_agent = SocialAgent(client, model)
        self.expert_agent = ExpertAgent(client, model)
        self.knowledge_agent = KnowledgeAgent(client, db_conn, model)
        self.market_agent = MarketAgent(client, model)

    async def run(self, query: str) -> RunResult:
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        logger.info("Run %s started | query=%r", run_id, query)

        result = RunResult(run_id=run_id, query=query, started_at=started_at)

        decomposed: DecomposedQuery = await self.decomposer.decompose(query)
        logger.info("Decomposed intent: %s", decomposed.intent)

        news_out, social_out, expert_out, knowledge_out, market_out = await asyncio.gather(
            self.news_agent.run_decomposed(decomposed.news),
            self.social_agent.run_decomposed(decomposed.social),
            self.expert_agent.run_decomposed(decomposed.expert),
            self.knowledge_agent.run_decomposed(decomposed.knowledge),
            self.market_agent.run_decomposed(decomposed.market),
        )

        result.agent_outputs = {
            "news": news_out,
            "social": social_out,
            "expert": expert_out,
            "knowledge": knowledge_out,
            "market": market_out,
        }

        result.forum_log = await self.forum.run(
            query=query,
            agent_outputs=result.agent_outputs,
        )

        result.report_html = await self.reporter.generate(
            query=query,
            agent_outputs=result.agent_outputs,
            forum_log=result.forum_log,
        )

        result.finished_at = datetime.now(timezone.utc)
        logger.info("Run %s finished in %.1fs", run_id,
                    (result.finished_at - started_at).total_seconds())

        if self.db:
            await self._persist(result)

        return result

    async def _persist(self, result: RunResult) -> None:
        """Write run metadata to the `runs` table."""
        # TODO: implement with asyncpg
        pass
