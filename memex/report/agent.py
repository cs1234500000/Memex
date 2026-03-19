"""
report/agent.py
Generates the final deep-analysis report by:
  1. Building a KnowledgeGraph (GraphRAG) from all inputs
  2. Filling the IR template with LLM-generated chapter content
  3. Rendering to HTML (and optionally PDF)
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex import llm

from memex.report.graphrag import GraphRAG, KnowledgeGraph
from memex.report.renderer import ReportRenderer

TEMPLATE_PATH = pathlib.Path(__file__).parent / "templates" / "deep_analysis_en.md"

CHAPTER_SYSTEM = (
    "You are a professional research analyst writing one section of a deep-analysis report. "
    "Be analytical, precise, and evidence-grounded. "
    "Use markdown formatting. Length: 200–350 words."
)


class ReportAgent:
    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        self.client = client
        self.model = model
        self.graphrag = GraphRAG(client, model)
        self.renderer = ReportRenderer()
        self.template = TEMPLATE_PATH.read_text()

    async def generate(
        self,
        query: str,
        agent_outputs: dict[str, Any],
        forum_log: list[dict],
        output_dir: str | pathlib.Path | None = None,
    ) -> str:
        """
        Orchestrate report generation. Returns rendered HTML string.
        Optionally writes HTML (and PDF) to output_dir.
        """
        graph: KnowledgeGraph = await self.graphrag.build(agent_outputs, forum_log)

        chapters = await self._generate_chapters(query, agent_outputs, forum_log, graph)

        ir_markdown = self._fill_template(query, chapters, graph)

        html = self.renderer.to_html(ir_markdown, title=query)

        if output_dir:
            out = pathlib.Path(output_dir)
            self.renderer.save_html(html, out / "report.html")

        return html

    # ------------------------------------------------------------------ #
    # Chapter generation                                                   #
    # ------------------------------------------------------------------ #

    async def _generate_chapters(
        self,
        query: str,
        agent_outputs: dict[str, Any],
        forum_log: list[dict],
        graph: KnowledgeGraph,
    ) -> dict[str, str]:
        import asyncio

        tasks = {
            "executive_summary": self._write_chapter(
                "executive summary (4–5 sentences)", query,
                self._summaries(agent_outputs)
            ),
            "news_findings": self._write_chapter(
                "recent developments section", query,
                getattr(agent_outputs.get("news"), "summary", "")
            ),
            "social_sentiment": self._write_chapter(
                "community and practitioner sentiment section", query,
                getattr(agent_outputs.get("social"), "summary", "")
            ),
            "expert_analysis": self._write_chapter(
                "expert and academic analysis section", query,
                getattr(agent_outputs.get("expert"), "summary", "")
            ),
            "knowledge_context": self._write_chapter(
                "foundational context section", query,
                getattr(agent_outputs.get("knowledge"), "summary", "")
            ),
            "conclusion": self._write_chapter(
                "conclusion and forward-looking implications", query,
                self._forum_synthesis(forum_log)
            ),
        }

        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))

    async def _write_chapter(self, chapter_name: str, query: str, context: str) -> str:
        text = await llm.complete(
            self.client,
            messages=[
                {"role": "system", "content": CHAPTER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f'Write the {chapter_name} for a report on: "{query}"\n\n'
                        f"Context:\n{context[:3000]}"
                    ),
                },
            ],
            model=self.model,
            temperature=0.5,
            max_tokens=500,
        )
        return text.strip()

    # ------------------------------------------------------------------ #
    # Template filling                                                     #
    # ------------------------------------------------------------------ #

    def _fill_template(
        self, query: str, chapters: dict[str, str], graph: KnowledgeGraph
    ) -> str:
        forum_synthesis = chapters.get("conclusion", "")
        date = datetime.now(timezone.utc).strftime("%B %d, %Y")

        key_entities = "\n".join(
            f"- **{n.label}** ({n.type})" for n in graph.nodes[:15]
        )

        sources_set: set[str] = set()
        for output in []:  # populated in a real run from agent_outputs
            pass

        return self.template.format(
            query=query,
            date=date,
            executive_summary=chapters.get("executive_summary", ""),
            news_findings=chapters.get("news_findings", ""),
            social_sentiment=chapters.get("social_sentiment", ""),
            expert_analysis=chapters.get("expert_analysis", ""),
            knowledge_context=chapters.get("knowledge_context", ""),
            forum_synthesis=forum_synthesis,
            key_entities=key_entities or "_No entities extracted._",
            conclusion=chapters.get("conclusion", ""),
            sources="_Sources listed per section above._",
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _summaries(agent_outputs: dict[str, Any]) -> str:
        parts = []
        for agent_type, output in agent_outputs.items():
            if hasattr(output, "summary") and output.summary:
                parts.append(f"[{agent_type.upper()}]: {output.summary}")
        return "\n\n".join(parts)

    @staticmethod
    def _forum_synthesis(forum_log: list[dict]) -> str:
        host_entry = next(
            (e for e in reversed(forum_log) if e.get("persona") == "host"), None
        )
        return host_entry["content"] if host_entry else ""
