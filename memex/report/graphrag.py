"""
report/graphrag.py
Builds a lightweight knowledge graph from forum logs and agent outputs.

Nodes: entities (companies, people, concepts, events)
Edges: relationships extracted by the LLM (supports, contradicts, caused_by, …)

The graph is used by the Report Agent to:
  1. Identify central themes (high-degree nodes)
  2. Surface contradictions (opposing-edge pairs)
  3. Ground citations to specific sources
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex import llm

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM = """
You are a knowledge graph extractor.
Given a text passage, return a JSON object:
{
  "nodes": [{"id": "...", "label": "...", "type": "concept|entity|event"}],
  "edges": [{"source": "...", "target": "...", "relation": "..."}]
}
Keep node IDs short (snake_case). Limit to the 10 most important nodes.
Respond with ONLY valid JSON.
"""


@dataclass
class KGNode:
    id: str
    label: str
    type: str  # concept | entity | event


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str


@dataclass
class KnowledgeGraph:
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)

    def merge(self, other: "KnowledgeGraph") -> None:
        existing_ids = {n.id for n in self.nodes}
        for node in other.nodes:
            if node.id not in existing_ids:
                self.nodes.append(node)
                existing_ids.add(node.id)
        self.edges.extend(other.edges)

    def to_dict(self) -> dict:
        return {
            "nodes": [vars(n) for n in self.nodes],
            "edges": [vars(e) for e in self.edges],
        }


class GraphRAG:
    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        self.client = client
        self.model = model

    async def build(
        self,
        agent_outputs: dict[str, Any],
        forum_log: list[dict],
    ) -> KnowledgeGraph:
        """Extract entities and relations from all sources, merge into one graph."""
        passages: list[str] = []

        for output in agent_outputs.values():
            if hasattr(output, "summary") and output.summary:
                passages.append(output.summary)

        for entry in forum_log:
            if entry.get("persona") in ("optimist", "pessimist", "realist", "host"):
                passages.append(entry["content"])

        graph = KnowledgeGraph()
        for passage in passages:
            sub_graph = await self._extract(passage)
            graph.merge(sub_graph)

        logger.info(
            "GraphRAG: %d nodes, %d edges", len(graph.nodes), len(graph.edges)
        )
        return graph

    async def _extract(self, text: str) -> KnowledgeGraph:
        raw = await llm.complete_json(
            self.client,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": text[:3000]},
            ],
            model=self.model,
            temperature=0,
        )
        nodes = [KGNode(**n) for n in raw.get("nodes", [])]
        edges = [KGEdge(**e) for e in raw.get("edges", [])]
        return KnowledgeGraph(nodes=nodes, edges=edges)
