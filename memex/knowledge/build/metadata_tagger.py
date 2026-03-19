"""
metadata_tagger.py
Uses an LLM to attach structured metadata to each text chunk:
  - concept    : the core idea or thesis of the chunk
  - scenarios  : 2–3 situations where this knowledge applies
  - caveats    : limitations, counter-arguments, or exceptions
  - source     : passed through from the caller
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from memex.config import EXPERT_MODEL
from memex import llm

from memex.knowledge.build.concept_chunker import TextChunk


TAG_SYSTEM = """
You are a knowledge metadata tagger.
Given a text chunk, return a JSON object with exactly these keys:
  "concept"   – one sentence naming the core idea or argument
  "scenarios" – a list of 2–3 strings describing when this applies
  "caveats"   – a list of 1–3 strings describing limitations or exceptions
Respond with ONLY valid JSON.
"""


@dataclass
class TaggedChunk:
    text: str
    word_count: int
    source: str
    title: str
    source_url: str
    concept: str
    scenarios: list[str]
    caveats: list[str]


class MetadataTagger:
    def __init__(self, client: AsyncOpenAI, model: str = EXPERT_MODEL):
        self.client = client
        self.model = model

    async def tag(
        self,
        chunk: TextChunk,
        source: str = "",
        title: str = "",
        source_url: str = "",
    ) -> TaggedChunk:
        """Attach metadata to a single chunk via LLM."""
        raw = await llm.complete_json(
            self.client,
            messages=[
                {"role": "system", "content": TAG_SYSTEM},
                {"role": "user", "content": chunk.text[:2000]},
            ],
            model=self.model,
            temperature=0.2,
        )

        return TaggedChunk(
            text=chunk.text,
            word_count=chunk.word_count,
            source=source,
            title=title,
            source_url=source_url,
            concept=raw.get("concept", ""),
            scenarios=raw.get("scenarios", []),
            caveats=raw.get("caveats", []),
        )

    async def tag_all(
        self,
        chunks: list[TextChunk],
        source: str = "",
        title: str = "",
        source_url: str = "",
    ) -> list[TaggedChunk]:
        """Tag a list of chunks (sequential to avoid rate limits)."""
        import asyncio
        return await asyncio.gather(
            *[self.tag(c, source, title, source_url) for c in chunks]
        )
