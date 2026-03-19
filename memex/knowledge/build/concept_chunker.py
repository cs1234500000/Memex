"""
concept_chunker.py
Splits raw text into argument-unit chunks of 200–400 words each.

Strategy:
  1. Split on paragraph/sentence boundaries (not mid-sentence)
  2. Merge short paragraphs until the chunk hits the target word count
  3. Each chunk is a self-contained argumentative unit
"""

from __future__ import annotations

import re
from dataclasses import dataclass


TARGET_MIN_WORDS = 200
TARGET_MAX_WORDS = 400


@dataclass
class TextChunk:
    text: str
    word_count: int
    char_start: int
    char_end: int


class ConceptChunker:
    def __init__(
        self,
        min_words: int = TARGET_MIN_WORDS,
        max_words: int = TARGET_MAX_WORDS,
    ):
        self.min_words = min_words
        self.max_words = max_words

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into argument-unit chunks."""
        paragraphs = self._split_paragraphs(text)
        return self._merge_into_chunks(paragraphs, text)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines; fall back to sentence splitting if needed."""
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        # Further split very long paragraphs at sentence boundaries
        result: list[str] = []
        for para in paras:
            if len(para.split()) > self.max_words:
                result.extend(self._split_sentences(para))
            else:
                result.append(para)
        return result

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        groups: list[str] = []
        current: list[str] = []
        current_words = 0

        for sent in sentences:
            wc = len(sent.split())
            if current_words + wc > self.max_words and current:
                groups.append(" ".join(current))
                current = [sent]
                current_words = wc
            else:
                current.append(sent)
                current_words += wc

        if current:
            groups.append(" ".join(current))
        return groups

    def _merge_into_chunks(self, paragraphs: list[str], original: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_words = 0
        offset = 0

        for para in paragraphs:
            wc = len(para.split())
            if current_words + wc > self.max_words and current_words >= self.min_words:
                text = "\n\n".join(current_parts)
                chunks.append(TextChunk(
                    text=text,
                    word_count=current_words,
                    char_start=offset - len(text),
                    char_end=offset,
                ))
                current_parts = [para]
                current_words = wc
            else:
                current_parts.append(para)
                current_words += wc

            offset += len(para) + 2  # +2 for "\n\n"

        if current_parts:
            text = "\n\n".join(current_parts)
            chunks.append(TextChunk(
                text=text,
                word_count=current_words,
                char_start=offset - len(text),
                char_end=offset,
            ))

        return chunks
