"""
ingest/summarizer.py
Synthesizes a set of filtered, enriched items into a structured research brief
oriented around the user's query.

The system prompt was authored by GPT-4o and produces a 5-part JSON brief:
  executive_summary  — 3-4 sentence landscape overview
  key_developments   — 3-5 named developments with headline + significance
  emerging_signals   — 2-3 early/weak signals worth watching
  entities           — notable companies, people, technologies (names only)
  sentiment          — "bullish" | "bearish" | "mixed" | "neutral"

Usage:
    from memex.ingest.summarizer import summarize
    brief = await summarize(items, user_query="AI startup funding 2026")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from memex.config import EXPERT_MODEL
from memex import llm

logger = logging.getLogger(__name__)


# ── GPT-4o-generated summarizer system prompt ─────────────────────────────────
_SUMMARIZER_SYSTEM_PROMPT = """\
You are an advanced language model tasked with summarizing research articles \
in response to a user's query. Your goal is to provide a comprehensive and \
insightful summary based on the provided articles. Follow these instructions \
carefully:

1. Understand the Query and Content: Begin by thoroughly understanding the \
user's research query or topic of interest. Review the curated list of relevant \
articles, which may include titles, snippets, full texts, URLs, and sources.

2. Executive Summary: Craft a concise synthesis of the overall landscape \
relevant to the query. This should be 3-4 sentences long and capture the \
essence of the topic based on the provided content. Ensure that this summary \
reflects the main themes and insights from the articles.

3. Key Developments: Identify and list 3-5 specific named developments. For \
each development, provide:
   - "headline": A brief, descriptive title.
   - "significance": A detailed explanation of why this development is \
important, including any relevant data, quotes, or paraphrased information \
from the articles.

4. Emerging Signals: Detect and list 2-3 early trends or weak signals that may \
be worth watching. These should be nascent ideas or patterns that could indicate \
future developments.

5. Entities: Extract and list notable companies, people, or technologies \
mentioned in the articles. Provide just the names without additional context.

6. Sentiment: Determine the overall sentiment of the market or community \
regarding the topic. Choose one of: "bullish", "bearish", "mixed", or \
"neutral". Base this on the tone and content of the articles.

7. Grounded Analysis: Ensure all information is grounded in the provided \
content. Do not hallucinate. If content is insufficient for a comprehensive \
summary, acknowledge this honestly in executive_summary.

8. Audience: Write for a sophisticated technical or investor audience. Be \
specific and precise. Name companies, cite numbers, and use quotes or \
paraphrased content from sources to support your analysis.

Output ONLY a strict JSON object with exactly these keys:
  "executive_summary" (string)
  "key_developments"  (array of objects with "headline" and "significance")
  "emerging_signals"  (array of strings)
  "entities"          (array of strings)
  "sentiment"         (string: one of bullish|bearish|mixed|neutral)

Do not include any text outside the JSON object.\
"""

_FALLBACK_BRIEF: dict[str, Any] = {
    "executive_summary": "Summary unavailable — LLM call failed or no items provided.",
    "key_developments": [],
    "emerging_signals": [],
    "entities": [],
    "sentiment": "neutral",
}


def _build_items_block(items: list[dict], max_items: int = 15, full_text_chars: int = 800) -> str:
    """Render items as a numbered text block for the prompt."""
    lines = []
    for i, item in enumerate(items[:max_items], 1):
        title = item.get("title") or item.get("text", "")[:80]
        url = item.get("url", "")
        source = item.get("source", "")
        # Prefer full_text (from enricher) over snippet over raw text
        body = (
            item.get("full_text", "")[:full_text_chars]
            or item.get("snippet", "")
            or item.get("text", "")[:300]
        )
        reason = item.get("_llm_reason") or item.get("reason", "")

        parts = [f"{i}. [{source}] {title}"]
        if url:
            parts.append(f"   URL: {url}")
        if reason:
            parts.append(f"   Why relevant: {reason}")
        if body:
            parts.append(f"   Content: {body}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


async def summarize(
    items: list[dict],
    user_query: str,
    *,
    client=None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Produce a structured research brief for the given items and query.

    Args:
        items: Filtered (and optionally enriched) items. Each dict should have
               at least one of: title, text, snippet, full_text.
        user_query: The user's research question (e.g. "AI startup funding 2026").
        client: AsyncOpenAI instance. If None, one is created from env.
        model: Override the model (default: NEWS_LLM_FILTER_MODEL or gpt-4o-mini).

    Returns:
        Dict with keys: executive_summary, key_developments, emerging_signals,
        entities, sentiment.
    """
    if not items:
        return {
            **_FALLBACK_BRIEF,
            "executive_summary": f"No relevant content found for query: \"{user_query}\".",
        }

    from openai import AsyncOpenAI
    if client is None:
        client = AsyncOpenAI()

    resolved_model = model or EXPERT_MODEL
    items_block = _build_items_block(items)
    user_msg = f'Research query: "{user_query}"\n\nArticles:\n\n{items_block}'

    try:
        brief = await llm.complete_json(
            client,
            messages=[
                {"role": "system", "content": _SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=resolved_model,
            temperature=0.2,
        )
        for key, default in _FALLBACK_BRIEF.items():
            if key not in brief:
                brief[key] = default
        return brief
    except Exception as exc:
        logger.warning("Summarizer LLM call failed: %s", exc)
        return _FALLBACK_BRIEF


async def summarize_topic_group(
    filtered_items,   # list[FilteredItem] from filter.py
    user_query: str,
    *,
    client=None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper that accepts FilteredItem objects directly (from
    filter.py) and extracts the underlying item dicts, threading the LLM
    reason through to the summarizer context.
    """
    items = []
    for fi in filtered_items:
        item = dict(fi.item)
        item["_llm_reason"] = getattr(fi, "reason", "")
        items.append(item)
    return await summarize(items, user_query, client=client, model=model)
