"""
ingest/filter.py
Filters fetched items against a user query using a two-step LLM pipeline.

Step 1 — Criteria generation (GPT-4o):
  For each unique user query, GPT-4o writes a precise, query-specific set of
  filter criteria. Different queries produce different criteria. The result is
  cached in-memory so batch calls within one run share the same criteria.

Step 2 — Binary classification (GPT-4o-mini, structured output):
  Items are sent in batches. The model returns a JSON array where each item is
  classified as relevant: true or false, with a one-sentence reason.
  Only items where relevant=true are returned.

Fallback (no OpenAI / NEWS_USE_LLM=false):
  Keyword-only matching against the interest's keyword list.

Environment variables:
  NEWS_USE_LLM              true/false (default: true)
  NEWS_CRITERIA_MODEL       model for criteria generation (default: gpt-4o)
  NEWS_FILTER_MODEL         model for binary classification (default: gpt-4o-mini)
  NEWS_LLM_BATCH_SIZE       items per classification call (default: 10)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from memex.config import EXPERT_MODEL, LIGHTWEIGHT_MODEL
from memex import llm

logger = logging.getLogger(__name__)

# In-memory criteria cache: user_query → generated criteria string
_criteria_cache: dict[str, str] = {}


# ── Public data structure ──────────────────────────────────────────────────────

@dataclass
class FilteredItem:
    topic_id: str
    topic_label: str
    item: dict[str, Any]
    reason: str = ""    # one-sentence explanation from the classifier


# ── Structured output schema ───────────────────────────────────────────────────
# GPT-4o-mini returns this exact shape for each batch.

_CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "filter_results",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url":      {"type": "string"},
                            "relevant": {"type": "boolean"},
                            "reason":   {"type": "string"},
                        },
                        "required": ["url", "relevant", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}

# Meta-prompt sent to GPT-4o to produce a query-specific criteria prompt.
_CRITERIA_META_PROMPT = """\
You are a prompt engineer specialising in information-retrieval filters.

A user has entered a research query. Your job is to write a FILTER CRITERIA \
PROMPT that will be given to a smaller LLM (GPT-4o-mini) along with a batch \
of news article summaries. The smaller model will use your criteria to decide \
whether each article is relevant (true) or not (false).

Requirements for the criteria prompt you write:
- Be specific to the user's exact query — different queries must yield \
  meaningfully different criteria.
- List concrete signals that indicate relevance (e.g. named entities, event \
  types, recency signals, source types).
- List concrete signals that indicate irrelevance (e.g. off-topic domains, \
  keyword coincidences without substance, generic takes).
- Stay under 300 words.
- Write in imperative style ("Mark as relevant if…", "Exclude if…").
- Do NOT include any preamble or explanation — output only the criteria text.\
"""


# ── Step 1: criteria generation ────────────────────────────────────────────────

async def _generate_criteria(user_query: str, client) -> str:
    """
    Ask GPT-4o to write filter criteria tailored to this specific query.
    Result is cached so multiple batches in one run reuse the same criteria.
    """
    if user_query in _criteria_cache:
        return _criteria_cache[user_query]

    model = os.environ.get("NEWS_CRITERIA_MODEL", EXPERT_MODEL)
    criteria = await llm.complete(
        client,
        messages=[
            {"role": "system", "content": _CRITERIA_META_PROMPT},
            {"role": "user", "content": f'User research query: "{user_query}"'},
        ],
        model=model,
        temperature=0.3,
    )
    criteria = criteria.strip()
    _criteria_cache[user_query] = criteria
    logger.debug("Generated filter criteria for query %r:\n%s", user_query, criteria)
    return criteria


# ── Step 2: binary classification ─────────────────────────────────────────────

def _item_to_repr(item: dict) -> dict:
    return {
        "title":        item.get("title") or item.get("text", "")[:120],
        "snippet":      (item.get("snippet") or item.get("text", ""))[:300],
        "url":          item.get("url", ""),
        "source":       item.get("source", ""),
        "published_at": item.get("createdAt") or item.get("published_at", ""),
    }


async def _classify_batch(
    items: list[dict],
    criteria: str,
    model: str,
    client,
) -> list[dict]:
    """
    Classify one batch of items as relevant=true/false using structured output.
    Returns a list of enriched item dicts (only the relevant=true ones).
    """
    payload = json.dumps([_item_to_repr(it) for it in items], ensure_ascii=False)
    system_msg = (
        "You are a strict content filter. Apply the criteria below to each article "
        "and classify it as relevant (true) or not relevant (false).\n\n"
        f"CRITERIA:\n{criteria}\n\n"
        "Return the results array with one entry per input article, in the same order."
    )
    user_msg = f"Articles to classify:\n{payload}"

    try:
        parsed = await llm.complete_json(
            client,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            model=model,
            temperature=0.0,
            response_format=_CLASSIFICATION_SCHEMA,
        )
        results = parsed.get("results", [])
    except Exception as exc:
        logger.warning("Classification batch failed: %s", exc)
        return []

    url_to_item = {it.get("url", ""): it for it in items}
    kept = []
    for i, r in enumerate(results):
        if not r.get("relevant", False):
            continue
        url = r.get("url", "")
        original = url_to_item.get(url) or (items[i] if i < len(items) else None)
        if original is None:
            continue
        kept.append({**original, "_reason": r.get("reason", "")})
    return kept


# ── Keyword fallback ───────────────────────────────────────────────────────────

def _count_keyword_matches(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    count = 0
    for kw in keywords:
        if not kw:
            continue
        kw_lower = kw.lower().strip()
        if " " in kw_lower:
            count += kw_lower in lower
        else:
            count += bool(re.search(rf"\b{re.escape(kw_lower)}\b", lower))
    return count


def _keyword_filter(items: list[dict], interests: list[dict]) -> list[FilteredItem]:
    results = []
    for interest in interests:
        for item in items:
            if _count_keyword_matches(item.get("text", ""), interest.get("keywords", [])) > 0:
                results.append(FilteredItem(
                    topic_id=interest["id"],
                    topic_label=interest["label"],
                    item=item,
                ))
    return results


# ── Public entry points ────────────────────────────────────────────────────────

async def filter_items(
    items: list[dict],
    user_query: str,
    interests: list[dict],
    *,
    client=None,
) -> list[FilteredItem]:
    """
    Main entry point.

    1. GPT-4o generates query-specific filter criteria (cached per query).
    2. GPT-4o-mini classifies each item true/false using structured output.
    3. Returns only the relevant items as FilteredItem objects.

    Falls back to keyword matching if OpenAI is unavailable or NEWS_USE_LLM=false.
    """
    use_llm = os.environ.get("NEWS_USE_LLM", "true").lower() != "false"
    if not use_llm:
        return _keyword_filter(items, interests)

    from openai import AsyncOpenAI
    if client is None:
        client = AsyncOpenAI()

    default_topic = interests[0] if interests else {"id": "general", "label": user_query}
    filter_model = os.environ.get("NEWS_FILTER_MODEL", LIGHTWEIGHT_MODEL)
    batch_size = int(os.environ.get("NEWS_LLM_BATCH_SIZE", "10"))

    try:
        # Step 1 — generate criteria once for this query
        criteria = await _generate_criteria(user_query, client)

        # Step 2 — classify all batches concurrently
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        batch_results = await asyncio.gather(
            *[_classify_batch(b, criteria, filter_model, client) for b in batches],
            return_exceptions=True,
        )

        filtered: list[FilteredItem] = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.warning("Batch exception: %s", result)
                continue
            for enriched in result:
                filtered.append(FilteredItem(
                    topic_id=default_topic.get("id", "general"),
                    topic_label=default_topic.get("label", user_query),
                    reason=enriched.pop("_reason", ""),
                    item=enriched,
                ))

        if filtered:
            return filtered
        logger.warning("LLM filter passed 0 items — falling back to keyword filter")

    except Exception as exc:
        logger.warning("LLM filter failed, falling back to keyword: %s", exc)

    return _keyword_filter(items, interests)


def filter_by_interest(items: list[dict], interests: list[dict]) -> list[FilteredItem]:
    """Sync entry point — keyword only. Use filter_items() for LLM filtering."""
    return _keyword_filter(items, interests)


def group_by_topic(items: list[FilteredItem]) -> dict[str, dict]:
    """Group a flat list of FilteredItems by topic_id."""
    groups: dict[str, dict] = {}
    for fi in items:
        if fi.topic_id not in groups:
            groups[fi.topic_id] = {
                "topic_id": fi.topic_id,
                "topic_label": fi.topic_label,
                "items": [],
            }
        groups[fi.topic_id]["items"].append(fi)
    return groups
