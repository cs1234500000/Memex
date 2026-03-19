"""
llm.py
Central gateway for all OpenAI API calls in the memex codebase.

Every module that needs to talk to the LLM should go through this module
instead of calling the SDK directly. Benefits:
  - One place to add logging, retries, cost tracking, or swap providers
  - Consistent error surfacing
  - Removes boilerplate (resp.choices[0].message.content / .parsed)

Three helpers, three patterns:

  parse_structured  →  client.beta.chat.completions.parse()
                        Returns a validated Pydantic model.
                        Use this for all structured outputs.

  complete          →  client.chat.completions.create()
                        Returns a plain string.
                        Use this for free-text generations (summaries, queries).

  complete_json     →  client.chat.completions.create(response_format=json_object)
                        Returns a parsed dict.
                        Use for legacy JSON calls that predate Pydantic schemas,
                        or custom JSON schemas (e.g. filter classification).
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def parse_structured(
    client: AsyncOpenAI,
    messages: list[dict],
    response_format: type[T],
    *,
    model: str,
    temperature: float = 0.3,
) -> T | None:
    """
    Structured output via client.beta.chat.completions.parse().

    The SDK generates a spec-compliant strict JSON schema from the Pydantic
    model automatically (including additionalProperties:false on all nested
    objects), so the response is guaranteed to match the schema.

    Returns the parsed Pydantic object, or None if parsing fails.
    """
    resp = await client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        response_format=response_format,
        messages=messages,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        logger.warning(
            "parse_structured: model returned no parsed object "
            "(model=%s, response_format=%s). Raw: %.200s",
            model, response_format.__name__,
            resp.choices[0].message.content,
        )
    return parsed


async def complete(
    client: AsyncOpenAI,
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> str:
    """
    Plain text completion via client.chat.completions.create().
    Returns the content string (never None — empty string on no output).
    """
    kwargs: dict = dict(model=model, temperature=temperature, messages=messages)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def complete_json(
    client: AsyncOpenAI,
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> dict:
    """
    JSON-mode completion via client.chat.completions.create().

    Defaults to {"type": "json_object"}. Pass a custom response_format dict
    for strict JSON schema mode (e.g. the filter classification schema).
    Returns a parsed dict (empty dict on parse failure).
    """
    fmt = response_format or {"type": "json_object"}
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format=fmt,
        messages=messages,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("complete_json: JSON decode failed (%s). Raw: %.200s", exc, raw)
        return {}
