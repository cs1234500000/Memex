"""
api/schemas.py
Pydantic request/response models for all API routes.
Mirrors the data shapes from src/news/subscriptions.js, src/news/knowledgeStore.js,
and src/api/newsHandlers.js.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


# ------------------------------------------------------------------ #
# Subscriptions                                                        #
# ------------------------------------------------------------------ #

class NewsAPISourceConfig(BaseModel):
    language: str = "en"
    sort_by: str = Field("publishedAt", alias="sortBy")
    domains: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list, alias="sourceIds")

    model_config = {"populate_by_name": True}


class SourcesConfig(BaseModel):
    newsapi: NewsAPISourceConfig = Field(default_factory=NewsAPISourceConfig)


class Interest(BaseModel):
    id: str
    label: str
    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords")
    @classmethod
    def dedupe_keywords(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for kw in v:
            kw = kw.strip().lower()
            if kw and kw not in seen:
                seen.add(kw)
                result.append(kw)
        return result


class Subscriptions(BaseModel):
    source: str = "newsapi"
    twitter_handles: list[str] = Field(
        default_factory=lambda: [
            "OpenAI", "AnthropicAI", "perplexity_ai", "ycombinator", "a16z", "sama"
        ],
        alias="twitterHandles",
    )
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    interests: list[Interest] = Field(default_factory=list)
    max_items_per_run: int = Field(60, alias="maxItemsPerRun", ge=1, le=100)
    max_tweets_per_run: int = Field(60, alias="maxTweetsPerRun", ge=1, le=100)

    model_config = {"populate_by_name": True}


# ------------------------------------------------------------------ #
# Users                                                                #
# ------------------------------------------------------------------ #

class UserSignupRequest(BaseModel):
    user_id: str = Field(..., alias="userId", min_length=1)
    name: str = ""
    email: str = ""

    model_config = {"populate_by_name": True}


class UserSigninRequest(BaseModel):
    user_id: str = Field(..., alias="userId", min_length=1)
    email: str = Field(..., min_length=1)

    model_config = {"populate_by_name": True}


class UserResponse(BaseModel):
    user_id: str
    name: str
    email: str


# ------------------------------------------------------------------ #
# Ingest pipeline                                                      #
# ------------------------------------------------------------------ #

class RunNewsRequest(BaseModel):
    user_id: str = Field("default", alias="userId")
    user: dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# ------------------------------------------------------------------ #
# Knowledge                                                            #
# ------------------------------------------------------------------ #

class TopicStats(BaseModel):
    total_knowledge_items: int
    total_runs_merged: int
    first_merged_at: str | None
    last_merged_at: str | None
    latest_run_id: str | None
    latest_source: str | None


class KnowledgeItemResponse(BaseModel):
    id: str
    topic_id: str
    source: str
    content_type: str
    url: str
    author: str
    origin: str
    title: str
    text: str
    published_at: str | None
    discovered_at: str | None
    run_id: str
    score: float
    summary: str
    insights: list[str]
    metrics: dict[str, Any]


class RunSummaryResponse(BaseModel):
    run_id: str
    generated_at: str
    source: str
    matched_count: int
    summary: str
    insights: list[str]


class TopicDocument(BaseModel):
    schema_version: int = 1
    user_id: str
    topic: dict[str, str]
    stats: TopicStats
    latest: dict[str, Any]
    summaries: list[RunSummaryResponse]
    knowledge_items: list[KnowledgeItemResponse]


# ------------------------------------------------------------------ #
# Analyze (query-driven pipeline)                                      #
# ------------------------------------------------------------------ #

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    format: str = Field("html", pattern="^(html|json)$")
