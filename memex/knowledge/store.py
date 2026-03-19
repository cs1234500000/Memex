"""
knowledge/store.py
Two responsibilities:

1. pgvector corpus store  — embedding-based read/write for the offline
   knowledge corpus used by KnowledgeAgent.

2. User topic knowledge store — user/topic/run/item persistence that
   powers the ingest pipeline. Migrated from src/news/knowledgeStore.js.
   Supabase REST calls are replaced by direct asyncpg queries.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.70

# Supabase table names (kept as env-configurable constants)
USERS_TABLE = "news_users"
TOPICS_TABLE = "news_user_topics"
TOPIC_RUNS_TABLE = "news_user_topic_runs"
ITEMS_TABLE = "news_user_knowledge_items"


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def sanitize_id(value: str, fallback: str = "default") -> str:
    cleaned = re.sub(r"[^a-z0-9_-]", "_", str(value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


def infer_content_type(url: str) -> str:
    url_lower = url.lower()
    if re.search(r"\.(mp4|mov|avi|mkv|webm)(\?|$)|youtube\.com|youtu\.be|vimeo\.com", url_lower):
        return "video"
    if re.search(r"\.(mp3|wav|m4a|aac|ogg)(\?|$)|spotify\.com|soundcloud\.com|podcast", url_lower):
        return "audio"
    return "article"


def truncate_text(text: str, max_len: int = 240) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    return clean[:max_len] + "..." if len(clean) > max_len else clean


# ------------------------------------------------------------------ #
# KnowledgeStore                                                       #
# ------------------------------------------------------------------ #

class KnowledgeStore:
    def __init__(self, conn, client: AsyncOpenAI):
        """
        conn   — asyncpg connection or pool (None → offline/test mode)
        client — OpenAI async client (for embedding generation)
        """
        self.conn = conn
        self.client = client

    # ============================================================== #
    # 1. pgvector corpus (used by KnowledgeAgent)                    #
    # ============================================================== #

    async def similarity_search(
        self, query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Embed query and retrieve top-k nearest chunks."""
        embedding = await self._embed(query)
        if self.conn is None:
            return []

        rows = await self.conn.fetch(
            """
            SELECT id, text, title, source_url, concept, scenarios, caveats,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM knowledge_chunks
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            json.dumps(embedding),
            SIMILARITY_THRESHOLD,
            top_k,
        )
        return [dict(r) for r in rows]

    async def upsert_chunk(self, chunk: dict[str, Any]) -> None:
        embedding = await self._embed(chunk["text"])
        await self.conn.execute(
            """
            INSERT INTO knowledge_chunks
              (id, text, embedding, title, source_url, concept, scenarios, caveats)
            VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE
              SET text=EXCLUDED.text, embedding=EXCLUDED.embedding,
                  title=EXCLUDED.title, concept=EXCLUDED.concept,
                  scenarios=EXCLUDED.scenarios, caveats=EXCLUDED.caveats
            """,
            chunk["id"],
            chunk["text"],
            json.dumps(embedding),
            chunk.get("title", ""),
            chunk.get("source_url", ""),
            chunk.get("concept", ""),
            chunk.get("scenarios", ""),
            chunk.get("caveats", ""),
        )

    # ============================================================== #
    # 2. User / topic knowledge (migrated from knowledgeStore.js)    #
    # ============================================================== #

    async def upsert_user(
        self, user_id: str, name: str = "", email: str = ""
    ) -> dict:
        """Ensure a news_users row exists. Mirrors ensureUserRow() + upsertNewsUser()."""
        if self.conn is None:
            raise RuntimeError("DB connection required for user operations.")

        safe_id = sanitize_id(user_id)
        now = datetime.now(timezone.utc).isoformat()

        existing = await self.conn.fetchrow(
            f"SELECT display_name, email FROM {USERS_TABLE} WHERE user_id = $1 LIMIT 1",
            safe_id,
        )
        display_name = name.strip() or (existing["display_name"] if existing else "")
        norm_email = email.strip().lower() or (existing["email"] if existing else "")

        await self.conn.execute(
            f"""
            INSERT INTO {USERS_TABLE} (user_id, display_name, email, last_run_at, updated_at)
            VALUES ($1, $2, $3, $4, $4)
            ON CONFLICT (user_id) DO UPDATE
              SET display_name = EXCLUDED.display_name,
                  email        = EXCLUDED.email,
                  last_run_at  = EXCLUDED.last_run_at,
                  updated_at   = EXCLUDED.updated_at
            """,
            safe_id, display_name, norm_email, now,
        )
        return {"user_id": safe_id, "name": display_name, "email": norm_email}

    async def find_user(self, user_id: str = "", email: str = "") -> dict | None:
        """Look up a user by ID and/or email. Mirrors findNewsUser()."""
        if self.conn is None:
            raise RuntimeError("DB connection required for user operations.")

        safe_id = sanitize_id(user_id, "")
        norm_email = email.strip().lower()

        query = f"SELECT user_id, display_name, email FROM {USERS_TABLE} WHERE "
        if safe_id and norm_email:
            query += "user_id = $1 AND email = $2 LIMIT 1"
            row = await self.conn.fetchrow(query, safe_id, norm_email)
        elif safe_id:
            query += "user_id = $1 LIMIT 1"
            row = await self.conn.fetchrow(query, safe_id)
        elif norm_email:
            query += "email = $1 LIMIT 1"
            row = await self.conn.fetchrow(query, norm_email)
        else:
            return None

        if not row:
            return None
        return {"user_id": row["user_id"], "name": row["display_name"] or "", "email": row["email"] or ""}

    async def merge_report(
        self, user_id: str, report, user_profile: dict | None = None
    ) -> dict:
        """
        Persist a full IngestReport to the DB.
        Mirrors mergeUserKnowledgeFromReport() from knowledgeStore.js.
        """
        if self.conn is None:
            raise RuntimeError("DB connection required.")

        safe_id = sanitize_id(user_id)
        await self.upsert_user(safe_id, **(user_profile or {}))

        merged_topics = []
        for topic in report.topics:
            merged = await self._upsert_topic_knowledge(
                user_id=safe_id,
                source=report.source,
                report_id=report.id,
                generated_at=report.generated_at,
                topic=topic,
            )
            merged_topics.append(merged)

        return {"user_id": safe_id, "merged_topics": merged_topics}

    async def get_subscriptions(self, user_id: str = "default") -> dict | None:
        """Load a user's saved subscriptions config from DB."""
        if self.conn is None:
            return None
        row = await self.conn.fetchrow(
            f"SELECT config FROM user_subscriptions WHERE user_id = $1 LIMIT 1",
            sanitize_id(user_id),
        )
        if not row:
            return None
        return _parse_json(row["config"], None)

    async def save_subscriptions(self, user_id: str, config: dict) -> None:
        """Persist a user's subscriptions config to DB."""
        if self.conn is None:
            raise RuntimeError("DB connection required.")
        import json as _json
        await self.conn.execute(
            """
            INSERT INTO user_subscriptions (user_id, config, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id) DO UPDATE
              SET config = EXCLUDED.config, updated_at = EXCLUDED.updated_at
            """,
            sanitize_id(user_id),
            _json.dumps(config),
        )

    async def get_latest_report(self, user_id: str) -> dict | None:
        """Return the most recent topic snapshot for a user."""
        if self.conn is None:
            return None
        rows = await self.conn.fetch(
            f"""
            SELECT t.topic_id, t.topic_label, t.latest_summary, t.latest_insights,
                   t.last_merged_at, t.total_knowledge_items
            FROM {TOPICS_TABLE} t
            WHERE t.user_id = $1
            ORDER BY t.last_merged_at DESC NULLS LAST
            """,
            sanitize_id(user_id),
        )
        if not rows:
            return None
        return {
            "user_id": user_id,
            "topics": [
                {
                    "topicId": r["topic_id"],
                    "topicLabel": r["topic_label"],
                    "summary": r["latest_summary"] or "",
                    "insights": _parse_json(r["latest_insights"], []),
                    "lastMergedAt": _iso(r.get("last_merged_at")),
                    "totalItems": r["total_knowledge_items"],
                }
                for r in rows
            ],
        }

    async def get_user_topic_knowledge(
        self, user_id: str, topic_id: str | None = None
    ) -> list[dict] | dict | None:
        """
        Retrieve topic document(s) for a user.
        Mirrors getUserTopicKnowledge() from knowledgeStore.js.
        """
        if self.conn is None:
            raise RuntimeError("DB connection required.")

        safe_id = sanitize_id(user_id)

        if topic_id:
            return await self._fetch_topic_document(safe_id, sanitize_id(topic_id))

        rows = await self.conn.fetch(
            f"SELECT topic_id FROM {TOPICS_TABLE} WHERE user_id = $1 ORDER BY topic_id",
            safe_id,
        )
        docs = []
        for row in rows:
            doc = await self._fetch_topic_document(safe_id, row["topic_id"])
            if doc:
                docs.append(doc)
        return docs

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _upsert_topic_knowledge(
        self,
        user_id: str,
        source: str,
        report_id: str,
        generated_at: str,
        topic,
    ) -> dict:
        safe_topic_id = sanitize_id(getattr(topic, "topic_id", "general"))

        existing = await self.conn.fetchrow(
            f"SELECT first_merged_at FROM {TOPICS_TABLE} WHERE user_id=$1 AND topic_id=$2 LIMIT 1",
            user_id, safe_topic_id,
        )
        first_merged_at = (existing["first_merged_at"] if existing else None) or generated_at

        await self.conn.execute(
            f"""
            INSERT INTO {TOPICS_TABLE}
              (user_id, topic_id, topic_label, total_knowledge_items, total_runs_merged,
               first_merged_at, last_merged_at, latest_run_id, latest_source,
               latest_summary, latest_insights, updated_at)
            VALUES ($1,$2,$3,0,0,$4,$4,$5,$6,$7,$8,$4)
            ON CONFLICT (user_id, topic_id) DO UPDATE
              SET topic_label    = EXCLUDED.topic_label,
                  last_merged_at = EXCLUDED.last_merged_at,
                  latest_run_id  = EXCLUDED.latest_run_id,
                  latest_source  = EXCLUDED.latest_source,
                  latest_summary = EXCLUDED.latest_summary,
                  latest_insights= EXCLUDED.latest_insights,
                  updated_at     = EXCLUDED.updated_at
            """,
            user_id, safe_topic_id,
            getattr(topic, "topic_label", safe_topic_id),
            first_merged_at,
            report_id, source,
            getattr(topic, "summary", ""),
            json.dumps(getattr(topic, "insights", [])),
        )

        # Upsert knowledge items
        all_items = getattr(topic, "all_items", [])
        for entry in all_items:
            item = entry.get("item", {})
            url = str(item.get("url") or "")
            text_content = str(item.get("text") or "")
            if not text_content and not url:
                continue

            item_id = str(
                item.get("id")
                or f"{safe_topic_id}_{source}_{item.get('createdAt', generated_at)}"
            )
            await self.conn.execute(
                f"""
                INSERT INTO {ITEMS_TABLE}
                  (user_id, topic_id, item_id, source, content_type, url,
                   author, origin, title, text_content, published_at,
                   discovered_at, run_id, score, item_summary, item_insights,
                   metrics, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$12)
                ON CONFLICT (user_id, topic_id, item_id) DO UPDATE
                  SET score        = EXCLUDED.score,
                      item_summary = EXCLUDED.item_summary,
                      item_insights= EXCLUDED.item_insights,
                      updated_at   = EXCLUDED.updated_at
                """,
                user_id, safe_topic_id, item_id,
                str(item.get("source") or source),
                infer_content_type(url),
                url,
                str(item.get("author") or ""),
                str(item.get("handle") or ""),
                truncate_text(text_content),
                text_content,
                item.get("createdAt") or generated_at,
                generated_at,
                report_id,
                float(entry.get("score") or 0),
                str(entry.get("summary") or truncate_text(text_content)),
                json.dumps(entry.get("insights") or []),
                json.dumps(item.get("metrics") or {}),
            )

        # Upsert run summary
        await self.conn.execute(
            f"""
            INSERT INTO {TOPIC_RUNS_TABLE}
              (user_id, topic_id, run_id, generated_at, source,
               matched_count, summary, insights)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            ON CONFLICT (user_id, topic_id, run_id) DO NOTHING
            """,
            user_id, safe_topic_id, report_id, generated_at, source,
            getattr(topic, "matched_count", 0),
            getattr(topic, "summary", ""),
            json.dumps(getattr(topic, "insights", [])),
        )

        # Update aggregate counts
        total_items = await self.conn.fetchval(
            f"SELECT COUNT(*) FROM {ITEMS_TABLE} WHERE user_id=$1 AND topic_id=$2",
            user_id, safe_topic_id,
        )
        total_runs = await self.conn.fetchval(
            f"SELECT COUNT(*) FROM {TOPIC_RUNS_TABLE} WHERE user_id=$1 AND topic_id=$2",
            user_id, safe_topic_id,
        )
        await self.conn.execute(
            f"""
            UPDATE {TOPICS_TABLE}
            SET total_knowledge_items=$3, total_runs_merged=$4
            WHERE user_id=$1 AND topic_id=$2
            """,
            user_id, safe_topic_id, total_items, total_runs,
        )

        return {"user_id": user_id, "topic_id": safe_topic_id, "total_knowledge_items": total_items}

    async def _fetch_topic_document(self, user_id: str, topic_id: str) -> dict | None:
        topic_row = await self.conn.fetchrow(
            f"SELECT * FROM {TOPICS_TABLE} WHERE user_id=$1 AND topic_id=$2 LIMIT 1",
            user_id, topic_id,
        )
        if not topic_row:
            return None

        summary_rows, item_rows = await asyncio.gather(
            self.conn.fetch(
                f"SELECT * FROM {TOPIC_RUNS_TABLE} WHERE user_id=$1 AND topic_id=$2 "
                f"ORDER BY generated_at DESC LIMIT 100",
                user_id, topic_id,
            ),
            self.conn.fetch(
                f"SELECT * FROM {ITEMS_TABLE} WHERE user_id=$1 AND topic_id=$2 "
                f"ORDER BY published_at DESC",
                user_id, topic_id,
            ),
        )

        return {
            "schema_version": 1,
            "user_id": topic_row["user_id"],
            "topic": {"id": topic_row["topic_id"], "label": topic_row["topic_label"] or topic_row["topic_id"]},
            "stats": {
                "total_knowledge_items": topic_row["total_knowledge_items"],
                "total_runs_merged": topic_row["total_runs_merged"],
                "first_merged_at": _iso(topic_row.get("first_merged_at")),
                "last_merged_at": _iso(topic_row.get("last_merged_at")),
                "latest_run_id": topic_row.get("latest_run_id"),
                "latest_source": topic_row.get("latest_source"),
            },
            "latest": {
                "summary": topic_row.get("latest_summary") or "",
                "insights": _parse_json(topic_row.get("latest_insights"), []),
            },
            "summaries": [
                {
                    "run_id": r["run_id"],
                    "generated_at": _iso(r.get("generated_at")),
                    "source": r.get("source") or "",
                    "matched_count": r["matched_count"],
                    "summary": r.get("summary") or "",
                    "insights": _parse_json(r.get("insights"), []),
                }
                for r in summary_rows
            ],
            "knowledge_items": [
                {
                    "id": i["item_id"],
                    "topic_id": i["topic_id"],
                    "source": i.get("source") or "",
                    "content_type": i.get("content_type") or "article",
                    "url": i.get("url") or "",
                    "author": i.get("author") or "",
                    "origin": i.get("origin") or "",
                    "title": i.get("title") or "",
                    "text": i.get("text_content") or "",
                    "published_at": _iso(i.get("published_at")),
                    "discovered_at": _iso(i.get("discovered_at")),
                    "run_id": i.get("run_id") or "",
                    "score": float(i.get("score") or 0),
                    "summary": i.get("item_summary") or truncate_text(i.get("text_content") or ""),
                    "insights": _parse_json(i.get("item_insights"), []),
                    "metrics": _parse_json(i.get("metrics"), {}),
                }
                for i in item_rows
            ],
        }

    async def _embed(self, text: str) -> list[float]:
        resp = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000],
        )
        return resp.data[0].embedding


# ------------------------------------------------------------------ #
# Module-level helpers                                                 #
# ------------------------------------------------------------------ #

import asyncio


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_json(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
