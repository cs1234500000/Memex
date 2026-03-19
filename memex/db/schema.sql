-- schema.sql
-- Memex unified database schema
-- Requires: PostgreSQL 15+ with pgvector extension
--
-- Two layers:
--   1. Research engine tables (runs, content_items, forum_logs, knowledge_chunks)
--   2. User knowledge tables  (news_users, news_user_topics, news_user_topic_runs,
--                              news_user_knowledge_items)
--      Migrated from supabase/news_knowledge_schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- LAYER 1: Query-driven research engine
-- ============================================================

-- One row per POST /analyze request
CREATE TABLE IF NOT EXISTS runs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query        TEXT NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at  TIMESTAMPTZ,
    status       TEXT NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running', 'done', 'error')),
    report_html  TEXT,
    report_pdf   TEXT,
    error_msg    TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs (started_at DESC);


-- Raw content items fetched by the four agents during a run
CREATE TABLE IF NOT EXISTS content_items (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id       UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    agent_type   TEXT NOT NULL
                   CHECK (agent_type IN ('news', 'social', 'expert', 'knowledge')),
    sub_query    TEXT,
    title        TEXT,
    url          TEXT,
    body         TEXT,
    source       TEXT,
    relevance    FLOAT,
    metadata     JSONB,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_content_items_run_id ON content_items (run_id);
CREATE INDEX IF NOT EXISTS idx_content_items_agent  ON content_items (agent_type);


-- One row per debate turn (Optimist / Pessimist / Realist / Host / moderator)
CREATE TABLE IF NOT EXISTS forum_logs (
    id           BIGSERIAL PRIMARY KEY,
    run_id       UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    turn_index   INT NOT NULL,
    persona      TEXT NOT NULL
                   CHECK (persona IN ('optimist', 'pessimist', 'realist', 'host', 'moderator')),
    content      TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forum_logs_run_id ON forum_logs (run_id, turn_index);


-- Offline corpus stored as pgvector embeddings (used by KnowledgeAgent)
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text         TEXT NOT NULL,
    embedding    vector(1536),           -- text-embedding-3-small
    title        TEXT,
    source_url   TEXT,
    concept      TEXT,
    scenarios    TEXT,
    caveats      TEXT,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
    ON knowledge_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);


-- ============================================================
-- LAYER 2: User knowledge / passive ingest pipeline
-- Migrated from supabase/news_knowledge_schema.sql
-- ============================================================

-- Registered users (maps to news_users in Supabase)
CREATE TABLE IF NOT EXISTS news_users (
    user_id       TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL DEFAULT '',
    email         TEXT NOT NULL DEFAULT '',
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_run_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- One row per (user, topic) pair — aggregated stats + latest snapshot
CREATE TABLE IF NOT EXISTS news_user_topics (
    user_id               TEXT NOT NULL REFERENCES news_users(user_id) ON DELETE CASCADE,
    topic_id              TEXT NOT NULL,
    topic_label           TEXT NOT NULL DEFAULT '',
    total_knowledge_items INTEGER NOT NULL DEFAULT 0,
    total_runs_merged     INTEGER NOT NULL DEFAULT 0,
    first_merged_at       TIMESTAMPTZ,
    last_merged_at        TIMESTAMPTZ,
    latest_run_id         TEXT,
    latest_source         TEXT,
    latest_summary        TEXT NOT NULL DEFAULT '',
    latest_insights       JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_news_topics_user
    ON news_user_topics (user_id);


-- One row per pipeline run for a (user, topic) pair
CREATE TABLE IF NOT EXISTS news_user_topic_runs (
    user_id       TEXT NOT NULL REFERENCES news_users(user_id) ON DELETE CASCADE,
    topic_id      TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    generated_at  TIMESTAMPTZ NOT NULL,
    source        TEXT NOT NULL,
    matched_count INTEGER NOT NULL DEFAULT 0,
    summary       TEXT NOT NULL DEFAULT '',
    insights      JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, topic_id, run_id),
    FOREIGN KEY (user_id, topic_id)
        REFERENCES news_user_topics (user_id, topic_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_news_runs_user_topic_generated
    ON news_user_topic_runs (user_id, topic_id, generated_at DESC);


-- Individual knowledge items discovered by the ingest pipeline
CREATE TABLE IF NOT EXISTS news_user_knowledge_items (
    user_id       TEXT NOT NULL REFERENCES news_users(user_id) ON DELETE CASCADE,
    topic_id      TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    content_type  TEXT NOT NULL DEFAULT 'article',
    url           TEXT NOT NULL DEFAULT '',
    author        TEXT NOT NULL DEFAULT '',
    origin        TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    text_content  TEXT NOT NULL DEFAULT '',
    published_at  TIMESTAMPTZ,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id        TEXT NOT NULL DEFAULT '',
    score         DOUBLE PRECISION NOT NULL DEFAULT 0,
    item_summary  TEXT NOT NULL DEFAULT '',
    item_insights JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics       JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, topic_id, item_id),
    FOREIGN KEY (user_id, topic_id)
        REFERENCES news_user_topics (user_id, topic_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_news_items_user_topic_published
    ON news_user_knowledge_items (user_id, topic_id, published_at DESC);


-- ============================================================
-- LAYER 3: User subscriptions config
-- ============================================================

CREATE TABLE IF NOT EXISTS user_subscriptions (
    user_id    TEXT PRIMARY KEY REFERENCES news_users(user_id) ON DELETE CASCADE,
    config     JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- LAYER 4: Waitlist signups (from Supabase waitlist_signups)
-- ============================================================

CREATE TABLE IF NOT EXISTS waitlist_signups (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL DEFAULT '',
    email      TEXT NOT NULL UNIQUE,
    interest   TEXT NOT NULL DEFAULT 'waitlist',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_waitlist_email ON waitlist_signups (email);
