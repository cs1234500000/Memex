# News Intelligence Backend (Multi-Source Ready)

## Architecture

- `src/news/sources/index.js`
  - Source router that selects the configured source adapter.
  - Keeps ingest logic extensible for multiple platforms.
- `src/news/sources/newsApiSource.js`
  - NewsAPI adapter for live article ingestion.
  - Uses `NEWSAPI_API_KEY` and fetches via `/v2/everything`.
  - Best-effort mode: if one interest query fails, pipeline continues.
- `src/news/sources/twitterApiSource.js`
  - Twitter API v2 adapter for tracked handles.
  - Uses `TWITTER_BEARER_TOKEN` and `/2/tweets/search/recent`.
- `src/news/subscriptions.js`
  - Loads/saves user interests and source configs.
  - Keeps compatibility fields such as `twitterHandles` and `maxTweetsPerRun`.
  - Stored in `data/news/subscriptions.json`.
- `src/news/relevance.js`
  - Scores ingested items with local embedding similarity plus keyword/engagement weighting.
  - Falls back to keyword-only mode if embeddings are disabled or unavailable.
  - Groups ranked matches by topic.
- `src/news/summarizer.js`
  - Produces summary + insights.
  - Uses OpenAI when `OPENAI_API_KEY` exists, otherwise heuristic fallback.
- `src/news/knowledgeStore.js`
  - Persists per-user, per-topic knowledge into Supabase.
  - Upserts users, topic state, topic run summaries, and topic knowledge items.
- `src/news/pipeline.js`
  - Orchestrates ingest -> filter -> summarize -> persist.
  - Persists reports to `data/news/reports/*.json` and `data/news/latest.json`.
- `src/api/newsHandlers.js`
  - API handlers consumed by `server.js`.

## API Endpoints

- `GET /api/news/subscriptions`
- `PUT /api/news/subscriptions`
- `POST /api/news/run`
- `GET /api/news/latest`
- `GET /api/news/knowledge?userId=<id>[&topicId=<topic>]`
- `POST /api/news/users/signup`
- `POST /api/news/users/signin`
- `GET /dashboard.html` (user dashboard UI)

## Local Usage

1. Start server:
   - `npm start`
2. Run pipeline:
   - `curl -X POST "http://localhost:3000/api/news/run?userId=demo_user"`
3. Read latest report:
   - `curl http://localhost:3000/api/news/latest`
4. Read merged user knowledge:
   - all topics: `curl "http://localhost:3000/api/news/knowledge?userId=demo_user"`
   - single topic: `curl "http://localhost:3000/api/news/knowledge?userId=demo_user&topicId=ai_startups"`
5. Open dashboard:
   - `http://localhost:3000/dashboard.html`

## Environment Variables

- `NEWSAPI_API_KEY` (required for NewsAPI ingest)
- `TWITTER_BEARER_TOKEN` (required when `source` is `twitter`)
- `OPENAI_API_KEY` (optional, enables LLM summarization)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (required for knowledge storage)
- `NEWS_USE_LOCAL_EMBEDDINGS` (optional, default `true`; set `false` to force keyword-only relevance)
- `NEWS_LOCAL_EMBEDDING_MODEL` (optional, default `fast-all-MiniLM-L6-v2`)
- `NEWS_LOCAL_EMBEDDING_CACHE_DIR` (optional, default `local_cache`)
- `NEWS_SEMANTIC_MIN_SIMILARITY` (optional, default `0.2`; low-sim items without keyword hits are filtered)
- `NEWS_SEMANTIC_WEIGHT` and `NEWS_KEYWORD_WEIGHT` (optional score blend controls; defaults `20` and `8`)

## Supabase Tables

Apply `supabase/news_knowledge_schema.sql` in Supabase SQL editor.

- `news_users`
  - General user table with `user_id`, profile fields, run timestamps.
- `news_user_topics`
  - Per-user/per-topic state table (latest summary, insights, totals).
- `news_user_topic_runs`
  - Run history table (summary + full insights per run).
- `news_user_knowledge_items`
  - All merged relevant knowledge items per user/topic.

Design note: instead of creating a physical table per user (which does not scale and is hard to manage), each table is scoped by `user_id`, giving each user an isolated logical dataset with strong keys and indexes.

## Per-User Knowledge Schema

Knowledge is persisted in Supabase and materialized into this logical shape by `GET /api/news/knowledge`.

```json
{
  "schemaVersion": 1,
  "userId": "demo_user",
  "topic": { "id": "ai_startups", "label": "AI startup companies" },
  "stats": {
    "totalKnowledgeItems": 123,
    "totalRunsMerged": 9,
    "firstMergedAt": "2026-03-13T23:00:56.499Z",
    "lastMergedAt": "2026-03-14T01:10:11.000Z",
    "latestRunId": "2026-03-14T01-10-11-000Z",
    "latestSource": "newsapi"
  },
  "latest": {
    "summary": "Topic-level summary for latest run",
    "insights": ["Insight 1", "Insight 2", "Insight 3", "Insight N"]
  },
  "summaries": [
    {
      "runId": "2026-03-14T01-10-11-000Z",
      "generatedAt": "2026-03-14T01:10:11.000Z",
      "source": "newsapi",
      "matchedCount": 15,
      "summary": "Run-level summary",
      "insights": ["..."]
    }
  ],
  "knowledgeItems": [
    {
      "id": "stable-item-id",
      "topicId": "ai_startups",
      "source": "newsapi",
      "contentType": "article",
      "url": "https://example.com/news",
      "author": "Author Name",
      "origin": "source_handle_or_domain",
      "title": "Short preview text",
      "text": "Full text used for relevance/summarization",
      "publishedAt": "2026-03-12T22:58:10Z",
      "discoveredAt": "2026-03-14T01:10:11.000Z",
      "runId": "2026-03-14T01-10-11-000Z",
      "score": 12.5,
      "summary": "Per-item summary",
      "insights": ["Per-item insight A", "Per-item insight B", "Per-item insight N"],
      "metrics": { "likeCount": 0, "repostCount": 0, "replyCount": 0 }
    }
  ]
}
```

Notes:
- `contentType` supports `article`, `text`, `audio`, `video` (currently inferred from source URL).
- `insights` are not capped at 3; any number can be returned and persisted.
- Merge behavior is idempotent by item `id`; existing items are updated rather than duplicated.

