"""
api/main.py
FastAPI application — single entrypoint for the Python backend.

Routes
------
Query-driven research:
  POST /analyze                   → run multi-agent pipeline, return HTML or JSON report
  GET  /runs/{run_id}             → retrieve a previous run from DB

Passive ingest pipeline:
  POST /api/news/run              → run ingest pipeline for a user
  GET  /api/news/latest           → most recent ingest report (DB)

Subscriptions:
  GET  /api/news/subscriptions    → get current subscriptions config (DB)
  PUT  /api/news/subscriptions    → update subscriptions config (DB)

Knowledge:
  GET  /api/news/knowledge        → user topic knowledge document(s) (DB)

Users:
  POST /api/news/users/signup     → create / upsert a news user (DB)
  POST /api/news/users/signin     → look up a user by ID + email (DB)

Utilities:
  GET  /health                    → liveness check
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (two levels up from memex/api/)
load_dotenv(Path(__file__).parents[2] / ".env")
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from openai import AsyncOpenAI

from memex.api.schemas import (
    AnalyzeRequest,
    RunNewsRequest,
    Subscriptions,
    UserSigninRequest,
    UserSignupRequest,
)
from memex.core.orchestrator import Orchestrator, RunResult
from memex.ingest.pipeline import DEFAULT_SUBSCRIPTIONS, IngestPipeline
from memex.knowledge.store import KnowledgeStore

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    db_conn = None
    dsn = os.environ.get("POSTGRES_DSN")
    if dsn:
        import ssl as _ssl
        import asyncpg
        # asyncpg doesn't support ?ssl=require in the URL — pass ssl separately.
        # Supabase uses a self-signed chain, so skip cert verification.
        clean_dsn = dsn.replace("?ssl=require", "").replace("&ssl=require", "")
        ssl_ctx = None
        if "ssl=require" in dsn:
            ssl_ctx = _ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = _ssl.CERT_NONE
        db_conn = await asyncpg.connect(clean_dsn, ssl=ssl_ctx)

    from memex.config import EXPERT_MODEL
    model = os.environ.get("LLM_MODEL", EXPERT_MODEL)

    _state["db"] = db_conn
    _state["orchestrator"] = Orchestrator(client=client, db_conn=db_conn, model=model)
    _state["pipeline"] = IngestPipeline(client=client, db_conn=db_conn)
    _state["knowledge_store"] = KnowledgeStore(conn=db_conn, client=client)

    yield

    if db_conn:
        await db_conn.close()


app = FastAPI(
    title="Memex Research API",
    version="2.0.0",
    description="Query-driven multi-agent research engine + passive news ingest pipeline",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user_id(
    x_user_id: str = Header("default", alias="x-user-id"),
    user_id: str = Query("default"),
) -> str:
    return (x_user_id or user_id or "default").strip()


def _require_db(label: str = "This endpoint"):
    store = _state.get("knowledge_store")
    if store is None or store.conn is None:
        raise HTTPException(503, f"{label} requires a DB connection (set POSTGRES_DSN)")
    return store


def _report_to_dict(report) -> dict:
    return {
        "id": report.id,
        "generatedAt": report.generated_at,
        "source": report.source,
        "subscriptions": report.subscriptions,
        "totalFetchedItems": report.total_fetched_items,
        "warnings": report.warnings,
        "userId": report.user_id,
        "topics": [
            {
                "topicId": t.topic_id,
                "topicLabel": t.topic_label,
                "matchedCount": t.matched_count,
                "summary": t.summary,
                "insights": t.insights,
                "topItems": t.top_items,
            }
            for t in report.topics
        ],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Query-driven research
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the full multi-agent pipeline for a research query."""
    orchestrator: Orchestrator | None = _state.get("orchestrator")
    if orchestrator is None:
        raise HTTPException(503, "Orchestrator not initialised")

    result: RunResult = await orchestrator.run(req.query)

    if req.format == "html":
        return HTMLResponse(content=result.report_html)

    return JSONResponse({
        "run_id": result.run_id,
        "query": result.query,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat() if result.finished_at else None,
        "report_html": result.report_html,
        "forum_turns": len(result.forum_log),
    })


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    store: KnowledgeStore = _require_db("GET /runs")
    raise HTTPException(501, "Not yet implemented")


# ---------------------------------------------------------------------------
# Passive ingest pipeline
# ---------------------------------------------------------------------------

@app.post("/api/news/run")
async def run_news(
    req: RunNewsRequest,
    x_user_id: str = Header("", alias="x-user-id"),
):
    """Run the passive ingest pipeline for a user. Persists results to DB."""
    pipeline: IngestPipeline | None = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")

    store: KnowledgeStore = _require_db("POST /api/news/run")

    user_id = (x_user_id or req.user_id or "default").strip()
    subs = await _load_subscriptions(store)
    report = await pipeline.run(user_id=user_id, subscriptions=subs)
    await store.merge_report(user_id, report)

    return _report_to_dict(report)


@app.get("/api/news/latest")
async def get_latest_news(user_id: str = Depends(_get_user_id)):
    """Return the most recent ingest report for a user from DB."""
    store: KnowledgeStore = _require_db("GET /api/news/latest")
    data = await store.get_latest_report(user_id)
    if data is None:
        raise HTTPException(404, "No report found. Run POST /api/news/run first.")
    return data


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

@app.get("/api/news/subscriptions")
async def get_subscriptions(user_id: str = Depends(_get_user_id)):
    store: KnowledgeStore = _require_db("GET /api/news/subscriptions")
    return await _load_subscriptions(store, user_id)


@app.put("/api/news/subscriptions")
async def update_subscriptions(
    payload: Subscriptions,
    user_id: str = Depends(_get_user_id),
):
    store: KnowledgeStore = _require_db("PUT /api/news/subscriptions")
    subs = payload.model_dump(by_alias=False)
    await store.save_subscriptions(user_id, subs)
    return subs


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------

@app.get("/api/news/knowledge")
async def get_knowledge(
    topic_id: str = Query(""),
    user_id: str = Depends(_get_user_id),
):
    """Return topic knowledge document(s) for a user."""
    store: KnowledgeStore = _require_db("GET /api/news/knowledge")
    data = await store.get_user_topic_knowledge(user_id, topic_id.strip() or None)
    if topic_id and data is None:
        raise HTTPException(404, "No knowledge found for this topic.")
    return {"user_id": user_id, "topic_id": topic_id or None, "data": data}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

@app.post("/api/news/users/signup")
async def news_user_signup(req: UserSignupRequest):
    store: KnowledgeStore = _require_db("POST /api/news/users/signup")
    user = await store.upsert_user(req.user_id, name=req.name, email=req.email)
    return {"ok": True, "user": user}


@app.post("/api/news/users/signin")
async def news_user_signin(req: UserSigninRequest):
    store: KnowledgeStore = _require_db("POST /api/news/users/signin")
    user = await store.find_user(user_id=req.user_id, email=req.email)
    if not user:
        raise HTTPException(404, "User not found.")
    return {"ok": True, "user": user}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _load_subscriptions(store: KnowledgeStore, user_id: str = "default") -> dict:
    saved = await store.get_subscriptions(user_id)
    return saved or DEFAULT_SUBSCRIPTIONS
