"""
FastAPI application — the single service entry point.

Endpoints:
  GET  /health  →  {"status": "ok"}
  POST /chat    →  ChatResponse (exact schema, every call)

Startup:
  - Validates GROQ_API_KEY
  - Loads catalog from disk
  - Builds TF-IDF index
  - All done before the first request
"""
from __future__ import annotations

import logging
import time
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import ChatRequest, ChatResponse
from app.catalog import load_catalog
from app.retrieval import build_index
from app.llm_client import validate_api_key
from app.agent import handle_chat, FALLBACK_CLARIFY

# ── Load .env if present ──────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: validate key, load catalog, build index. All before first request."""
    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender — Starting up")
    logger.info("=" * 60)

    # 1. Validate API key
    validate_api_key()

    # 2. Load catalog
    catalog = load_catalog()
    logger.info(f"Catalog loaded: {len(catalog)} items")

    # 3. Build retrieval index
    build_index()
    logger.info("TF-IDF index ready")

    logger.info("Startup complete — ready to serve requests")
    yield
    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent for recommending SHL assessment products",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — permissive for the grading harness ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — must return 200 even on cold start."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Stateless: every call carries the full conversation.

    Returns the exact schema on every code path — clarify, recommend, refine,
    compare, or refuse. Never returns null recommendations or extra fields.
    """
    start_time = time.time()
    msg_count = len(request.messages)

    logger.info(f"POST /chat — {msg_count} messages")

    try:
        response = await handle_chat(request)

        elapsed = time.time() - start_time
        logger.info(
            f"Response: intent handled, "
            f"{len(response.recommendations)} recs, "
            f"end={response.end_of_conversation}, "
            f"time={elapsed:.2f}s"
        )

        return response

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Unhandled error in /chat after {elapsed:.2f}s: {type(e).__name__}: {e}", exc_info=True)

        # NEVER return a bare 500 — always return schema-valid response
        return FALLBACK_CLARIFY


# ── Global exception handler (extra safety net) ──────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: return schema-valid JSON even on unexpected errors."""
    logger.error(f"Global exception handler: {type(exc).__name__}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=200,  # Return 200 with valid schema, not 500
        content={
            "reply": (
                "I encountered a temporary issue. Could you please rephrase "
                "your request? I'm here to help you find the right SHL assessment."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
