"""
api/main.py — FastAPI application with SSE streaming, CORS, and share links.
"""
from __future__ import annotations

import logging
import os
import sys

# ── Load .env FIRST, before any other import that might read env vars ────────
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Startup key check ─────────────────────────────────────────────────────────
REQUIRED_KEYS = {

    "GROQ_API_KEY":    "Groq console → console.groq.com → API Keys",
    "MISTRAL_API_KEY": "Mistral console → console.mistral.ai → API Keys",
}
OPTIONAL_KEYS = {
    "GEMINI_API_KEY":   "Optional — add billing for higher limits",
    "OPENROUTER_API_KEY": "openrouter.ai (needed for DeepSeek R1 + Qwen writers)",
    "TAVILY_API_KEY":     "tavily.com (web search — 1000 free/month)",
    "EXA_API_KEY":        "exa.ai (semantic search — free tier)",
    "SUPABASE_URL":       "Supabase project URL (needed for persistence)",
    "SUPABASE_SERVICE_KEY": "Supabase service role key (needed for persistence)",
}

missing_required = [k for k in REQUIRED_KEYS if not os.getenv(k)]
missing_optional = [k for k in OPTIONAL_KEYS if not os.getenv(k)]

if missing_required:
    logger.error("=" * 60)
    logger.error("MISSING REQUIRED API KEYS — pipeline will fail without these:")
    for k in missing_required:
        logger.error("  ✗ %s → %s", k, REQUIRED_KEYS[k])
    logger.error("Add them to your .env file and restart.")
    logger.error("=" * 60)
else:
    logger.info("✓ All required API keys present")

if missing_optional:
    logger.warning("Missing optional keys (some features disabled): %s", missing_optional)

# ── App ───────────────────────────────────────────────────────────────────────
from api.routers import generate, validate, memory as memory_router

app = FastAPI(
    title="Voice Agent API",
    description="Agentic content generation with human-in-the-loop validation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router,      prefix="/api")
app.include_router(validate.router,      prefix="/api")
app.include_router(memory_router.router, prefix="/api")


@app.get("/health")
async def health():
    key_status = {k: bool(os.getenv(k)) for k in {**REQUIRED_KEYS, **OPTIONAL_KEYS}}
    return {"status": "ok", "version": "2.0.0", "keys": key_status}


# Silence noisy low-level HTTP debug loggers
import logging as _logging
for _noisy in ("hpack", "httpcore", "httpx", "h2"):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)