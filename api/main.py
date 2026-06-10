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
    "GROQ_API_KEY":    "Groq console → console.groq.com → API Keys (analyst + critic via llama-stack)",
    "GEMINI_API_KEY":  "Google AI Studio → aistudio.google.com → API Keys (writer iteration 1)",
    "MISTRAL_API_KEY": "Mistral console → console.mistral.ai → API Keys (writer rewrite + LinkedIn formatter)",
}
OPTIONAL_KEYS = {
    "GROQ_API_KEY_2":     "Second Groq key — llama-stack fallback",
    "GEMINI_API_KEY_2":   "Second Gemini key — fallback",
    "CEREBRA_KEY":        "Cerebras — cerebras.ai (llama-stack provider 3)",
    "OPENROUTER_API_KEY": "OpenRouter — openrouter.ai (llama-stack provider 4)",
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

_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
_cors_origins = ["*"] if _cors_raw.strip() == "*" else [o.strip() for o in _cors_raw.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_raw.strip() != "*",  # credentials forbidden with wildcard
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