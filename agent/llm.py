"""
agent/llm.py — Multi-provider LLM helpers.
All functions: async, httpx, 60s timeout, return plain str (empty on total failure).
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Llama stack — OpenAI-compatible endpoints, shared request builder
# ---------------------------------------------------------------------------

_LLAMA_PROVIDERS = [
    {
        "name":    "groq-key1",
        "base":    "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model":   "llama-3.3-70b-versatile",
    },
    {
        "name":    "groq-key2",
        "base":    "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY_2",
        "model":   "llama-3.3-70b-versatile",
    },
    {
        "name":    "cerebras",
        "base":    "https://api.cerebras.ai/v1",
        "key_env": "CEREBRA_KEY",
        "model":   "llama-3.3-70b",
    },
    {
        "name":    "openrouter",
        "base":    "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model":   "meta-llama/llama-3.3-70b-instruct",
    },
]


async def call_llama_stack(prompt: str, system: str = "", max_tokens: int = 600) -> str:
    """Try Groq key1 → Groq key2 → Cerebras → OpenRouter. No sleep on 429 — just next provider."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for p in _LLAMA_PROVIDERS:
        key = os.getenv(p["key_env"], "")
        if not key:
            continue
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{p['base']}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": p["model"], "messages": messages, "max_tokens": max_tokens},
                )
            if resp.status_code == 429:
                logger.warning("llama-stack %s 429 — next provider", p["name"])
                continue
            if not resp.is_success:
                logger.error("llama-stack %s HTTP %s: %s", p["name"], resp.status_code, resp.text[:300])
                continue
            text = resp.json()["choices"][0]["message"]["content"].strip()
            logger.debug(
                "llama-stack %s OK — len=%d preview=%r",
                p["name"], len(text), text[:200],
            )
            return text
        except Exception as exc:
            logger.error("llama-stack %s exception: %s", p["name"], exc)
            continue

    logger.error("llama-stack: all providers failed")
    return ""


# ---------------------------------------------------------------------------
# Gemini — generateContent format
# ---------------------------------------------------------------------------

_GEMINI_MODEL = "gemini-2.0-flash"
_GEMINI_BASE  = "https://generativelanguage.googleapis.com/v1beta/models"


async def call_gemini(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Gemini key1 → key2. Returns empty string on total failure."""
    keys = [k for k in [os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_API_KEY_2", "")] if k]
    if not keys:
        logger.warning("call_gemini: no GEMINI_API_KEY set")
        return ""

    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.8},
    }
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}

    for key in keys:
        try:
            url = f"{_GEMINI_BASE}/{_GEMINI_MODEL}:generateContent?key={key}"
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=body)
            if resp.status_code == 429:
                logger.warning("call_gemini key ...%s 429 — next key", key[-4:])
                continue
            if not resp.is_success:
                logger.error("call_gemini HTTP %s: %s", resp.status_code, resp.text[:300])
                continue
            candidates = resp.json().get("candidates", [])
            if not candidates:
                logger.warning("call_gemini: no candidates — %s", resp.json().get("promptFeedback", ""))
                continue
            text = candidates[0]["content"]["parts"][0]["text"].strip()
            logger.debug(
                "call_gemini key ...%s OK — len=%d preview=%r",
                key[-4:], len(text), text[:200],
            )
            return text
        except Exception as exc:
            logger.error("call_gemini exception: %s", exc)
            continue

    logger.error("call_gemini: all keys failed")
    return ""


# ---------------------------------------------------------------------------
# Mistral — OpenAI-compatible
# ---------------------------------------------------------------------------

_MISTRAL_MODEL = "mistral-small-latest"
_MISTRAL_URL   = "https://api.mistral.ai/v1/chat/completions"


async def call_mistral(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Single Mistral key. Returns empty string on failure."""
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        logger.warning("call_mistral: no MISTRAL_API_KEY set")
        return ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                _MISTRAL_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": _MISTRAL_MODEL, "messages": messages, "max_tokens": max_tokens},
            )
        if resp.status_code == 429:
            logger.warning("call_mistral 429: %s", resp.text[:200])
            return ""
        if not resp.is_success:
            logger.error("call_mistral HTTP %s: %s", resp.status_code, resp.text[:300])
            return ""
        text = resp.json()["choices"][0]["message"]["content"].strip()
        logger.debug("call_mistral OK — len=%d preview=%r", len(text), text[:200])
        return text
    except Exception as exc:
        logger.error("call_mistral exception: %s", exc)
        return ""
