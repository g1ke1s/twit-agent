"""
agent/memory.py — Per-run context cache and user instruction memory.
JSON file-backed with atomic writes. Falls back gracefully if disk is unavailable.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR  = Path("memory")
_RUN_CACHE  = _CACHE_DIR / "run_cache.json"
_USER_PREFS = _CACHE_DIR / "user_prefs.json"

# Entries older than this are evicted on next save
_RUN_TTL_SECONDS = 3600 * 4  # 4 hours


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("memory: read failed for %s: %s", path.name, exc)
    return {}


def _save(path: Path, data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)   # atomic rename
    except Exception as exc:
        logger.warning("memory: write failed for %s: %s", path.name, exc)


# ---------------------------------------------------------------------------
# Run context cache
# Stores researched sources + extracted claims so rewrites reuse the same
# grounded facts without re-searching or re-embedding.
# ---------------------------------------------------------------------------

def save_run_context(
    run_id:  str,
    sources: list[dict],   # serialised Source.model_dump() dicts
    claims:  list[dict],   # serialised Claim.model_dump() dicts
    thesis:  str,
) -> None:
    cache = _load(_RUN_CACHE)
    now = time.time()
    # Evict stale entries
    cache = {k: v for k, v in cache.items()
             if now - v.get("ts", 0) < _RUN_TTL_SECONDS}
    cache[run_id] = {"ts": now, "sources": sources, "claims": claims, "thesis": thesis}
    _save(_RUN_CACHE, cache)
    logger.debug("memory: saved context for run %s (%d sources, %d claims)",
                 run_id[:8], len(sources), len(claims))


def get_run_context(run_id: str) -> Optional[dict]:
    """Return cached context dict or None if absent/expired."""
    entry = _load(_RUN_CACHE).get(run_id)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > _RUN_TTL_SECONDS:
        return None
    return entry


# ---------------------------------------------------------------------------
# User instruction memory
# Persists per-user preferences (tone, forbidden phrases, citation style)
# across runs so they are applied automatically without re-asking.
# ---------------------------------------------------------------------------

_PREF_DEFAULTS: dict = {
    "forbidden_phrases": [],
    "tone":              "",          # "analytical" | "provocative" | "educational"
    "citation_style":    "inline",    # "inline" | "footnote"
    "notes":             "",          # free-form instruction text
}


def get_user_prefs(user_id: str) -> dict:
    """Return merged prefs (saved + defaults). Always safe to call."""
    saved = _load(_USER_PREFS).get(user_id, {})
    return {**_PREF_DEFAULTS, **saved}


def merge_user_prefs(user_id: str, new_prefs: dict) -> None:
    """Non-destructive update — lists are unioned, scalars overwritten."""
    store = _load(_USER_PREFS)
    existing = store.get(user_id, {})
    if "forbidden_phrases" in new_prefs:
        combined = list(dict.fromkeys(
            existing.get("forbidden_phrases", []) + new_prefs["forbidden_phrases"]
        ))
        new_prefs = {**new_prefs, "forbidden_phrases": combined}
    store[user_id] = {**existing, **new_prefs}
    _save(_USER_PREFS, store)
    logger.debug("memory: saved prefs for user %s", user_id)
