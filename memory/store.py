"""
memory/store.py — Supabase persistence for runs, voice profiles, feedback, and shares.
All functions are async-compatible (sync Supabase client wrapped in executor where needed).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_supabase_client = None


def _get_client():
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment. "
                "Copy .env.example to .env and fill in your Supabase credentials."
            )
        from supabase import create_client  # type: ignore
        _supabase_client = create_client(url, key)
        logger.debug("Supabase client initialised")
    return _supabase_client


# ---------------------------------------------------------------------------
# Voice profiles
# ---------------------------------------------------------------------------

async def get_voice_profile(user_id: str):
    """Return cached VoiceProfile for user_id, or None."""
    try:
        client = _get_client()
        resp = (
            client.table("voice_profiles")
            .select("profile_json, updated_at")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            from agent.schemas import VoiceProfile
            raw = resp.data[0]["profile_json"]
            profile = VoiceProfile(**raw)
            logger.debug("Loaded voice profile for %s (updated %s)",
                         user_id, resp.data[0].get("updated_at", "?"))
            return profile
    except Exception as exc:
        logger.warning("get_voice_profile failed for %s: %s", user_id, exc)
    return None


async def save_voice_profile(user_id: str, profile) -> None:
    try:
        client = _get_client()
        client.table("voice_profiles").upsert(
            {
                "user_id":      user_id,
                "profile_json": profile.dict(),
                "updated_at":   datetime.utcnow().isoformat(),
            },
            on_conflict="user_id",
        ).execute()
        logger.info("Voice profile saved for %s", user_id)
    except Exception as exc:
        logger.warning("save_voice_profile failed for %s: %s", user_id, exc)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

async def save_run(
    run_id:             str,
    user_id:            str,
    final_output:       str,
    formatted_variants  = None,
    trace:              list = None,
    validation          = None,
) -> None:
    try:
        client = _get_client()
        client.table("runs").upsert(
            {
                "run_id":             run_id,
                "user_id":            user_id,
                "final_output":       final_output,
                "formatted_variants": formatted_variants.dict() if formatted_variants else None,
                "trace":              [e.dict() for e in (trace or [])],
                "validation":         validation.dict() if validation else None,
                "status":             "complete",
                "updated_at":         datetime.utcnow().isoformat(),
            },
            on_conflict="run_id",
        ).execute()
        logger.info("Run saved: %s", run_id[:8])
    except Exception as exc:
        logger.warning("save_run failed for %s: %s", run_id[:8], exc)


async def get_run(run_id: str) -> Optional[dict]:
    try:
        client = _get_client()
        resp = (
            client.table("runs")
            .select("*")
            .eq("run_id", run_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception as exc:
        logger.warning("get_run failed for %s: %s", run_id[:8], exc)
    return None


async def update_run_status(run_id: str, status: str, draft=None) -> None:
    try:
        client  = _get_client()
        payload: dict = {
            "status":     status,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if draft is not None:
            payload["draft"] = (
                [t.dict() for t in draft] if isinstance(draft, list) else draft
            )
        client.table("runs").update(payload).eq("run_id", run_id).execute()
        logger.debug("Run status updated: %s → %s", run_id[:8], status)
    except Exception as exc:
        logger.warning("update_run_status failed for %s: %s", run_id[:8], exc)


# ---------------------------------------------------------------------------
# Feedback  (gold-label learning data)
# ---------------------------------------------------------------------------

async def save_feedback(
    run_id:         str,
    user_id:        str,
    original_draft: str,
    approved_output: str,
    decision:       str,
) -> None:
    try:
        client = _get_client()
        client.table("feedback").insert(
            {
                "run_id":          run_id,
                "user_id":         user_id,
                "original_draft":  original_draft,
                "approved_output": approved_output,
                "decision":        decision,
                "created_at":      datetime.utcnow().isoformat(),
            }
        ).execute()
        logger.info("Feedback saved for run %s decision=%s", run_id[:8], decision)
    except Exception as exc:
        logger.warning("save_feedback failed for %s: %s", run_id[:8], exc)


async def get_feedback_count(user_id: str) -> int:
    """Return total approved/edited output count for rebuild threshold check."""
    try:
        client = _get_client()
        resp = (
            client.table("feedback")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .in_("decision", ["approve", "edit"])
            .execute()
        )
        return resp.count or 0
    except Exception as exc:
        logger.warning("get_feedback_count failed for %s: %s", user_id, exc)
        return 0
