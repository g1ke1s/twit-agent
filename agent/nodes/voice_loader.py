"""
agent/nodes/voice_loader.py — Layer 1: Load or build VoiceProfile for user.
"""
from __future__ import annotations

import logging
import time

from agent.schemas import AgentState, TraceEvent
from agent.voice_profile import build_voice_profile

logger = logging.getLogger(__name__)

# Fallback minimal profile when no archive is provided
_DEFAULT_STYLE_SUMMARY = (
    "Clear, direct writing with strong opinions backed by data. "
    "Favours short paragraphs and concrete examples. "
    "Never uses passive voice or hedge words."
)


async def voice_loader_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    user_id = state.get("user_id", "default")

    # 1. Check memory store for cached profile
    try:
        from memory.store import get_voice_profile, save_voice_profile
        cached = await get_voice_profile(user_id)
    except Exception as exc:
        logger.warning("Memory store unavailable: %s", exc)
        cached = None
        save_voice_profile = None  # type: ignore

    if cached:
        duration_ms = (time.monotonic() - t0) * 1000
        event = TraceEvent(
            node="voice_loader",
            model="cache",
            duration_ms=duration_ms,
            detail=f"Loaded cached VoiceProfile for user {user_id}",
        )
        return {"voice_profile": cached, "status": "voice_loaded", "trace": [event]}

    # 2. Check if voice archive text is available in input_context
    input_ctx = state.get("input_context")
    archive_texts: list[str] = []
    if input_ctx and input_ctx.uploaded_content:
        # Split by newlines or paragraph breaks for individual posts
        raw = input_ctx.uploaded_content
        archive_texts = [p.strip() for p in raw.split("\n\n") if p.strip()]

    if archive_texts:
        try:
            profile = await build_voice_profile(archive_texts, user_id)
            if save_voice_profile:
                await save_voice_profile(user_id, profile)
            detail = f"Built new VoiceProfile from {len(archive_texts)} samples"
        except Exception as exc:
            logger.error("build_voice_profile failed: %s", exc)
            profile = _minimal_profile()
            detail = "Fallback minimal VoiceProfile (build failed)"
    else:
        # No archive: use minimal profile
        profile = _minimal_profile()
        detail = "No archive provided — using minimal VoiceProfile"

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="voice_loader",
        model="gemini-2.0-flash-light",
        duration_ms=duration_ms,
        detail=detail,
    )
    return {"voice_profile": profile, "status": "voice_loaded", "trace": [event]}


def _minimal_profile():
    from agent.schemas import VoiceProfile
    import os
    return VoiceProfile(
        avg_sentence_length=14.0,
        vocabulary_richness=0.55,
        preferred_structures=[
            "Opens with a specific data point",
            "Uses em-dash for asides",
            "Ends with a provocative question",
        ],
        forbidden_phrases=[
            "in conclusion", "it's worth noting", "game-changer",
            "dive into", "delve", "revolutionize", "seamless", "leverage",
        ],
        style_summary=_DEFAULT_STYLE_SUMMARY,
        few_shot_examples=[
            "Most founders wait too long to fire. The cost isn't just performance — it's the 3 people who quit because they had to work around the problem.",
            "73% of Series A companies never reach Series B. The bottleneck is almost never product. It's distribution, almost every time.",
        ],
        faiss_index_path=os.path.join("memory", "faiss", "default.index"),
    )