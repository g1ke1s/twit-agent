"""
agent/nodes/critic.py — Layer 6: Local scoring, zero LLM calls.
Scores derived from heuristics to preserve free tier quota.
"""
from __future__ import annotations

import logging
import re
import time

from agent.schemas import AgentState, CritiqueResult, Source, TraceEvent, VoiceProfile
from agent.voice_profile import get_voice_similarity

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2
APPROVE_THRESHOLD = 6.5  # Lowered — local scoring is conservative

BANNED_RE = re.compile(
    r"\b(in conclusion|it's worth noting|as an ai|game.changer|dive into|delve|"
    r"revolutionize|simply put|touch base|move the needle|cutting.edge|seamless|leverage)\b",
    re.IGNORECASE,
)


def _score_draft(draft, profile: VoiceProfile, sources: list[Source]) -> dict:
    """Heuristic scoring — no LLM calls needed."""
    if isinstance(draft, list):
        full_text = " ".join(t.text for t in draft)
        tweet_count = len(draft)
    else:
        full_text = draft or ""
        tweet_count = 1

    # Hook strength: does first tweet have a number or specific claim?
    first = (draft[0].text if isinstance(draft, list) and draft else full_text[:280])
    has_number = bool(re.search(r"\d+", first))
    has_question_mark = "?" in first
    hook_strength = 7.0 if has_number else (6.5 if has_question_mark else 5.5)

    # Voice match via FAISS
    if profile:
        sim = get_voice_similarity(full_text[:500], profile)
        voice_match = round(1.0 + sim * 9.0, 1)
    else:
        voice_match = 6.0

    # Insight density: avg words per tweet (threads), or word count (essays)
    if isinstance(draft, list) and draft:
        avg_words = sum(len(t.text.split()) for t in draft) / len(draft)
        insight_density = min(9.0, max(4.0, avg_words / 5.0))
    else:
        word_count = len(full_text.split())
        insight_density = min(9.0, max(4.0, word_count / 100.0))

    # Clarity: penalise long sentences
    sentences = re.split(r"[.!?]+", full_text)
    avg_sent_len = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
    clarity = 8.0 if avg_sent_len < 20 else (7.0 if avg_sent_len < 30 else 6.0)

    # Cliche flags
    cliche_flags = list(set(m.lower() for m in BANNED_RE.findall(full_text)))

    # Hallucination flags: tweets with no claim_ids (threads only)
    hallucination_flags = []
    if isinstance(draft, list):
        for t in draft:
            if not t.is_hook and not t.is_cta and not t.claim_ids:
                hallucination_flags.append(f"Tweet {t.position}: no source citation")

    average_score = (hook_strength + voice_match + insight_density + clarity) / 4.0

    return {
        "hook_strength": round(hook_strength, 1),
        "voice_match": round(voice_match, 1),
        "insight_density": round(insight_density, 1),
        "clarity": round(clarity, 1),
        "hallucination_flags": hallucination_flags[:3],
        "cliche_flags": cliche_flags[:3],
        "average_score": round(average_score, 2),
    }


async def critic_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    draft = state.get("draft")
    profile: VoiceProfile = state.get("voice_profile")
    sources: list[Source] = state.get("sources", [])
    iteration = state.get("iteration_count", 0)

    scores = _score_draft(draft, profile, sources)

    # Verdict
    if iteration >= MAX_ITERATIONS:
        verdict = "approve"
    elif scores["average_score"] >= APPROVE_THRESHOLD:
        verdict = "approve"
    else:
        verdict = "rewrite"

    rewrite_instructions = ""
    if verdict == "rewrite":
        tips = []
        if scores["hook_strength"] < 6.5:
            tips.append("Strengthen hook: open with a specific number or surprising fact.")
        if scores["cliche_flags"]:
            tips.append(f"Remove clichés: {', '.join(scores['cliche_flags'])}")
        if scores["insight_density"] < 5.0:
            tips.append("Add more specific claims — each tweet needs one non-obvious idea.")
        rewrite_instructions = " ".join(tips)

    critique = CritiqueResult(
        hook_strength=scores["hook_strength"],
        voice_match=scores["voice_match"],
        insight_density=scores["insight_density"],
        clarity=scores["clarity"],
        hallucination_flags=scores["hallucination_flags"],
        cliche_flags=scores["cliche_flags"],
        average_score=scores["average_score"],
        rewrite_instructions=rewrite_instructions,
        verdict=verdict,
    )

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="critic", model="local heuristics (0 LLM calls)",
        duration_ms=duration_ms,
        detail=(
            f"avg={scores['average_score']:.2f} hook={scores['hook_strength']} "
            f"voice={scores['voice_match']} density={scores['insight_density']} "
            f"clarity={scores['clarity']} verdict={verdict} iteration={iteration}"
        ),
    )
    return {"critique": critique, "status": "critique_done", "trace": [event]}


def route_after_critic(state: AgentState) -> str:
    critique: CritiqueResult = state.get("critique")
    if critique and critique.verdict == "rewrite":
        existing = state.get("writer_instruction", "")
        state["writer_instruction"] = (
            existing + f"\n\nCRITIC FEEDBACK: {critique.rewrite_instructions}"
        ).strip()
        return "rewrite"
    return "advance"