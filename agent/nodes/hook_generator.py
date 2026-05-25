"""
agent/nodes/hook_generator.py — Layer 4: Generate hooks locally from claims.
Zero LLM calls — hooks derived directly from claim data to preserve quota.
"""
from __future__ import annotations

import logging
import time

from agent.schemas import AgentState, Claim, HookVariant, TraceEvent, VoiceProfile
from agent.voice_profile import get_voice_similarity

logger = logging.getLogger(__name__)


def _make_hooks(claims: list[Claim], topic: str, profile: VoiceProfile) -> list[HookVariant]:
    """Generate hooks from claims without an LLM call."""
    hooks: list[HookVariant] = []

    # Find best verified claim with a number (for specific_number hook)
    number_claim = next(
        (c for c in claims if c.verified and any(ch.isdigit() for ch in c.text)), None
    )
    if number_claim:
        hooks.append(HookVariant(
            text=number_claim.text[:280],
            pattern="specific_number",
            score=min(number_claim.surprise_score, 9.5),
            reasoning="Leading with the most surprising verified statistic",
        ))

    # Counterintuitive: highest surprise_score claim
    if claims:
        top = claims[0]
        text = f"Everyone talks about {topic[:40]}. The data says something different: {top.text[:200]}"
        text = text[:280]
        hooks.append(HookVariant(
            text=text,
            pattern="counterintuitive",
            score=min(top.surprise_score, 9.0),
            reasoning="Most surprising claim used as counterintuitive hook",
        ))

    # Bold prediction: contradiction claim if exists
    contradiction = next((c for c in claims if c.contradiction_flag), None)
    if contradiction:
        text = f"By 2027, this won't be debated: {contradiction.text[:220]}"[:280]
        hooks.append(HookVariant(
            text=text,
            pattern="bold_prediction",
            score=7.5,
            reasoning="Contradiction claim reframed as prediction",
        ))

    # Fallback
    if not hooks:
        hooks.append(HookVariant(
            text=f"What nobody tells you about {topic[:60]}:",
            pattern="counterintuitive",
            score=6.0,
            reasoning="Fallback hook",
        ))

    # FAISS voice similarity boost
    if profile:
        for h in hooks:
            sim = get_voice_similarity(h.text, profile)
            if sim > 0.7:
                h.score = min(10.0, h.score + 0.5)

    return sorted(hooks, key=lambda h: h.score, reverse=True)


async def hook_generator_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    profile: VoiceProfile = state.get("voice_profile")
    claims: list[Claim] = state.get("claims", [])
    topic = state["input_context"].user_instruction or state["input_context"].raw_input

    hooks = _make_hooks(claims, topic, profile)

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="hook_generator", model="local (no LLM)",
        duration_ms=duration_ms,
        detail=f"Generated {len(hooks)} hooks (claim-derived, 0 API calls)",
    )
    return {"hook_variants": hooks, "status": "hooks_generated", "trace": [event]}