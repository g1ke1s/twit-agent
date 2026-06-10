"""
agent/nodes/angle_finder.py — Adversarial angle selection.

Produces a non-obvious, CONCRETE thesis — one that names a specific entity
(company, product, person, or number) and is falsifiable like a bet.
"Zapier is dead" is bettable. "The paradigm will shift" is not.

Two validation gates:
  1. Overlap check  — chosen_angle must not echo obvious_takes
  2. Concreteness   — chosen_angle must contain at least one proper noun or number
Fallback chain: chosen_angle → contrarian_thesis → second_order → analyst_thesis
"""
from __future__ import annotations

import json
import logging
import re
import time

from agent.llm import call_llama_stack
from agent.schemas import AgentState, Claim, TraceEvent

logger = logging.getLogger(__name__)

# ── Concreteness detection ─────────────────────────────────────────────────────
_NUMBER_RE = re.compile(r'\b\d[\d,.%$x]*\b')

# Words that are capitalized as function words — NOT evidence of a proper noun
_NOT_PROPER = {
    "the", "a", "an", "this", "that", "these", "those", "it", "they", "we",
    "you", "i", "if", "when", "while", "as", "but", "and", "or", "so", "yet",
    "for", "nor", "in", "on", "at", "by", "to", "of", "is", "are", "was",
    "will", "would", "could", "should", "may", "might", "do", "does", "be",
    "have", "has", "had", "get", "its", "my", "your", "our", "their",
    "new", "old", "big", "just", "also", "even", "still", "only", "most",
    "more", "some", "all", "not", "now", "no",
}


def _has_concrete_anchor(text: str) -> bool:
    """
    Returns True if text contains at least one concrete entity:
      - any digit sequence (number, stat, year, %)
      - any Title/ALL-CAPS word that is not a common function word
    Rejects pure-abstraction sentences like "The paradigm will shift."
    """
    if _NUMBER_RE.search(text):
        return True
    for word in text.split():
        cleaned = re.sub(r'[^a-zA-Z]', '', word)
        if len(cleaned) >= 2 and cleaned[0].isupper() and cleaned.lower() not in _NOT_PROPER:
            return True
    return False


# ── Overlap detection ──────────────────────────────────────────────────────────

def _overlaps_obvious(chosen: str, obvious: list[str], threshold: float = 0.45) -> bool:
    """True if chosen shares ≥ threshold fraction of content words with any obvious take."""
    chosen_words = set(chosen.lower().split()) - _NOT_PROPER
    for ob in obvious:
        ob_words = set(ob.lower().split()) - _NOT_PROPER
        if not ob_words:
            continue
        if len(chosen_words & ob_words) / len(ob_words) >= threshold:
            return True
    return False


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        return {}


async def angle_finder_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    claims: list[Claim] = state.get("claims", [])
    analyst_thesis = state.get("chosen_thesis", "")
    topic = state["input_context"].raw_input or "the topic"
    focus = state["input_context"].user_instruction or ""

    claims_text = "\n".join(
        f"- {'[VERIFIED]' if c.verified else '[hypothesis]'} {c.text}"
        for c in claims[:8]
    )

    system = (
        "You are an adversarial editorial strategist. Your job is to find the non-obvious, "
        "SPECIFIC angle that names real entities. Vague thesis = useless thesis. "
        "Return ONLY valid JSON — no preamble, no markdown wrapper."
    )
    prompt = f"""Topic: "{topic[:300]}"
{f'Writing focus: "{focus}"' if focus else ""}
Analyst's draft thesis: "{analyst_thesis}"

VERIFIED CLAIMS (evidence available):
{claims_text if claims_text else "(no claims — reason from topic alone)"}

CONCRETENESS RULE — applies to every field below:
A good thesis is something you could bet money on and be proven right or wrong.
"Zapier will lose 30% of its SMB customers to LLM wrappers by 2026" is bettable.
"The paradigm will shift" is not — it names nothing, predicts nothing, loses nothing.
Name names. Every output below MUST include at least one: company, product, person, or number.

INSTRUCTIONS:
STEP 1 — List the 3 takes that EVERYONE already has on this topic (name the specific clichés).
STEP 2 — Write a contrarian thesis that contradicts one of those takes. Name a specific company, product, or mechanism. Support it with the claims above.
STEP 3 — Identify the second-order consequence. What happens next, downstream? Name the actor that gets hurt or benefits.
STEP 4 — chosen_angle: the sharpest, most falsifiable sentence. MUST contain at least one named entity or number. MUST NOT restate any obvious_take.

Return JSON (no other text):
{{
  "obvious_takes": ["<specific cliché 1>", "<specific cliché 2>", "<specific cliché 3>"],
  "contrarian_thesis": "<defensible, named claim contradicting consensus>",
  "second_order": "<named downstream consequence — who loses, who wins, by how much>",
  "chosen_angle": "<single sentence — names entities, is falsifiable, does not restate obvious_takes>"
}}"""

    logger.debug("angle_finder: prompt (first 300) = %r", prompt[:300])
    raw  = await call_llama_stack(prompt, system, max_tokens=520)
    logger.debug("angle_finder: raw response (first 300) = %r", raw[:300])
    data = _parse(raw)
    if not data:
        logger.warning("angle_finder: JSON parse failed — full raw response: %s", raw)

    obvious_takes     = [str(t) for t in data.get("obvious_takes", []) if t][:3]
    contrarian_thesis = str(data.get("contrarian_thesis", "")).strip()
    second_order      = str(data.get("second_order", "")).strip()
    chosen_angle      = str(data.get("chosen_angle", "")).strip()

    # ── Gate 1: overlap check ─────────────────────────────────────────────────
    # Fallback chain: chosen_angle → contrarian_thesis → second_order
    if chosen_angle and obvious_takes and _overlaps_obvious(chosen_angle, obvious_takes):
        logger.warning("angle_finder: chosen_angle overlaps obvious_takes — trying contrarian_thesis")
        if contrarian_thesis and not _overlaps_obvious(contrarian_thesis, obvious_takes):
            chosen_angle = contrarian_thesis
        elif second_order:
            logger.warning("angle_finder: contrarian_thesis also overlaps — falling back to second_order")
            chosen_angle = second_order
        else:
            chosen_angle = contrarian_thesis
            logger.warning("angle_finder: all candidates overlap — using contrarian_thesis as least-bad")

    # ── Gate 2: concreteness check ────────────────────────────────────────────
    # Rejects pure-abstraction angles ("the paradigm will shift", "interface improves").
    # Tries each candidate in order until one passes, then logs the outcome.
    if chosen_angle and not _has_concrete_anchor(chosen_angle):
        logger.warning(
            "angle_finder: chosen_angle has no concrete anchor: %r — searching candidates",
            chosen_angle[:80],
        )
        replaced = False
        for label, candidate in [("contrarian_thesis", contrarian_thesis), ("second_order", second_order)]:
            if candidate and _has_concrete_anchor(candidate):
                chosen_angle = candidate
                logger.info("angle_finder: replaced with concrete %s: %r", label, chosen_angle[:80])
                replaced = True
                break
        if not replaced:
            logger.warning(
                "angle_finder: no candidate has a concrete anchor — "
                "proceeding with abstract angle (will likely fail critic concreteness gate)"
            )

    # ── Empty-string guard ────────────────────────────────────────────────────
    if not chosen_angle:
        chosen_angle = analyst_thesis
        logger.warning("angle_finder: empty angle from LLM — falling back to analyst thesis")

    concrete = _has_concrete_anchor(chosen_angle)
    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="angle_finder",
        model="llama-3.3-70b (llama-stack)",
        duration_ms=duration_ms,
        detail=f"angle={chosen_angle[:80]} concrete={concrete}",
    )
    return {
        "chosen_thesis":     chosen_angle,
        "obvious_takes":     obvious_takes,
        "contrarian_thesis": contrarian_thesis,
        "second_order":      second_order,
        "status":            "angle_found",
        "trace":             [event],
    }
