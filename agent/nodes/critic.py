"""
agent/nodes/critic.py — LLM critic + originality check + FAISS voice_match.

Scoring axes:
  hook_strength, insight_density, clarity — always scored by LLM
  voice_match   — cosine similarity to user's FAISS archive (null if no archive)
  originality   — web-search count of how many results say the same thing (null if Tavily unavailable)

Verdict logic:
  originality < 6  → REWRITE regardless of other scores (polished cliché must fail)
  average < 6.5    → REWRITE
  iteration ≥ MAX  → approve (fail-safe)
  voice_match      → excluded from average when null (don't punish for no archive)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from agent.llm import call_llama_stack
from agent.schemas import AgentState, CritiqueResult, TraceEvent
from agent.tools import tavily_search

logger = logging.getLogger(__name__)

MAX_ITERATIONS       = 2
APPROVE_THRESHOLD    = 6.5
ORIGINALITY_VETO     = 6.0   # below this → mandatory rewrite
CONCRETENESS_VETO    = 5.0   # below this → mandatory rewrite

BANNED_RE = re.compile(
    r"\b(in conclusion|it's worth noting|as an ai|game.changer|dive into|delve|"
    r"revolutionize|simply put|touch base|move the needle|cutting.edge|seamless|leverage)\b",
    re.IGNORECASE,
)

# Stop-words excluded from word-overlap comparison
_STOP = {"the", "a", "an", "is", "are", "of", "in", "to", "and", "or", "but",
         "for", "with", "this", "that", "was", "has", "be", "have", "at", "by"}

_NUMBER_RE = re.compile(r'\b\d[\d,.%$x]*\b')
_NOT_PROPER = {
    "the", "a", "an", "this", "that", "it", "they", "we", "you", "i",
    "if", "when", "as", "but", "and", "or", "so", "for", "in", "on",
    "at", "by", "to", "of", "is", "are", "was", "will", "would", "could",
    "should", "may", "might", "do", "does", "be", "have", "has", "had",
}


def _count_concrete_anchors(text: str) -> int:
    """Count concrete signals: distinct numbers + Title-Case proper nouns."""
    numbers = set(_NUMBER_RE.findall(text))
    proper = {
        re.sub(r'[^a-zA-Z]', '', w).lower()
        for w in text.split()
        if len(re.sub(r'[^a-zA-Z]', '', w)) >= 2
        and re.sub(r'[^a-zA-Z]', '', w)[0].isupper()
        and re.sub(r'[^a-zA-Z]', '', w).lower() not in _NOT_PROPER
    }
    return len(numbers) + len(proper)


def _score_concreteness(draft) -> tuple[Optional[float], float]:
    """
    Syntactic concreteness: avg count of numbers + Title-Case proper nouns per tweet.
    Returns (score 1-10, avg_anchors). None for essay drafts (str).
    """
    if not isinstance(draft, list) or not draft:
        return None, 0.0
    counts = [_count_concrete_anchors(t.text) for t in draft]
    avg = sum(counts) / len(counts)
    score = 9.0 if avg >= 2 else (7.0 if avg >= 1 else (4.5 if avg >= 0.5 else 2.0))
    logger.info("critic: concreteness avg_anchors=%.2f score=%.1f", avg, score)
    return round(score, 1), round(avg, 2)


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


def _draft_text(draft) -> str:
    if isinstance(draft, list):
        return " ".join(t.text for t in draft)
    return draft or ""


def _extract_hook(draft) -> str:
    """First tweet or first sentence — used as originality search probe."""
    if isinstance(draft, list) and draft:
        hook_tweet = next((t for t in draft if getattr(t, "is_hook", False)), draft[0])
        return hook_tweet.text[:160]
    if isinstance(draft, str) and draft:
        m = re.search(r'^[^\n.!?]{20,}[.!?]', draft.strip())
        return m.group(0)[:160] if m else draft[:160]
    return ""


def _word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split()) - _STOP
    wb = set(b.lower().split()) - _STOP
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _count_similar_semantic(hook: str, results: list[dict]) -> int:
    """
    Primary: MiniLM cosine similarity between hook and each result snippet.
    Threshold 0.70 catches same-idea with different vocabulary.
    Fallback: word-overlap at 0.25 if embedder unavailable.
    Zero extra API cost — MiniLM is local.
    """
    snippets = [r.get("title", "") + " " + r.get("content", "")[:300] for r in results]
    if not snippets:
        return 0
    try:
        from memory.embedder import embed_texts
        import numpy as np
        vecs     = embed_texts([hook] + snippets)   # unit-normalized by MiniLM
        hook_vec = vecs[0]
        snip_vecs = np.array(vecs[1:])
        sims     = snip_vecs @ hook_vec             # cosine (dot of unit vecs)
        count    = int((sims >= 0.70).sum())
        logger.debug("critic: semantic originality — %d/%d snippets ≥ 0.70 cosine", count, len(snippets))
        return count
    except Exception as exc:
        logger.debug("critic: semantic embedder unavailable (%s) — fallback to word-overlap 0.25", exc)
        return sum(
            1 for r in results
            if _word_overlap(hook, r.get("title", "") + " " + r.get("content", "")[:400]) >= 0.25
        )


async def _score_originality(
    hook: str,
    obvious_takes: list[str] | None = None,
) -> tuple[Optional[float], list[dict]]:
    """
    Search for the hook phrase. Count results that express the same idea
    (semantic cosine ≥ 0.70, fallback word-overlap ≥ 0.25).
    Also cross-checks hook against angle_finder's obvious_takes (cosine ≥ 0.65) —
    the most reliable cliché signal because we explicitly named the consensus.
    Returns (score 1-10, raw search results for feedback).
    Returns (None, []) if Tavily unavailable.
    """
    if not os.getenv("TAVILY_API_KEY", ""):
        return None, []
    if not hook.strip():
        return None, []

    # Log the exact probe text so we can verify it's searching the right string
    logger.info("critic: originality probe text = %r", hook[:160])

    try:
        results = await tavily_search(hook[:150], max_results=6)
        similar = _count_similar_semantic(hook, results)

        # Cross-check against angle_finder's known-obvious takes.
        # If the hook is semantically close to what angle_finder flagged as cliché,
        # that's a direct originality signal independent of what Tavily found.
        ob_matches = 0
        if obvious_takes:
            try:
                from memory.embedder import embed_texts
                import numpy as np
                vecs     = embed_texts([hook] + obvious_takes)
                hook_vec = vecs[0]
                ob_vecs  = np.array(vecs[1:])
                ob_sims  = ob_vecs @ hook_vec
                ob_matches = int((ob_sims >= 0.65).sum())
                if ob_matches:
                    logger.info(
                        "critic: hook matches %d obvious_take(s) at cosine ≥ 0.65 "
                        "(top sim=%.3f) — boosting similar count",
                        ob_matches, float(ob_sims.max()),
                    )
            except Exception as exc:
                logger.debug("critic: obvious_takes cosine check failed: %s", exc)

        similar = min(similar + ob_matches, 7)  # cap: avoid over-penalizing

        if similar >= 4:
            score = max(1.0, 4.0 - (similar - 4) * 0.5)
        elif similar >= 2:
            score = 5.0
        elif similar == 1:
            score = 7.0
        else:
            score = 8.5

        logger.info(
            "critic: originality search '%s...' → web_similar=%d ob_matches=%d total=%d → score=%.1f",
            hook[:50], similar - ob_matches, ob_matches, similar, score,
        )
        return round(score, 1), results
    except Exception as exc:
        logger.warning("critic: originality search failed: %s", exc)
        return None, []


async def _score_voice_match(full_text: str, user_id: str) -> Optional[float]:
    """
    Embed draft, compute max cosine to user's FAISS archive.
    Returns 1-10 or None if no archive exists.
    Purely local — zero API cost.
    """
    faiss_idx_path = f"memory/faiss/{user_id}.index"
    faiss_txt_path = f"memory/faiss/{user_id}_texts.json"
    if not (os.path.exists(faiss_idx_path) and os.path.exists(faiss_txt_path)):
        return None
    try:
        from memory.embedder import embed_texts
        import numpy as np

        all_texts = json.loads(open(faiss_txt_path, encoding="utf-8").read())
        if not all_texts:
            return None
        sample = all_texts[:30]   # keep embedding cost low
        vecs   = embed_texts([full_text[:600]] + sample)
        draft_vec    = vecs[0]
        archive_vecs = np.array(vecs[1:])
        # MiniLM vectors are unit-normalized → dot product = cosine similarity
        sims = archive_vecs @ draft_vec
        max_sim = float(sims.max())
        score   = round(max(1.0, min(10.0, max_sim * 10)), 1)
        logger.info("critic: FAISS voice_match for user %s → cosine %.3f → score %.1f",
                    user_id, max_sim, score)
        return score
    except Exception as exc:
        logger.debug("critic: FAISS voice_match failed: %s", exc)
        return None


async def critic_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    draft     = state.get("draft")
    iteration = state.get("iteration_count", 0)
    topic     = state["input_context"].raw_input or "the topic"
    user_id   = state.get("user_id", "default")

    full_text   = _draft_text(draft)
    hook_phrase = _extract_hook(draft)

    # ── Local cliché detection (fast, zero cost) ───────────────────────────────
    cliche_flags = list(set(m.lower() for m in BANNED_RE.findall(full_text)))

    # ── LLM scoring — 3 style axes: hook / density / clarity ──────────────────
    system = "You are a strict content editor. Score objectively. Return ONLY valid JSON."
    prompt = f"""Score this content written about: "{topic}"

CONTENT:
{full_text[:1200]}

Score each dimension 1-10. Be strict — 7+ means genuinely good.

Return JSON:
{{
  "hook_strength": <1-10: does first line grab with a specific fact/number/bold claim?>,
  "insight_density": <1-10: does every sentence carry a non-obvious idea?>,
  "clarity": <1-10: are sentences short and precise?>,
  "feedback": "<one specific actionable sentence — be concrete, name the weak sentence>"
}}"""

    logger.debug("critic: prompt (first 300) = %r", prompt[:300])
    raw  = await call_llama_stack(prompt, system, max_tokens=280)
    logger.debug("critic: raw response (first 300) = %r", raw[:300])
    data = _parse(raw)
    if not data:
        logger.warning("critic: JSON parse failed — full raw response: %s", raw)

    hook_strength   = float(data.get("hook_strength", 5.5))
    insight_density = float(data.get("insight_density", 5.0))
    clarity         = float(data.get("clarity", 7.0))
    llm_feedback    = data.get("feedback", "")

    if cliche_flags:
        hook_strength   = max(1.0, hook_strength - 0.5)
        insight_density = max(1.0, insight_density - len(cliche_flags) * 0.3)

    # ── Concreteness (local, zero cost — proper nouns + numbers per tweet) ───────
    concreteness_score, avg_anchors = _score_concreteness(draft)

    # ── Originality check (Tavily search) and FAISS voice_match ──────────────
    obvious_takes = state.get("obvious_takes", [])
    originality_score, search_results = await _score_originality(hook_phrase, obvious_takes)
    voice_match_score                  = await _score_voice_match(full_text, user_id)

    # ── Average — excludes null axes ──────────────────────────────────────────
    # Threshold invariant: APPROVE_THRESHOLD=6.5 is calibrated per-axis (each
    # axis is independently 1-10), so averaging over N axes means the same bar.
    scored_axes = [hook_strength, insight_density, clarity]
    if voice_match_score  is not None: scored_axes.append(voice_match_score)
    if originality_score  is not None: scored_axes.append(originality_score)
    if concreteness_score is not None: scored_axes.append(concreteness_score)
    average_score = sum(scored_axes) / len(scored_axes)

    # ── Verdict — hard vetos first, then threshold ────────────────────────────
    if concreteness_score is not None and concreteness_score < CONCRETENESS_VETO:
        verdict = "rewrite"
        veto_trigger = "concreteness_veto"
    elif originality_score is not None and originality_score < ORIGINALITY_VETO:
        verdict = "rewrite"
        veto_trigger = "originality_veto"
    elif iteration >= MAX_ITERATIONS:
        verdict = "approve"
        veto_trigger = "max_iterations"
    elif average_score >= APPROVE_THRESHOLD:
        verdict = "approve"
        veto_trigger = "passed"
    else:
        verdict = "rewrite"
        veto_trigger = "low_average"

    # ── Build rewrite instructions ────────────────────────────────────────────
    rewrite_instructions = ""
    if verdict == "rewrite":
        tips = []

        # Concreteness veto — name the offending abstract tweets
        if concreteness_score is not None and concreteness_score < CONCRETENESS_VETO:
            if isinstance(draft, list):
                abstract_tweets = [
                    f"Tweet {t.position}: {t.text[:60]!r}"
                    for t in draft
                    if _count_concrete_anchors(t.text) == 0
                ]
                tips.append(
                    f"CONCRETENESS FAIL (avg_anchors={avg_anchors:.2f}, score={concreteness_score:.0f}/10): "
                    f"Every tweet must name a company, product, person, or number. "
                    f"Vapor tweets to rewrite: {'; '.join(abstract_tweets[:3])}. "
                    "Replace each with a sentence that names a specific actor."
                )

        # Originality veto — cite competing phrasings
        if originality_score is not None and originality_score < ORIGINALITY_VETO:
            competing = [r.get("title", "") for r in search_results[:3] if r.get("title")]
            tips.append(
                f"ORIGINALITY FAIL (score={originality_score:.0f}/10): "
                f"Hook idea is not novel. Competing phrasings: "
                f"{'; '.join(competing) if competing else 'similar results found'}. "
                "Rewrite from the contrarian_thesis angle — name the specific company/actor being disrupted."
            )

        if hook_strength < 6.5:
            tips.append("Rewrite the hook: open with a specific named number or surprising named fact.")
        if insight_density < 5.5:
            tips.append("Every tweet must carry ONE non-obvious idea. Remove filler entirely.")
        if clarity < 6.0:
            tips.append("Shorter sentences. One claim per sentence. Cut hedge words.")
        if cliche_flags:
            tips.append(f"Remove these clichés: {', '.join(cliche_flags)}.")
        if llm_feedback:
            tips.append(llm_feedback)

        rewrite_instructions = " ".join(tips)

    # ── Hallucination check — tweets with no claim citations ──────────────────
    hallucination_flags = []
    if isinstance(draft, list):
        for t in draft:
            if not t.is_hook and not t.is_cta and not t.claim_ids:
                hallucination_flags.append(f"Tweet {t.position}: no source citation")

    critique = CritiqueResult(
        hook_strength       = round(hook_strength, 1),
        voice_match         = voice_match_score,
        originality         = originality_score,
        concreteness        = concreteness_score,
        insight_density     = round(insight_density, 1),
        clarity             = round(clarity, 1),
        hallucination_flags = hallucination_flags[:3],
        cliche_flags        = cliche_flags[:3],
        average_score       = round(average_score, 2),
        rewrite_instructions = rewrite_instructions,
        verdict             = verdict,
    )

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node  = "critic",
        model = "llama-3.3-70b (llama-stack) + tavily originality + local concreteness",
        duration_ms = duration_ms,
        detail = (
            f"avg={average_score:.2f} hook={hook_strength} "
            f"density={insight_density} clarity={clarity} "
            f"originality={originality_score} concreteness={concreteness_score} "
            f"avg_anchors={avg_anchors:.2f} voice_match={voice_match_score} "
            f"verdict={verdict} trigger={veto_trigger} iter={iteration}"
        ),
    )
    updates: dict = {"critique": critique, "status": "critique_done", "trace": [event]}
    if verdict == "rewrite":
        updates["writer_instruction"] = rewrite_instructions
    return updates


def route_after_critic(state: AgentState) -> str:
    critique: CritiqueResult = state.get("critique")
    if critique and critique.verdict == "rewrite":
        return "rewrite"
    return "advance"
