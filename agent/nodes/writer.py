"""
agent/nodes/writer.py — Multi-model writer with citation discipline.
iteration 1 → Gemini (strong first draft)
iteration ≥2 → Mistral (fresh model avoids self-consistency bias)

Citation contract:
  [S#]       = fact directly reported in a retrieved source
  [hypothesis] = writer's inference / forward-looking prediction — never [S#]
  All hooks must be grounded in a [S#] claim or explicitly tagged [hypothesis].

Non-obviousness contract:
  The writer argues chosen_thesis (from angle_finder) as a through-line.
  Every tweet/paragraph either states the thesis, evidences it, or follows from it.
  Summarising claims is forbidden — winning the argument is the job.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from agent.llm import call_gemini, call_mistral
from agent.schemas import AgentState, Claim, OutputType, TraceEvent, Tweet, VoiceProfile

logger = logging.getLogger(__name__)

BANNED_RE = re.compile(
    r"\b(in conclusion|it's worth noting|as an ai|game.changer|dive into|delve|"
    r"revolutionize|simply put|touch base|move the needle|cutting.edge|seamless|leverage)\b",
    re.IGNORECASE,
)

_FORWARD_LOOKING_RE = re.compile(
    r"\b(expect\b|watch for\b|this will\b|will likely\b|by 20\d\d\b|"
    r"if (this|x|that) happens?\b|poised to\b|set to\b|on track to\b)",
    re.IGNORECASE,
)

# Strip "Write a thread about / Create an essay on / Generate analysis of" prefixes
# so TOPIC in prompts shows the actual subject, not the mode instruction.
_MODE_LABEL_STRIP_RE = re.compile(
    r"^(write|create|generate|make|produce|draft|compose|give me)?\s*"
    r"(an?|the)?\s*"
    r"(x[\s_-]?thread|twitter[\s_-]?thread|quote[\s_-]?rt|quote[\s_-]?retweet|"
    r"thread|essay|blog\s*post|anal[iy]s[ie]s|strategic\s*narrative|"
    r"newsletter|article|post)\s+(about\s+|on\s+|regarding\s+)?",
    re.IGNORECASE,
)

# Detects subordinate-clause (wh-) prefixes that make grammatical nonsense as nouns
# e.g. "why AI will change everything" → can't use in "If why AI... is your moat"
_WH_RE = re.compile(r'^(why|how|what|when|where|whether|which)\b', re.IGNORECASE)

# Minimal stop-set to filter out false proper-noun matches in thesis extraction
_WRITER_NOT_PROPER = {
    "the", "a", "an", "this", "that", "it", "they", "we", "you", "i",
    "if", "when", "while", "as", "but", "and", "or", "so", "for",
    "in", "on", "at", "by", "to", "of", "is", "are", "was", "will",
    "would", "could", "should", "may", "might", "do", "does", "be",
    "have", "has", "had", "its", "my", "your", "our", "their",
    "new", "old", "big", "just", "also", "even", "still", "only",
    "most", "more", "some", "all", "not", "now", "no",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_cta_noun(clean_topic: str, thesis: str) -> str:
    """
    Returns a noun phrase safe for CTAs like "If {noun} is your moat...".
    Detects wh-clause prefixes in clean_topic ("why AI will...", "how models...") and
    falls back to the first proper noun in thesis, then first 6 thesis words.
    """
    if not _WH_RE.match(clean_topic.strip()):
        return clean_topic[:80]
    # wh-clause: extract the first proper noun from thesis as the subject
    for word in thesis.split():
        cleaned = re.sub(r'[^a-zA-Z]', '', word)
        if len(cleaned) >= 2 and cleaned[0].isupper() and cleaned.lower() not in _WRITER_NOT_PROPER:
            return cleaned[:60]
    # last resort: first 6 words of thesis (descriptive but grammatical)
    return ' '.join(thesis.split()[:6])[:80]


def _derive_clean_topic(raw_input: str, thesis: str) -> str:
    """
    Strip mode-label prefix from raw_input to expose the actual subject.
    Falls back to the first 120 chars of thesis if cleaning fails.
    """
    cleaned = _MODE_LABEL_STRIP_RE.sub("", raw_input.strip()).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # If the cleaned result is still instruction-like or too short, use thesis
    if len(cleaned) < 10 or re.match(r"^(write|create|generate|make)\b", cleaned, re.I):
        return thesis[:120] if thesis else raw_input[:120]
    return cleaned[:200]


_CONCRETE_NUMBER_RE = re.compile(r'\b\d[\d,.%$x]*\b')


def _count_tweet_anchors(text: str) -> int:
    """Count concrete anchors (numbers + proper nouns) in a single tweet."""
    numbers = set(_CONCRETE_NUMBER_RE.findall(text))
    proper_nouns: set[str] = set()
    for word in text.split():
        cleaned = re.sub(r'[^a-zA-Z]', '', word)
        if len(cleaned) >= 2 and cleaned[0].isupper() and cleaned.lower() not in _WRITER_NOT_PROPER:
            proper_nouns.add(cleaned.lower())
    return len(numbers) + len(proper_nouns)


def _parse_tweets(raw: str) -> list[Tweet]:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        tweets = []
        for item in data:
            text = item.get("text", "")[:280]
            tweets.append(Tweet(
                position=item.get("position", len(tweets) + 1),
                text=text,
                char_count=len(text),
                claim_ids=item.get("claim_ids", []),
                is_hook=item.get("is_hook", False),
                is_cta=item.get("is_cta", False),
            ))
        return tweets
    except Exception:
        return []


def _dedup_tweets(tweets: list[Tweet]) -> list[Tweet]:
    """Drop tweets with >90% word-level Jaccard overlap with any earlier tweet."""
    seen_words: list[set] = []
    out: list[Tweet] = []
    for t in tweets:
        words = set(t.text.lower().split())
        is_dup = any(
            len(words & s) / max(len(words | s), 1) > 0.90
            for s in seen_words
        )
        if not is_dup:
            seen_words.append(words)
            out.append(t)
        else:
            logger.info("writer: dedup dropped duplicate tweet: %r", t.text[:60])
    return out


def _get_voice_profile(state: AgentState) -> VoiceProfile:
    profile = state.get("voice_profile")
    if profile:
        return profile
    return VoiceProfile(
        avg_sentence_length=14.0,
        vocabulary_richness=0.55,
        preferred_structures=["Opens with a specific data point", "Ends with a provocative question"],
        forbidden_phrases=["in conclusion", "game-changer", "dive into", "seamless", "leverage"],
        style_summary=(
            "Clear, direct writing with strong opinions backed by data. "
            "Short paragraphs, concrete examples. Never passive voice or hedge words."
        ),
        few_shot_examples=[
            "73% of Series A companies never reach Series B. The bottleneck is almost never product. It's distribution, almost every time.",
            "Most founders wait too long to fire. The cost isn't just performance — it's the 3 people who quit working around the problem.",
        ],
        faiss_index_path="memory/faiss/default.index",
    )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

async def writer_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    profile: VoiceProfile = _get_voice_profile(state)
    claims: list[Claim]   = state.get("claims", [])
    thesis                = state.get("chosen_thesis", "")   # set by angle_finder
    output_type: OutputType = state["input_context"].output_type
    original_tweet        = state["input_context"].original_tweet
    critic_feedback       = state.get("writer_instruction", "")
    iteration             = state.get("iteration_count", 0) + 1

    topic = state["input_context"].raw_input or "the topic"
    focus = state["input_context"].user_instruction or ""

    # BUG 1/3 FIX: derive the actual subject, not the mode instruction
    # "Write a thread about why AI will change everything" → "why AI will change everything"
    clean_topic = _derive_clean_topic(topic, thesis)
    logger.info("writer: clean_topic=%r  thesis=%r", clean_topic[:80], thesis[:80])

    # ── Model selection ────────────────────────────────────────────────────────
    if iteration == 1:
        _call    = call_gemini
        provider = "gemini-2.0-flash"
    else:
        _call    = call_mistral
        provider = "mistral-small-latest"

    # ── FAISS-based few-shot loading (voice archive) ───────────────────────────
    user_id   = state.get("user_id", "default")
    faiss_idx = f"memory/faiss/{user_id}.index"
    faiss_txt = f"memory/faiss/{user_id}_texts.json"
    faiss_examples: list[str] = []
    if os.path.exists(faiss_idx) and os.path.exists(faiss_txt):
        try:
            from memory.embedder import search_faiss
            all_texts = json.loads(open(faiss_txt, encoding="utf-8").read())
            _, idxs   = search_faiss(faiss_idx, (thesis or clean_topic)[:200], k=2)
            faiss_examples = [all_texts[i] for i in idxs if 0 <= i < len(all_texts)]
        except Exception as exc:
            logger.debug("writer: FAISS load failed: %s", exc)

    # ── Source legend ─────────────────────────────────────────────────────────
    source_legend = "\n".join(
        f"[S{i+1}] {s.url}  ({s.title[:55]})"
        for i, s in enumerate(state.get("sources", []))
    )

    # ── Voice context ─────────────────────────────────────────────────────────
    voice_ctx = f"\nVOICE STYLE: {profile.style_summary}"
    if profile.forbidden_phrases:
        voice_ctx += f"\nNEVER USE: {', '.join(profile.forbidden_phrases[:8])}"
    examples = faiss_examples or profile.few_shot_examples
    if examples:
        voice_ctx += "\nEXAMPLES (match this voice exactly):\n" + "\n---\n".join(examples[:2])

    # ── Claims with citation tags ─────────────────────────────────────────────
    claims_ctx = "\n".join(
        (
            f"- [{c.source_id}] {c.text}"
            if c.verified and c.source_id
            else f"- [hypothesis] {'[speculative] ' if c.speculative else ''}{c.text}"
        )
        for c in claims
    )

    top_claim     = claims[0].text if claims else ""
    top_claim_sid = (
        f"[{claims[0].source_id}]"
        if claims and claims[0].source_id and claims[0].verified
        else "[hypothesis]"
    )

    # ── Angle context (from angle_finder) ────────────────────────────────────
    second_order      = state.get("second_order", "")
    contrarian_thesis = state.get("contrarian_thesis", "")
    obvious_takes     = state.get("obvious_takes", [])

    angle_ctx = ""
    if obvious_takes:
        angle_ctx += f"\nOBVIOUS TAKES TO AVOID (already saturating the feed): {'; '.join(obvious_takes[:3])}"
    if contrarian_thesis:
        angle_ctx += f"\nCONTRARIAN ANGLE: {contrarian_thesis}"
    if second_order:
        angle_ctx += f"\nSECOND-ORDER INSIGHT: {second_order}"

    # ── Citation rules ────────────────────────────────────────────────────────
    forward_rule = (
        "FORWARD-LOOKING SENTENCE RULE:\n"
        "- Sentences with: Expect..., Watch for..., Will likely..., By 20XX..., "
        "If X happens..., Poised to..., Set to... → always [hypothesis], never [S#].\n"
        "- [S#] = reported past fact. [hypothesis] = your inference about the future."
    )
    citation_rules = (
        "CITATION RULES:\n"
        "- Every factual sentence using a verified claim ends with its [S#] tag.\n"
        "- [hypothesis] claims framed as: 'If true...', 'Reportedly...', 'One reading is...'\n"
        "- Never assert anything not in the claims list.\n"
        + forward_rule
        + (f"\nSOURCES:\n{source_legend}" if source_legend else "")
    )

    # ── Negative examples (domain-agnostic generic AI-Twitter slop) ──────────
    negative_examples = (
        "NEGATIVE EXAMPLES — never write anything this predictable:\n"
        '  "AI is transforming the way we work and live."\n'
        '  "The future of [X] is here, and it\'s powered by [technology]."\n'
        '  "This will revolutionize everything and create unprecedented opportunities."\n'
        "If your sentence could appear verbatim in 100 other posts, delete it and try again."
    )

    # ── BUG 1 FIX: hook grounding — hook must COMMIT to the thesis ────────────
    # Previously: hook anchored on top verified claim (correct for grounding) but
    # didn't require it to express the thesis (so the model wrote freely).
    hook_req = f"""HOOK CONTRACT (NON-NEGOTIABLE):
- Tweet 1 / opening sentence must STATE the thesis: "{thesis[:140]}"
  Do NOT open with a vague setup. Commit to the argument in the first line.
- Ground the hook in a verified claim — cite it with [S#]:
  GOOD: "Zapier's no-code GMV fell 12% QoQ as Claude API calls tripled. [S1] Zapier is losing."
  BAD:  "The traditional software stack is collapsing." (no actor named, no number, no [S#])
- If no verified stat is available, open with a named contradiction labeled [hypothesis].
- Top verified anchor claim: {top_claim[:180] if top_claim else "(none — must use [hypothesis])"} {top_claim_sid}"""

    focus_ctx    = f"\nWRITING FOCUS: {focus}" if focus else ""
    feedback_ctx = f"\nCRITIC FEEDBACK TO FIX:\n{critic_feedback}" if critic_feedback else ""

    # ── Output-type specific generation ───────────────────────────────────────
    concrete_anchor_rule = (
        "CONCRETE ANCHOR RULE (NON-NEGOTIABLE):\n"
        "Every tweet MUST contain at least one named entity OR specific number.\n"
        "Named entities = company (Zapier, OpenAI), product (Claude API, GPT-4o), "
        "person (Sam Altman, Satya Nadella), or metric ($2B, 40%, Q3 2025).\n"
        '"AI is transforming work" names nothing — DELETE it. '
        '"Zapier lost 12% GMV to Claude API integrations" names everything — KEEP it.'
    )

    if output_type == OutputType.quote_rt:
        system = f"""MANDATORY THESIS — argue exactly this in your response:
"{thesis}"
{voice_ctx}
STRICT RULES:
- Produce EXACTLY 1-3 tweets. No more.
- Respond to the original tweet by ADVANCING the argument above, not summarising.
- Each tweet ≤ 280 chars, one non-obvious idea per tweet.
- Never use: in conclusion, game-changer, seamless, leverage, dive into.
- Return ONLY a JSON array, nothing else.
{concrete_anchor_rule}
{negative_examples}
{citation_rules}"""

        if original_tweet:
            prompt = f"""Write a quote-retweet (1-3 tweets max). ARGUE the thesis — don't summarise.

ORIGINAL TWEET:
{original_tweet[:500]}

SUBJECT: {clean_topic}{focus_ctx}{angle_ctx}
{hook_req}
THESIS TO ARGUE: {thesis}
USE THESE AS EVIDENCE (not as a list to cover):
{claims_ctx}{feedback_ctx}

JSON array (1-3 items only):
[{{"position": 1, "text": "...", "char_count": N, "claim_ids": [], "is_hook": true, "is_cta": false}}]"""
        else:
            prompt = f"""SUBJECT: {clean_topic}{focus_ctx}{angle_ctx}

{hook_req}
THESIS TO ARGUE: {thesis}
USE THESE AS EVIDENCE:
{claims_ctx}{feedback_ctx}

JSON array (1-3 items only):
[{{"position": 1, "text": "...", "char_count": N, "claim_ids": [], "is_hook": true, "is_cta": false}}]"""

        raw   = await _call(prompt, system, max_tokens=600)
        draft = _parse_tweets(raw)
        draft = _dedup_tweets(draft)   # BUG 2 FIX

        if not draft:
            hook_text = f"{thesis[:260]} {top_claim_sid}"[:280]
            draft = [Tweet(position=1, text=hook_text, char_count=len(hook_text), is_hook=True)]

        # Hard-trim to 3 — keep hook + most-cited content tweets
        if len(draft) > 3:
            hook_tweets    = [t for t in draft if t.is_hook or t.position == 1]
            content_tweets = sorted(
                [t for t in draft if not t.is_cta and t not in hook_tweets],
                key=lambda t: len(t.claim_ids), reverse=True,
            )
            draft = (hook_tweets + content_tweets)[:3]
            for i, t in enumerate(sorted(draft, key=lambda x: x.position), 1):
                t.position = i
            logger.info("quote_rt: trimmed to 3 tweets")

        for t in draft:
            if len(t.text) > 280:
                t.text       = t.text[:277] + "..."
                t.char_count = 280

    elif output_type == OutputType.x_thread:
        # BUG 1 FIX: thesis appears at the top of system prompt, not buried in user prompt
        system = f"""MANDATORY THESIS — every tweet must either state, support, or follow from:
"{thesis}"

Writing freely about "{clean_topic}" is NOT acceptable.
Each tweet is one step in the argument for the thesis above.

You are a ghostwriter writing a 6-8 tweet thread.{voice_ctx}
RULES:
- Every tweet ≤ 280 chars.
- Structure: hook (commits to thesis) → evidence tweets → implication tweet → CTA.
- Exactly 6-8 tweets. Never fewer.
- Never use: in conclusion, game-changer, seamless, leverage, dive into.
- Return ONLY a JSON array, nothing else.
{concrete_anchor_rule}
{negative_examples}
{citation_rules}"""

        # BUG 3 FIX: SUBJECT uses clean_topic (the actual subject), not raw_input
        prompt = f"""SUBJECT: {clean_topic}{focus_ctx}{angle_ctx}

{hook_req}
THESIS TO ARGUE: {thesis}
USE THESE AS EVIDENCE (each tweet advances the argument, doesn't just list facts):
{claims_ctx}{feedback_ctx}

JSON array (exactly 6-8 tweets):
[{{"position": 1, "text": "...", "char_count": N, "claim_ids": [], "is_hook": true, "is_cta": false}}]"""

        raw   = await _call(prompt, system, max_tokens=1500)
        draft = _parse_tweets(raw)
        draft = _dedup_tweets(draft)   # BUG 2 FIX

        # BUG 6 FIX: enforce 6-tweet minimum — pad with claim-based evidence tweets
        if len(draft) < 6:
            next_pos = (max(t.position for t in draft) + 1) if draft else 2
            for c in claims:
                if len(draft) >= 6:
                    break
                sid = f"[{c.source_id}]" if c.source_id and c.verified else "[hypothesis]"
                body = f"{c.text[:260]} {sid}"[:280]
                draft.append(Tweet(
                    position=next_pos, text=body, char_count=len(body), claim_ids=[c.id],
                ))
                next_pos += 1
            # Add implication tweet if still short
            if len(draft) < 6 and thesis:
                impl = f"Bottom line: {thesis[:250]}"[:280]
                draft.append(Tweet(position=next_pos, text=impl, char_count=len(impl)))
                next_pos += 1
            # BUG 3 FIX: CTA uses _derive_cta_noun to avoid wh-clause ("why AI...")
            if not any(t.is_cta for t in draft) and len(draft) < 8:
                cta_noun = _derive_cta_noun(clean_topic, thesis)
                cta = f"If {cta_noun} is your moat, the clock is running."[:280]
                draft.append(Tweet(position=next_pos, text=cta, char_count=len(cta), is_cta=True))
            logger.info("x_thread: padded to %d tweets after dedup", len(draft))

        # Post-process: fix any CTA that still echoes raw_input or contains a wh-clause
        for t in draft:
            if t.is_cta and (
                topic.lower()[:20] in t.text.lower()
                or _WH_RE.search(t.text)
            ):
                cta_noun     = _derive_cta_noun(clean_topic, thesis)
                t.text       = f"If {cta_noun} is your moat, the clock is running."[:280]
                t.char_count = len(t.text)
                logger.info("writer: fixed bad CTA → %r", t.text[:60])

        # ── Per-tweet concreteness scrub ──────────────────────────────────────
        # Every non-hook, non-CTA tweet must name at least one entity or number.
        # Abstract tweets are replaced with the next concrete claim from the
        # research list (verified first, then hypothesis).
        concrete_claims = sorted(
            [c for c in claims if _count_tweet_anchors(c.text) > 0],
            key=lambda c: (c.verified, c.surprise_score), reverse=True,
        )
        claim_cursor = 0
        for t in draft:
            if t.is_hook or t.is_cta:
                continue
            if _count_tweet_anchors(t.text) == 0 and claim_cursor < len(concrete_claims):
                c   = concrete_claims[claim_cursor]; claim_cursor += 1
                sid = f"[{c.source_id}]" if c.source_id and c.verified else "[hypothesis]"
                new_text     = f"{c.text[:260]} {sid}"[:280]
                logger.info(
                    "writer: abstract tweet %d replaced → %r",
                    t.position, new_text[:60],
                )
                t.text       = new_text
                t.char_count = len(new_text)
                t.claim_ids  = [c.id]

        # ── Hook promotion: most concrete tweet → position 1 ─────────────────
        # After scrub + padding, the sharpest/most-specific claim may not be
        # first. Sort by position, find the highest-anchor non-CTA tweet, and
        # swap it to the front so the boldest claim leads the thread.
        draft.sort(key=lambda t: t.position)
        if draft:
            best_i, best_score = max(
                ((i, _count_tweet_anchors(t.text)) for i, t in enumerate(draft) if not t.is_cta),
                key=lambda x: x[1],
                default=(0, 0),
            )
            hook_score = _count_tweet_anchors(draft[0].text)
            if best_i != 0 and best_score > hook_score:
                logger.info(
                    "writer: promoting tweet %d (anchors=%d) %r → hook; was %r (anchors=%d)",
                    draft[best_i].position, best_score, draft[best_i].text[:50],
                    draft[0].text[:50], hook_score,
                )
                draft.insert(0, draft.pop(best_i))
            # Reapply hook flag and sequential positions
            for t in draft:
                t.is_hook = False
            draft[0].is_hook = True
            for i, t in enumerate(draft, 1):
                t.position = i

        for t in draft:
            if len(t.text) > 280:
                t.text       = t.text[:277] + "..."
                t.char_count = 280

    else:
        # Essay / analysis / strategic_narrative
        # BUG 1 FIX: thesis is the first thing in system prompt
        system = f"""MANDATORY THESIS — you are writing to prove exactly this claim:
"{thesis}"
Every paragraph must either state, evidence, or logically follow from this thesis.
Do NOT "cover {clean_topic}" — WIN THE ARGUMENT for the thesis above.
{voice_ctx}
RULES:
1. First sentence: hook that COMMITS to the thesis, grounded in a [S#] claim.
   NEVER start with: "The field of...", "In today's...", "The role of..."
2. One idea per paragraph. If a paragraph doesn't advance the thesis, cut it.
3. Only cite URLs from the SOURCE LIST below — never invent URLs.
4. End with the sharpest implication of the thesis — no soft conclusions.
5. NEVER use: in conclusion, game-changer, seamless, leverage, dive into.
6. {citation_rules}
{concrete_anchor_rule}
{negative_examples}
SOURCES:
{source_legend if source_legend else "(no sources — use only claims provided)"}"""

        prompt = f"""SUBJECT: {clean_topic}{focus_ctx}{angle_ctx}

{hook_req}
THESIS TO ARGUE: {thesis}
USE THESE AS EVIDENCE — argue the thesis, don't narrate the claims:
{claims_ctx}{feedback_ctx}

Write a {output_type.value.replace('_', ' ')} in markdown. Start immediately with the hook — no title, no preamble."""

        raw   = await _call(prompt, system, max_tokens=2000)
        draft = raw if raw else f"# {thesis}\n\n" + "\n\n".join(c.text for c in claims)

        if isinstance(draft, str):
            draft = re.sub(
                r'^#{1,3} .*(conclusion|not allowed|rephrase|do not use|removed|final statement).*$',
                '', draft, flags=re.IGNORECASE | re.MULTILINE,
            ).strip()
            draft = re.sub(r'\n{3,}', '\n\n', draft).strip()

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="writer", model=provider,
        duration_ms=duration_ms,
        detail=f"iteration={iteration}, output_type={output_type.value}, thesis={thesis[:60]}",
    )
    return {"draft": draft, "iteration_count": iteration, "status": "writing_done", "trace": [event]}
