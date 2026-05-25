"""
agent/nodes/writer.py — Layer 5: Single Groq call for writing.
Consolidated to 1 LLM call to preserve free tier quota.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx

from agent.schemas import (
    AgentState, Claim, HookVariant, OutputType, TraceEvent, Tweet, VoiceProfile,
)

logger = logging.getLogger(__name__)

BANNED_RE = re.compile(
    r"\b(in conclusion|it's worth noting|as an ai|game.changer|dive into|delve|"
    r"revolutionize|simply put|touch base|move the needle|cutting.edge|seamless|leverage)\b",
    re.IGNORECASE,
)


async def _groq_call(prompt: str, system: str, max_tokens: int = 2000) -> str:
    keys = [k for k in [
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GROQ_API_KEY_2", ""),
    ] if k]
    if not keys:
        return ""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    for api_key in keys:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=90) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": max_tokens},
                    )
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("retry-after", 30))
                        if retry_after > 300:
                            logger.warning("Groq writer key exhausted — trying next key")
                            break
                        retry_after = min(retry_after, 60)
                        logger.warning("Groq writer 429 — waiting %ds (attempt %d/2)", retry_after, attempt + 1)
                        import asyncio; await asyncio.sleep(retry_after + 1)
                        continue
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.error("Groq writer failed (attempt %d): %s", attempt + 1, exc)
                import asyncio; await asyncio.sleep(3)
    return ""


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
            text = item.get("text", "")
            tweets.append(Tweet(
                position=item.get("position", len(tweets) + 1),
                text=text[:280], char_count=min(len(text), 280),
                claim_ids=item.get("claim_ids", []),
                is_hook=item.get("is_hook", False),
                is_cta=item.get("is_cta", False),
            ))
        return tweets
    except Exception:
        return []


async def writer_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    profile: VoiceProfile = state.get("voice_profile")
    claims: list[Claim] = state.get("claims", [])
    hooks: list[HookVariant] = state.get("hook_variants", [])
    thesis = state.get("chosen_thesis", "")
    output_type: OutputType = state["input_context"].output_type
    original_tweet = state["input_context"].original_tweet
    extra_instruction = state.get("writer_instruction", "")
    iteration = state.get("iteration_count", 0) + 1
    state_topic = (
        state["input_context"].user_instruction
        or state["input_context"].raw_input
        or "the topic"
    )

    # Build voice context
    voice_ctx = ""
    if profile:
        voice_ctx = f"\nVOICE: {profile.style_summary}"
        if profile.forbidden_phrases:
            voice_ctx += f"\nNEVER USE: {', '.join(profile.forbidden_phrases[:8])}"
        if profile.few_shot_examples:
            voice_ctx += f"\nEXAMPLES OF THIS PERSON'S WRITING:\n" + "\n---\n".join(profile.few_shot_examples[:3])

    # Build claims context
    claims_ctx = "\n".join(
        f"- {'[VERIFIED]' if c.verified else '[UNVERIFIED - hedge]'} {c.text}"
        for c in claims
    )

    # Build hook instruction — tell the writer to create one, don't pre-generate it
    top_claim = claims[0].text if claims else ""
    hook_instruction = f"""HOOK REQUIREMENT (this is tweet 1 / opening paragraph):
Write a hook that scores 9/10 on these criteria:
- tension: creates an unresolved question the reader CANNOT ignore
- specificity: uses a concrete number, name, or fact — never a vague claim
- pattern: pick ONE of these:
  * counterintuitive: "[what everyone believes]. The data says otherwise: [specific contradiction]"
  * specific_number: "[surprising number]. [why that number matters]"  
  * bold_prediction: "By [year], [specific falsifiable claim]"

NEVER open with: "Most people...", "In today's world...", "The role of X is undergoing..."
ALWAYS open with: a specific number, a named contradiction, or a falsifiable prediction.

Best claim to anchor the hook: {top_claim[:200]}"""

    if output_type in (OutputType.x_thread, OutputType.quote_rt):
        system = f"""You are a ghostwriter. Write in the exact voice described.{voice_ctx}

RULES:
- Every tweet ≤ 280 characters
- No filler tweets — every tweet = one non-obvious idea
- NEVER use or mention these phrases (follow silently, do not reference this rule): in conclusion, game-changer, dive into, seamless, leverage
- Write a strong hook following the HOOK REQUIREMENT below
- End with a question or provocation
- 6-8 tweets for a thread, 1-3 for quote RT
- Return ONLY a JSON array, no explanation"""

        if output_type == OutputType.quote_rt and original_tweet:
            prompt = f"""Write a quote-retweet response. Do NOT summarise — add something new.

ORIGINAL TWEET: {original_tweet}

{hook_instruction}
NARRATIVE: {thesis}
CLAIMS:
{claims_ctx}
{extra_instruction}

Return JSON array: [{{"position": 1, "text": "...", "char_count": N, "claim_ids": [], "is_hook": true, "is_cta": false}}]"""
        else:
            prompt = f"""Write a Twitter thread.

{hook_instruction}
NARRATIVE ANGLE: {thesis}
RESEARCH CLAIMS TO USE:
{claims_ctx}
{extra_instruction}

Return JSON array: [{{"position": 1, "text": "...", "char_count": N, "claim_ids": [], "is_hook": true, "is_cta": false}}]"""

        raw = await _groq_call(prompt, system, max_tokens=1500)
        draft = _parse_tweets(raw)

        # Fallback if parsing failed
        if not draft:
            draft = [Tweet(position=1, text=top_claim[:280] if top_claim else f"Here is what the data shows about {state["input_context"].raw_input[:60]}:", char_count=min(len(top_claim), 280), is_hook=True)]
            for i, c in enumerate(claims[:6], 2):
                t = c.text[:280]
                draft.append(Tweet(position=i, text=t, char_count=len(t), claim_ids=[c.id]))
            draft.append(Tweet(
                position=len(draft)+1,
                text=f"What's your take on {state['input_context'].raw_input[:60]}?",
                char_count=80, is_cta=True,
            ))

        # Validate char counts
        for t in draft:
            if len(t.text) > 280:
                t.text = t.text[:277] + "..."
                t.char_count = 280

    else:
        # Essay / analysis / strategic narrative
        # Build a sources-only URL list to prevent hallucination
        allowed_urls = "\n".join(f"- {s.url}" for s in state.get("sources", []))

        system = f"""You are a ghostwriter producing long-form content.{voice_ctx}

RULES:
1. FIRST SENTENCE must be a strong hook — one of these patterns ONLY:
   - A specific number: "73% of data scientists still can't [X]. Here's what the gap actually is."
   - A named contradiction: "Every job posting asks for SQL and Python. That's not the skill gap."
   - A bold falsifiable claim: "By 2026, [specific prediction about this topic]."
   NEVER start with: "The field of X is undergoing...", "The role of X is...", "In today's world..."

2. Every paragraph earns its place — one idea per paragraph, no filler.

3. SOURCE CITATION RULE — CRITICAL:
   Only cite URLs from this exact list. Do NOT invent any other URLs:
{allowed_urls}
   Format: [Source: URL]. If you don't have a source for a claim, state it as your own analysis.

4. End the piece with a direct statement or open question — do NOT write a conclusion section.
   Do NOT write any heading that contains the words "conclusion", "summary", "final", "closing", "rephrase", "not allowed".

5. NEVER use: in conclusion, game-changer, seamless, leverage, dive into, delve, it's worth noting."""

        prompt = f"""Write a {output_type.value.replace('_', ' ')} about: {state_topic}

{hook_instruction}

NARRATIVE ANGLE: {thesis}
RESEARCH CLAIMS (verified sources only):
{claims_ctx}
{f"ORIGINAL CONTENT: {original_tweet}" if original_tweet else ""}
{extra_instruction}

Write the full piece in markdown. Start immediately with the hook — no title, no preamble."""

        raw = await _groq_call(prompt, system, max_tokens=2000)
        draft = raw if raw else f"# {thesis}\n\n" + "\n\n".join(f"{c.text}" for c in claims)

    # Post-process: strip any leaked rule references from essay output
    if isinstance(draft, str):
        import re as _re
        # Remove any heading that acknowledges a rule or acts as a conclusion
        draft = _re.sub(
            r'^#{1,3} .*?(conclusion|not allowed|not necessary|rephrase|do not use|removed|replaced|final statement|closing).*$',
            '',
            draft,
            flags=_re.IGNORECASE | _re.MULTILINE,
        ).strip()
        # Clean up double blank lines left by removed headings
        draft = _re.sub(r'\n{3,}', '\n\n', draft).strip()

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="writer", model="llama-3.3-70b-versatile (groq, 1 call)",
        duration_ms=duration_ms,
        detail=f"iteration={iteration}, output_type={output_type.value}, drafts=1",
    )
    return {"draft": draft, "iteration_count": iteration, "status": "writing_done", "trace": [event]}