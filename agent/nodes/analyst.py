"""
agent/nodes/analyst.py — Layer 3: Single Groq call for claim extraction + thesis.
Consolidated to 1 LLM call to preserve free tier quota.
"""
from __future__ import annotations

import json
import logging
import os
import time
from uuid import uuid4

import httpx

from agent.schemas import AgentState, Claim, Source, TraceEvent

logger = logging.getLogger(__name__)


async def _groq_call(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    # Support two Groq keys — if primary is exhausted, try secondary
    keys = [k for k in [
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GROQ_API_KEY_2", ""),
    ] if k]
    if not keys:
        logger.error("No GROQ_API_KEY set")
        return "{}"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for api_key in keys:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": max_tokens},
                    )
                    if resp.status_code == 429:
                        retry_after = min(int(resp.headers.get("retry-after", 30)), 60)
                        # If daily limit (retry-after > 3600), skip to next key immediately
                        if int(resp.headers.get("retry-after", 0)) > 300:
                            logger.warning("Groq key exhausted (daily limit) — trying next key")
                            break
                        logger.warning("Groq 429 — waiting %ds (attempt %d/2)", retry_after, attempt + 1)
                        import asyncio; await asyncio.sleep(retry_after + 1)
                        continue
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.error("Groq call failed (attempt %d): %s", attempt + 1, exc)
                import asyncio; await asyncio.sleep(3)
    return "{}"


def _parse_json(text: str) -> dict:
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


def _rough_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


async def analyst_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    sources: list[Source] = state.get("sources", [])

    if not sources:
        event = TraceEvent(node="analyst", model="none", duration_ms=0,
                           detail="No sources — returning empty claims")
        return {"claims": [], "thesis_options": ["Explore this topic from first principles."],
                "chosen_thesis": "Explore this topic from first principles.",
                "status": "analysis_done", "trace": [event]}

    # Single consolidated LLM call — analyst + thesis in one shot
    state_topic = (
        state.get("input_context").user_instruction
        or state.get("input_context").raw_input
        or "the topic"
    )
    sources_text = "\n\n---\n\n".join(
        f"[{s.source_type.upper()}] {s.title}\nURL: {s.url}\n{s.content[:600]}"
        for s in sources
    )

    system = """You are a research analyst. Extract insights and narrative angles from sources.
CRITICAL: Only extract claims directly relevant to the user's topic. 
If a source is about a completely different subject, ignore it entirely.
Return ONLY valid JSON, no markdown, no explanation."""

    prompt = f"""Topic: "{state_topic}"

Analyse the sources below and extract only insights DIRECTLY relevant to this topic.
If a source is about something unrelated (e.g. physics papers for a business topic), skip it.

Return JSON:
{{"insights": [{{"claim": "...", "source_url": "...", "surprise_score": 7.5, "contradiction_flag": false}}], "angles": ["...", "...", "..."]}}

SOURCES:
{sources_text}"""

    logger.info("[analyst] Single consolidated Groq call")
    raw = await _groq_call(prompt, system, max_tokens=1500)
    data = _parse_json(raw)

    valid_urls = {s.url for s in sources}
    claims: list[Claim] = []
    seen: list[str] = []

    for ins in data.get("insights", []):
        text = ins.get("claim", "").strip()
        if not text or any(_rough_similarity(text, t) > 0.8 for t in seen):
            continue
        seen.append(text)
        src_url = ins.get("source_url", "")
        claims.append(Claim(
            id=str(uuid4()),
            text=text,
            source_url=src_url if src_url in valid_urls else "",
            verified=src_url in valid_urls,
            surprise_score=min(max(float(ins.get("surprise_score", 5.0)), 1.0), 10.0),
            contradiction_flag=bool(ins.get("contradiction_flag", False)),
        ))

    claims.sort(key=lambda c: c.surprise_score, reverse=True)
    top_claims = claims[:6]

    thesis_options = data.get("angles", [
        "The mainstream view is incomplete — here's what the data actually shows.",
        "Three things are changing simultaneously that most people haven't connected.",
        "The real skill gap isn't what anyone is talking about.",
    ])[:3]

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="analyst", model="llama-3.3-70b-versatile (groq, 1 call)",
        duration_ms=duration_ms,
        detail=f"Extracted {len(top_claims)} claims ({sum(1 for c in top_claims if c.verified)} verified)",
    )
    return {"claims": top_claims, "thesis_options": thesis_options,
            "chosen_thesis": thesis_options[0],
            "status": "analysis_done", "trace": [event]}