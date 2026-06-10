"""
agent/nodes/analyst.py — Single LLM call: extract source-grounded claims + thesis.
Speculative-input detection prevents hypothesis→fact promotion.
"""
from __future__ import annotations

import json
import logging
import re
import time
from uuid import uuid4

from agent.llm import call_llama_stack
from agent.schemas import AgentState, Claim, Source, TraceEvent

logger = logging.getLogger(__name__)

# Topics that start with these patterns are treated as hypothetical — all
# derived claims get speculative=True, verified=False.
_SPECULATIVE_RE = re.compile(
    r"^\s*(suppose\b|imagine\b|what if\b|what would happen if\b|hypothetical\b|"
    r"pretend\b|let'?s say\b|assume (that|you|we)\b|if (this|x|y|that) happened\b)",
    re.IGNORECASE,
)


def _is_speculative(topic: str) -> bool:
    return bool(_SPECULATIVE_RE.search(topic))


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


async def analyst_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    sources: list[Source] = state.get("sources", [])

    if not sources:
        event = TraceEvent(node="analyst", model="none", duration_ms=0,
                           detail="No sources — empty claims")
        return {"claims": [], "chosen_thesis": "Explore this topic from first principles.",
                "status": "analysis_done", "trace": [event]}

    topic    = state["input_context"].raw_input or "the topic"
    focus    = state["input_context"].user_instruction or ""
    is_spec  = _is_speculative(topic)

    if is_spec:
        logger.info("analyst: speculative topic detected — all claims will be marked speculative")

    # Assign stable source IDs (S1, S2, ...) and build URL → ID map
    source_id_map: dict[str, str] = {s.url: f"S{i+1}" for i, s in enumerate(sources)}

    # Build sources text with IDs labelled so LLM can reference them
    sources_text = "\n\n---\n\n".join(
        f"[{source_id_map[s.url]}] [{s.source_type.upper()}] {s.title}\n"
        f"URL: {s.url}\n{s.content[:800]}"
        for s in sources
    )

    system = (
        "You are a research analyst. Extract only claims DIRECTLY supported by the provided sources. "
        "Never invent facts. If a claim cannot be tied to a source snippet, omit it. "
        "Return ONLY valid JSON."
    )
    prompt = f"""Topic: "{topic[:400]}"
{f'Writing focus: "{focus}"' if focus else ""}
{"NOTE: This topic is hypothetical. Mark all insights with low surprise scores." if is_spec else ""}

For each insight, cite the source ID (S1, S2, …) it comes from.

Return JSON:
{{"insights": [{{"claim": "...", "source_url": "exact URL from sources above", "source_id": "S1", "surprise_score": 7.5, "contradiction_flag": false}}], "angle": "one concise narrative angle"}}

SOURCES:
{sources_text}"""

    logger.debug("analyst: prompt (first 300) = %r", prompt[:300])
    raw  = await call_llama_stack(prompt, system, max_tokens=1200)
    logger.debug("analyst: raw response (first 300) = %r", raw[:300])
    data = _parse(raw)
    if not data:
        logger.warning("analyst: JSON parse failed — full raw response: %s", raw)

    valid_urls = {s.url for s in sources}
    claims: list[Claim] = []
    seen:   list[str]   = []

    for ins in data.get("insights", []):
        text = ins.get("claim", "").strip()
        if not text:
            continue
        # Dedup by Jaccard similarity
        if any(
            len(set(text.lower().split()) & set(s.lower().split())) /
            max(len(set(text.lower().split()) | set(s.lower().split())), 1) > 0.8
            for s in seen
        ):
            continue
        seen.append(text)

        src_url = ins.get("source_url", "")
        # Prefer source_id from LLM response; fall back to URL lookup
        sid = ins.get("source_id", "") or source_id_map.get(src_url, "")
        # If sid references a source that doesn't own the url, prefer the url-derived one
        if sid and src_url and src_url in source_id_map and source_id_map[src_url] != sid:
            sid = source_id_map[src_url]

        claims.append(Claim(
            id=str(uuid4()),
            text=text,
            source_url=src_url if src_url in valid_urls else "",
            source_id=sid,
            verified=(src_url in valid_urls) and not is_spec,
            speculative=is_spec,
            surprise_score=min(max(float(ins.get("surprise_score", 5.0)), 1.0), 10.0),
            contradiction_flag=bool(ins.get("contradiction_flag", False)),
        ))

    claims.sort(key=lambda c: c.surprise_score, reverse=True)
    top_claims = claims[:6]

    thesis = data.get("angle") or (
        "This is a hypothetical scenario — treat all claims as speculative."
        if is_spec
        else "The mainstream view is incomplete — here's what the data shows."
    )

    # Update run cache with extracted claims
    try:
        from agent.memory import get_run_context, save_run_context
        run_id = state.get("run_id", "")
        if run_id:
            ctx = get_run_context(run_id) or {}
            save_run_context(
                run_id,
                ctx.get("sources", [s.model_dump() for s in sources]),
                [c.model_dump() for c in top_claims],
                thesis,
            )
    except Exception as exc:
        logger.debug("analyst: memory update skipped: %s", exc)

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="analyst", model="llama-3.3-70b (llama-stack)",
        duration_ms=duration_ms,
        detail=(
            f"Extracted {len(top_claims)} claims "
            f"({sum(1 for c in top_claims if c.verified)} verified, "
            f"speculative={is_spec})"
        ),
    )
    return {"claims": top_claims, "chosen_thesis": thesis,
            "status": "analysis_done", "trace": [event]}
