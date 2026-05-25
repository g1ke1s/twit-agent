"""
agent/nodes/researcher.py — Layer 2: 3 concurrent research agents.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent.schemas import AgentState, InputType, Source, TraceEvent
from agent.tools import (
    arxiv_search,
    deduplicate_sources,
    exa_search,
    hn_search,
    jina_fetch,
    rate_limited_call,
    score_source,
    tavily_search,
)

logger = logging.getLogger(__name__)

MAX_SOURCES = 8
MAX_JINA_ENRICHMENTS = 4
MAX_CONTEXT_CHARS = 24_000


# ---------------------------------------------------------------------------
# Individual agents
# ---------------------------------------------------------------------------

async def _agent_a(topic: str) -> list[dict]:
    """Tavily + Exa web search."""
    results: list[dict] = []

    try:
        tavily_results = await tavily_search(f"{topic} counterarguments analysis", max_results=4)
        for r in tavily_results:
            results.append({
                "url":              r.get("url", ""),
                "title":            r.get("title", ""),
                "content":          r.get("content", "")[:1500],
                "source_type":      "web",
                "credibility_score": score_source(r.get("url", ""), "web"),
            })
    except Exception as exc:
        logger.warning("Tavily agent failed: %s", exc)

    await asyncio.sleep(0.5)  # rate limit gap

    try:
        exa_results = await exa_search(topic, num_results=3)
        for r in exa_results:
            results.append({
                "url":              r.get("url", ""),
                "title":            r.get("title", ""),
                "content":          (r.get("text") or r.get("content") or "")[:1500],
                "source_type":      "exa",
                "credibility_score": score_source(r.get("url", ""), "exa"),
            })
    except Exception as exc:
        logger.warning("Exa agent failed: %s", exc)

    return results


async def _agent_b(topic: str) -> list[dict]:
    """Discourse: HN Algolia."""
    hn_results = await hn_search(topic, min_score=100, max_items=5)
    out = []
    for r in hn_results:
        out.append({
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", "")[:1500],
            "source_type": "hn",
            "credibility_score": score_source(r.get("url", ""), "hn"),
        })
    return out


async def _agent_c(topic: str) -> list[dict]:
    """Academic: arXiv — with topic relevance filter."""
    arxiv_results = await arxiv_search(topic, max_results=8)

    # Filter: only keep results where title/content shares keywords with topic
    topic_words = set(topic.lower().split()) - {"the", "a", "an", "for", "of", "in", "to", "and", "is", "are", "what"}
    filtered = []
    for r in arxiv_results:
        title_words = set(r["title"].lower().split())
        content_words = set(r["content"].lower().split())
        overlap = len(topic_words & (title_words | content_words))
        if overlap >= 2:  # at least 2 topic keywords must appear
            filtered.append({
                "url": r["url"],
                "title": r["title"],
                "content": r["content"],
                "source_type": "arxiv",
                "credibility_score": 8.5,
            })

    logger.debug("arXiv relevance filter: %d/%d results kept for: %s",
                 len(filtered), len(arxiv_results), topic[:60])
    return filtered


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

async def _enrich_with_jina(sources: list[dict], max_enrich: int = MAX_JINA_ENRICHMENTS) -> list[dict]:
    """Fetch full article text via Jina for top sources that have short content."""
    enriched = 0
    for s in sources:
        if enriched >= max_enrich:
            break
        if len(s.get("content", "")) < 500 and s.get("url"):
            text = await jina_fetch(s["url"])
            if text:
                s["content"] = text[:2000]
                enriched += 1
    return sources


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

async def researcher_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    input_ctx = state["input_context"]
    topic = input_ctx.user_instruction or input_ctx.raw_input

    # Run 3 agents concurrently with a hard timeout per agent
    async def _safe(coro, name):
        try:
            logger.debug("[researcher] Starting agent %s", name)
            result = await asyncio.wait_for(coro, timeout=35.0)
            logger.debug("[researcher] Agent %s returned %d results", name, len(result or []))
            return result
        except asyncio.TimeoutError:
            logger.warning("Research agent %s timed out after 35s", name)
            return []
        except Exception as exc:
            logger.warning("Research agent %s failed: %s", name, exc)
            return []

    logger.info("[researcher] Starting 3 concurrent agents (arXiv, HN, Tavily/Exa)…")
    results_a, results_b, results_c = await asyncio.gather(
        _safe(_agent_a(topic), "A-tavily/exa"),
        _safe(_agent_b(topic), "B-hn"),
        _safe(_agent_c(topic), "C-arxiv"),
    )
    logger.info("[researcher] Agents done — A:%d B:%d C:%d results",
                len(results_a or []), len(results_b or []), len(results_c or []))

    all_raw: list[dict] = []
    for r in (results_a, results_b, results_c):
        if isinstance(r, list):
            all_raw.extend(r)
        else:
            logger.warning("Research agent returned exception: %s", r)

    # Deduplicate, sort, cap
    unique = deduplicate_sources(all_raw)
    unique = [s for s in unique if s.get("url")]
    unique.sort(key=lambda x: x.get("credibility_score", 0), reverse=True)
    top = unique[:MAX_SOURCES]

    # Jina enrich top-4
    top = await _enrich_with_jina(top, max_enrich=MAX_JINA_ENRICHMENTS)

    # Cap total context
    total_chars = 0
    final: list[dict] = []
    for s in top:
        c = s.get("content", "")
        if total_chars + len(c) > MAX_CONTEXT_CHARS:
            s["content"] = c[: MAX_CONTEXT_CHARS - total_chars]
            final.append(s)
            break
        total_chars += len(c)
        final.append(s)

    # Convert to Source objects
    sources: list[Source] = []
    for s in final:
        if s.get("url") and s.get("title"):
            sources.append(
                Source(
                    url=s["url"],
                    title=s["title"],
                    content=s.get("content", "")[:4000],
                    source_type=s.get("source_type", "web"),
                    credibility_score=s.get("credibility_score", 6.5),
                )
            )

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="researcher",
        model="tavily+exa+hn+arxiv",
        duration_ms=duration_ms,
        detail=f"Collected {len(sources)} sources ({total_chars} chars context)",
    )
    return {"sources": sources, "status": "research_done", "trace": [event]}