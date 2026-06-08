"""
agent/nodes/researcher.py — Layer 2: RAG pipeline with input-type branching.

Flow per input_type:
  tweet_url / quote_rt  → fetch tweet (jina→nitter) → inject as anchor, skip arXiv
  web_url               → jina-fetch article → inject as anchor
  file_upload           → uploaded_content as anchor, web search for context
  voice_archive         → build FAISS index for user, skip web search
  instruction (default) → web search + clean rerank query

Common tail: query-expand → jina-enrich → embed-rerank → token-budget → cache
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from agent.schemas import AgentState, InputContext, InputType, OutputType, Source, TraceEvent
from agent.tools import (
    arxiv_search,
    deduplicate_sources,
    exa_search,
    hn_search,
    jina_fetch,
    score_source,
    tavily_search,
)

logger = logging.getLogger(__name__)

RERANK_TOP_K        = 6
MAX_CONTEXT_TOKENS  = 6_000
CHARS_PER_TOKEN     = 4
CHUNK_WORDS         = 400
MAX_JINA_ENRICHMENTS = 3

# ---------------------------------------------------------------------------
# URL / tweet detection
# ---------------------------------------------------------------------------

_TWITTER_URL_RE = re.compile(
    r"https?://(www\.)?(twitter\.com|x\.com)/(\w+)/status/(\d+)"
)
_URL_RE = re.compile(r"https?://\S+")

# Mode-label words that should NOT pollute the rerank query
_MODE_LABEL_RE = re.compile(
    r"\b(write|create|generate|make|produce|draft|compose|give me)?\s*"
    r"(an?|the)?\s*"
    r"(x[\s_-]?thread|twitter[\s_-]?thread|quote[\s_-]?rt|quote[\s_-]?retweet|"
    r"quote[\s_-]?tweet|thread|essay|blog\s*post|anal[iy]s[ie]s|analytical\s*report|"
    r"strategic\s*narrative|newsletter|article|post)\b",
    re.IGNORECASE,
)

# Common role/instruction prefixes that add noise
_INSTRUCTION_PREFIX_RE = re.compile(
    r"^(suppose\s+you\s+are|you\s+are\s+a|you\s+need\s+to|as\s+a|acting\s+as)\s+[^,.:]+[,.:]\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

async def _fetch_tweet_text(url: str) -> str:
    """
    Fetch tweet content.
    Stage 1: Jina reader (r.jina.ai/{url})
    Stage 2: Nitter fallback via Jina (nitter.privacydev.net/{user}/status/{id})
    Returns cleaned text or empty string.
    """
    # Stage 1: Jina on original URL
    text = await jina_fetch(url, timeout=20)
    if text and len(text.strip()) >= 50:
        return text.strip()[:2000]

    # Stage 2: Nitter via Jina
    m = _TWITTER_URL_RE.search(url)
    if m:
        username, status_id = m.group(3), m.group(4)
        nitter_url = f"https://nitter.privacydev.net/{username}/status/{status_id}"
        logger.info("Tweet: Jina failed/short — trying nitter: %s", nitter_url)
        text = await jina_fetch(nitter_url, timeout=20)
        if text and len(text.strip()) >= 50:
            return text.strip()[:2000]

    return ""


def _clean_rerank_query(raw_input: str) -> str:
    """
    Strip mode labels, instruction prefixes, and URLs from raw_input so the
    reranker embeds topic semantics, not output-format noise.
    """
    q = _INSTRUCTION_PREFIX_RE.sub("", raw_input.strip())
    q = _MODE_LABEL_RE.sub("", q).strip()
    q = _URL_RE.sub("", q).strip()
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q if len(q) >= 10 else raw_input   # fallback if cleaning empties it


def _handle_voice_archive(user_id: str, content: str) -> None:
    """
    Chunk voice samples, build/update FAISS index + texts file for user_id.
    Used by writer_node to load personalised few-shot examples.
    """
    chunks = _chunk_content(content)
    if not chunks:
        return
    try:
        from memory.embedder import build_faiss_index

        faiss_dir = Path("memory/faiss")
        faiss_dir.mkdir(parents=True, exist_ok=True)
        index_path = str(faiss_dir / f"{user_id}.index")
        texts_path = faiss_dir / f"{user_id}_texts.json"

        # Merge with existing chunks (incremental update)
        existing: list[str] = []
        if texts_path.exists():
            try:
                existing = json.loads(texts_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        all_texts = existing + chunks

        build_faiss_index(all_texts, index_path)
        texts_path.write_text(json.dumps(all_texts), encoding="utf-8")
        logger.info("Voice archive: FAISS index built for user %s (%d total chunks)",
                    user_id, len(all_texts))
    except Exception as exc:
        logger.warning("Voice archive FAISS build failed: %s", exc)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

_DISSENT_SIGNALS = re.compile(
    r"\b(wrong|overrated|actually|critique|criticism|counterintuitive|contrarian|"
    r"debunk|myth|misunderstood|overlooked|underrated|nuance|pushback|rebuttal)\b",
    re.IGNORECASE,
)


async def _expand_query(topic: str) -> list[str]:
    """One cheap llama-stack call → [original, sub1, dissent_variant]. Falls back silently."""
    from agent.llm import call_llama_stack
    prompt = (
        f'Generate 1 focused search sub-query to find evidence for this topic.\n'
        f'Topic: "{topic[:300]}"\n'
        f'Return ONLY a JSON array of exactly 1 string. No explanation.'
    )
    raw = await call_llama_stack(prompt, max_tokens=60)
    queries = [topic]
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        sub = json.loads(raw)
        if isinstance(sub, list) and sub:
            q = str(sub[0]).strip()
            if q:
                queries.append(q)
    except Exception:
        pass
    # Always add one dissent-hunting query — hunts for non-consensus material
    clean_topic = _URL_RE.sub("", topic[:150]).strip()
    queries.append(f"{clean_topic} criticism counterintuitive contrarian")
    return queries


def _apply_dissent_boost(sources: list[dict]) -> list[dict]:
    """Boost sources that contain dissent/surprise signals so they rank above consensus summaries."""
    for s in sources:
        content = s.get("content", "")
        signal_count = len(_DISSENT_SIGNALS.findall(content))
        if signal_count >= 2:
            s = dict(s)  # don't mutate the original
            s["credibility_score"] = min(10.0, s.get("credibility_score", 6.5) + 0.5 * min(signal_count, 3))
    return sorted(sources, key=lambda x: x.get("credibility_score", 0), reverse=True)


# ---------------------------------------------------------------------------
# Chunking + reranking + token budget
# ---------------------------------------------------------------------------

def _chunk_content(content: str, words: int = CHUNK_WORDS) -> list[str]:
    word_list = content.split()
    return [
        " ".join(word_list[i:i + words])
        for i in range(0, len(word_list), words)
        if word_list[i:i + words]
    ]


def _rerank_sources(
    sources_raw: list[dict],
    queries: list[str],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    if not sources_raw:
        return []
    try:
        from memory.embedder import embed_texts
        import numpy as np

        all_chunks: list[dict] = []
        for s in sources_raw:
            content = s.get("content", "")
            for chunk in (_chunk_content(content) or [content[:800]]):
                if chunk.strip():
                    all_chunks.append({**s, "content": chunk})

        if not all_chunks:
            return sources_raw[:top_k]

        q_embs = embed_texts(queries)
        c_embs = embed_texts([c["content"] for c in all_chunks])
        sim    = (c_embs @ q_embs.T).max(axis=1)

        best: dict[str, tuple[float, dict]] = {}
        for score, chunk in zip(sim.tolist(), all_chunks):
            url = chunk.get("url", "")
            if url not in best or score > best[url][0]:
                best[url] = (score, chunk)

        ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)[:top_k]
        logger.debug("Reranker: %d/%d sources kept (embedding)", len(ranked), len(best))
        return [item[1] for item in ranked]

    except Exception as exc:
        logger.warning("Reranker unavailable (%s) — keyword fallback", exc)
        stop = {"the", "a", "an", "in", "of", "to", "and", "is", "are"}
        q_words = set(" ".join(queries).lower().split()) - stop

        def kw_score(s: dict) -> int:
            text = (s.get("title", "") + " " + s.get("content", ""))[:2000].lower()
            return len(q_words & set(text.split()))

        sources_raw.sort(key=kw_score, reverse=True)
        return sources_raw[:top_k]


def _enforce_token_budget(sources: list[dict]) -> list[dict]:
    max_chars = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN
    result, total = [], 0
    for s in sources:
        content = s.get("content", "")
        if total + len(content) > max_chars:
            remaining = max_chars - total
            if remaining > CHARS_PER_TOKEN * 100:
                s = {**s, "content": content[:remaining]}
                logger.warning("Token budget: source %s truncated (~%d tokens remaining)",
                               s.get("url", "?")[:50], remaining // CHARS_PER_TOKEN)
                result.append(s)
            break
        total += len(content)
        result.append(s)
    logger.debug("Token budget: %d sources, ~%d tokens total", len(result), total // CHARS_PER_TOKEN)
    return result


# ---------------------------------------------------------------------------
# Individual research agents
# ---------------------------------------------------------------------------

async def _agent_a(topic: str) -> list[dict]:
    # Run Tavily and Exa concurrently — previously serial with 0.5s sleep between them
    tavily_task = asyncio.create_task(
        tavily_search(f"{topic} counterarguments analysis", max_results=4)
    )
    exa_task = asyncio.create_task(exa_search(topic, num_results=3))
    tavily_raw, exa_raw = await asyncio.gather(tavily_task, exa_task, return_exceptions=True)

    results: list[dict] = []
    if not isinstance(tavily_raw, Exception):
        for r in tavily_raw:
            results.append({
                "url":               r.get("url", ""),
                "title":             r.get("title", ""),
                "content":           r.get("content", "")[:1500],
                "source_type":       "web",
                "credibility_score": score_source(r.get("url", ""), "web"),
            })
    else:
        logger.warning("Tavily agent failed: %s", tavily_raw)

    if not isinstance(exa_raw, Exception):
        for r in exa_raw:
            results.append({
                "url":               r.get("url", ""),
                "title":             r.get("title", ""),
                "content":           (r.get("text") or r.get("content") or "")[:1500],
                "source_type":       "exa",
                "credibility_score": score_source(r.get("url", ""), "exa"),
            })
    else:
        logger.warning("Exa agent failed: %s", exa_raw)

    return results


async def _agent_b(topic: str) -> list[dict]:
    hn = await hn_search(topic, min_score=100, max_items=5)
    return [{
        "url":               r.get("url", ""),
        "title":             r.get("title", ""),
        "content":           r.get("content", "")[:1500],
        "source_type":       "hn",
        "credibility_score": score_source(r.get("url", ""), "hn"),
    } for r in hn]


async def _agent_c(topic: str) -> list[dict]:
    stop = {"the", "a", "an", "for", "of", "in", "to", "and", "is", "are", "what"}
    topic_words = set(topic.lower().split()) - stop
    filtered = []
    for r in await arxiv_search(topic, max_results=8):
        words = set(r["title"].lower().split()) | set(r["content"].lower().split())
        if len(topic_words & words) >= 2:
            filtered.append({
                "url":               r["url"],
                "title":             r["title"],
                "content":           r["content"],
                "source_type":       "arxiv",
                "credibility_score": 8.5,
            })
    logger.debug("arXiv filter: %d kept for: %s", len(filtered), topic[:60])
    return filtered


async def _enrich_with_jina(sources: list[dict], max_enrich: int = MAX_JINA_ENRICHMENTS) -> list[dict]:
    # Previously serial (3 × up to 25s = 75s worst case). Now parallel with 10s cap each.
    to_enrich = [
        s for s in sources
        if len(s.get("content", "")) < 500 and s.get("url")
    ][:max_enrich]

    async def _fetch_one(s: dict) -> None:
        try:
            text = await asyncio.wait_for(jina_fetch(s["url"]), timeout=10.0)
            if text:
                s["content"] = text[:3000]
        except asyncio.TimeoutError:
            logger.debug("Jina enrich timeout: %s", s.get("url", "")[:60])
        except Exception as exc:
            logger.debug("Jina enrich error: %s — %s", s.get("url", "")[:60], exc)

    if to_enrich:
        await asyncio.gather(*[_fetch_one(s) for s in to_enrich])
    return sources


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

async def researcher_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    input_ctx   = state["input_context"]
    topic       = input_ctx.raw_input
    run_id      = state.get("run_id", "unknown")
    user_id     = state.get("user_id", "default")
    output_type = input_ctx.output_type
    input_type  = input_ctx.input_type

    # ── Run-cache check: skip research if already done for this run ───────────
    from agent.memory import get_run_context, save_run_context
    cached = get_run_context(run_id)
    if cached and cached.get("sources"):
        sources = []
        for sd in cached["sources"]:
            try:
                sources.append(Source(**sd))
            except Exception:
                pass
        if sources:
            logger.info("[researcher] Cache hit: reusing %d sources for run %s",
                        len(sources), run_id[:8])
            event = TraceEvent(
                node="researcher", model="cache",
                duration_ms=(time.monotonic() - t0) * 1000,
                detail=f"Reused {len(sources)} cached sources",
            )
            return {"sources": sources, "status": "research_done", "trace": [event]}

    # ── Per-input-type branching ───────────────────────────────────────────────
    anchor_sources: list[dict] = []   # always prepended (highest priority)
    rerank_query  = _clean_rerank_query(topic)
    updated_input_ctx: InputContext | None = None
    do_web_search = True
    do_arxiv      = True

    is_tweet_url = bool(_TWITTER_URL_RE.search(topic))

    if input_type == InputType.voice_archive:
        # Build FAISS voice index; use samples as the only sources
        _handle_voice_archive(user_id, input_ctx.uploaded_content or "")
        content = input_ctx.uploaded_content or ""
        if content:
            anchor_sources.append({
                "url":               f"voicearchive://{user_id}",
                "title":             "Voice Writing Samples",
                "content":           content[:3000],
                "source_type":       "file",
                "credibility_score": 7.0,
            })
        do_web_search = False
        do_arxiv      = False

    elif input_type == InputType.file_upload and input_ctx.uploaded_content:
        # Uploaded doc is the primary anchor; still run web search for context
        anchor_sources.append({
            "url":               "uploaded://document",
            "title":             "Uploaded Document",
            "content":           input_ctx.uploaded_content[:3000],
            "source_type":       "file",
            "credibility_score": 7.0,
        })
        do_arxiv = False   # skip arXiv — uploaded docs are the ground truth

    elif (output_type == OutputType.quote_rt or input_type == InputType.tweet_url) and is_tweet_url:
        # ── BUG 1 FIX: fetch the actual tweet content ─────────────────────────
        logger.info("[researcher] quote_rt: fetching tweet %s", topic)
        tweet_text = await _fetch_tweet_text(topic)
        if not tweet_text or len(tweet_text.strip()) < 50:
            raise RuntimeError(
                "Could not fetch tweet content — paste the tweet text directly instead."
            )
        logger.info("[researcher] Tweet fetched (%d chars)", len(tweet_text))
        anchor_sources.append({
            "url":               topic,
            "title":             "Original Tweet",
            "content":           tweet_text,
            "source_type":       "tweet",
            "credibility_score": 10.0,   # always survives rerank
        })
        # ── BUG 2 FIX: use tweet text for reranking, not the URL ─────────────
        rerank_query = tweet_text[:400]
        do_arxiv = False  # academic papers irrelevant for quote_rt

        # Update InputContext so writer sees original_tweet populated
        updated_input_ctx = InputContext(
            raw_input        = input_ctx.raw_input,
            input_type       = InputType.tweet_url,
            output_type      = input_ctx.output_type,
            original_tweet   = tweet_text,
            uploaded_content = input_ctx.uploaded_content,
            user_instruction = input_ctx.user_instruction,
            target_tone      = input_ctx.target_tone,
        )

    elif input_type == InputType.web_url or (_URL_RE.match(topic) and not is_tweet_url):
        # Non-tweet URL: jina-fetch the article as primary source
        logger.info("[researcher] web_url: fetching article %s", topic[:80])
        article_text = await jina_fetch(topic)
        if article_text:
            anchor_sources.append({
                "url":               topic,
                "title":             "Source Article",
                "content":           article_text[:3000],
                "source_type":       "web",
                "credibility_score": 8.0,
            })
            rerank_query = article_text[:400]   # article summary drives reranking

    # ── Query expansion (concurrent with research agents) ────────────────────
    queries_task = asyncio.create_task(_expand_query(rerank_query))

    # ── Research agents ────────────────────────────────────────────────────────
    search_query = rerank_query[:200]

    async def _safe(coro, name: str) -> list:
        try:
            result = await asyncio.wait_for(coro, timeout=35.0)
            logger.debug("[researcher] Agent %s → %d results", name, len(result or []))
            return result or []
        except asyncio.TimeoutError:
            logger.warning("Research agent %s timed out", name)
            return []
        except Exception as exc:
            logger.warning("Research agent %s failed: %s", name, exc)
            return []

    if do_web_search:
        if do_arxiv:
            logger.info("[researcher] Running web + HN + arXiv agents…")
            results_a, results_b, results_c = await asyncio.gather(
                _safe(_agent_a(search_query), "A-tavily/exa"),
                _safe(_agent_b(search_query), "B-hn"),
                _safe(_agent_c(search_query), "C-arxiv"),
            )
        else:
            logger.info("[researcher] Running web + HN agents (arXiv skipped)…")
            results_a, results_b = await asyncio.gather(
                _safe(_agent_a(search_query), "A-tavily/exa"),
                _safe(_agent_b(search_query), "B-hn"),
            )
            results_c = []
    else:
        logger.info("[researcher] Web search skipped (voice_archive input)")
        results_a = results_b = results_c = []

    queries = await queries_task
    logger.info("[researcher] Queries: %s", queries[:3])

    all_raw: list[dict] = []
    for r in (results_a, results_b, results_c):
        all_raw.extend(r)

    # ── Deduplicate + Jina enrich before reranking ────────────────────────────
    unique = deduplicate_sources(all_raw)
    unique = [s for s in unique if s.get("url")]
    unique = await _enrich_with_jina(unique, max_enrich=MAX_JINA_ENRICHMENTS)

    # Anchor sources skip reranking — add AFTER so they always survive
    # But rerank only the web results, then prepend anchors
    loop     = asyncio.get_running_loop()
    reranked = await loop.run_in_executor(
        None, _rerank_sources, unique, queries,
        max(0, RERANK_TOP_K - len(anchor_sources)),   # leave slots for anchors
    )

    # Boost reranked results that contain dissent signals — then anchors always first
    reranked = _apply_dissent_boost(reranked)
    final_raw = anchor_sources + reranked

    # ── Token budget ──────────────────────────────────────────────────────────
    final_raw = _enforce_token_budget(final_raw)

    # ── Convert to Source objects ─────────────────────────────────────────────
    sources: list[Source] = []
    for s in final_raw:
        if s.get("url") and s.get("title"):
            sources.append(Source(
                url=s["url"],
                title=s["title"],
                content=s.get("content", "")[:4000],
                source_type=s.get("source_type", "web"),
                credibility_score=min(s.get("credibility_score", 6.5), 10.0),
            ))

    # ── Persist to run cache ───────────────────────────────────────────────────
    save_run_context(run_id, [s.model_dump() for s in sources], [], "")

    total_chars = sum(len(s.content) for s in sources)
    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="researcher",
        model=f"tavily+exa+hn+arxiv+rerank({len(queries)}q)",
        duration_ms=duration_ms,
        detail=(
            f"Collected {len(sources)} sources "
            f"({total_chars} chars, ~{total_chars // CHARS_PER_TOKEN} tokens)"
        ),
    )
    result: dict = {"sources": sources, "status": "research_done", "trace": [event]}
    if updated_input_ctx is not None:
        result["input_context"] = updated_input_ctx
    return result
