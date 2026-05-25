"""
agent/voice_profile.py — FAISS-backed VoiceProfile construction, similarity scoring,
and incremental learning from human feedback.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("sentence-transformers model loaded")
        except ImportError:
            logger.error("sentence-transformers not installed — pip install sentence-transformers")
            raise
    return _MODEL


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_avg_sentence_length(texts: list[str]) -> float:
    import re
    all_sentences = []
    for t in texts:
        sents = re.split(r"[.!?]+", t)
        all_sentences.extend(s.strip() for s in sents if s.strip())
    if not all_sentences:
        return 0.0
    return sum(len(s.split()) for s in all_sentences) / len(all_sentences)


def compute_ttr(texts: list[str]) -> float:
    """Type/token ratio — vocabulary richness."""
    all_words: list[str] = []
    for t in texts:
        all_words.extend(t.lower().split())
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def select_mmr(
    embeddings: np.ndarray,
    texts: list[str],
    k: int = 10,
    lambda_: float = 0.7,
) -> list[str]:
    """
    Maximal Marginal Relevance: pick k diverse exemplars.
    lambda_=1 → pure relevance, 0 → pure diversity.
    """
    if len(texts) <= k:
        return texts

    query = embeddings.mean(axis=0)
    selected_idxs: list[int] = []
    remaining = list(range(len(texts)))

    for _ in range(k):
        scores = []
        for idx in remaining:
            rel = float(np.dot(query, embeddings[idx]))
            sim_to_sel = max(
                (float(np.dot(embeddings[idx], embeddings[s])) for s in selected_idxs),
                default=0.0,
            )
            scores.append((lambda_ * rel - (1 - lambda_) * sim_to_sel, idx))
        best = max(scores, key=lambda x: x[0])[1]
        selected_idxs.append(best)
        remaining.remove(best)

    return [texts[i] for i in selected_idxs]


# ---------------------------------------------------------------------------
# LLM style extraction
# ---------------------------------------------------------------------------

async def _extract_style_with_llm(samples: list[str]) -> dict:
    """Call Groq Llama to extract style fingerprint from writing samples."""
    import httpx

    sample_text = "\n---\n".join(samples[:50])
    prompt = f"""You are a literary analyst. Study these {len(samples[:50])} writing samples from one person.

Extract a precise style fingerprint:

1. preferred_structures: list of 5-8 structural/stylistic patterns you observe
   (e.g. "Opens with a specific number", "Uses em-dash for asides",
   "Ends arguments with an open question", "Never uses passive voice",
   "Short declarative sentences followed by one longer explanation")

2. style_summary: 3 sentences a ghostwriter could use to impersonate this person.
   Be specific about rhythm, vocabulary, sentence length, how they open and close ideas.

3. forbidden_phrases: 8-12 words and constructions this person NEVER uses.
   Infer from absence and from how different their style is from generic writing.
   Include both words and structural patterns (e.g. "it's worth noting", "leverage",
   "in conclusion", passive constructions like "it was found that").

SAMPLES:
{sample_text}

Return ONLY a JSON object. No markdown, no preamble.
Keys: preferred_structures (list), style_summary (string), forbidden_phrases (list)"""

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    try:
        import asyncio
        import httpx as _httpx
        async def _call():
            async with _httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": "llama-3.3-70b-versatile",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 1000},
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        text = asyncio.get_event_loop().run_until_complete(_call())
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        logger.info("Style extraction complete: %d structures, %d forbidden phrases",
                    len(result.get("preferred_structures", [])),
                    len(result.get("forbidden_phrases", [])))
        return result
    except Exception as exc:
        logger.error("Style extraction LLM call failed: %s", exc)
        return {
            "preferred_structures": [
                "Opens with a specific data point or number",
                "Uses em-dash for clarifying asides",
                "Ends with an open question or unresolved tension",
                "Short sentences after a longer setup",
                "States opinions as facts, not as opinions",
            ],
            "style_summary": (
                "Clear, direct prose with strong opinions stated as facts. "
                "Favours specific numbers and named examples over abstractions. "
                "Ends arguments with an open question that invites the reader to think further."
            ),
            "forbidden_phrases": [
                "in conclusion", "it's worth noting", "game-changer",
                "dive into", "delve", "revolutionize", "seamless", "leverage",
                "as we've seen", "moving forward", "circle back",
            ],
        }


# ---------------------------------------------------------------------------
# Public build API
# ---------------------------------------------------------------------------

async def build_voice_profile(archive_texts: list[str], user_id: str) -> "VoiceProfile":
    from agent.schemas import VoiceProfile

    faiss_dir  = Path("memory/faiss")
    faiss_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(faiss_dir / f"{user_id}.index")

    logger.info("Building voice profile for %s from %d samples", user_id, len(archive_texts))

    avg_sent_len  = compute_avg_sentence_length(archive_texts)
    vocab_richness = compute_ttr(archive_texts)
    style_data    = await _extract_style_with_llm(archive_texts)

    # Build FAISS index
    try:
        import faiss
        model      = _get_model()
        embeddings = model.encode(archive_texts, show_progress_bar=False)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        index      = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, index_path)
        few_shot   = select_mmr(embeddings, archive_texts, k=10, lambda_=0.7)
        logger.info("FAISS index built: %d vectors at %s", index.ntotal, index_path)
    except ImportError:
        logger.warning("faiss-cpu not installed — using random few-shot examples")
        few_shot   = random.sample(archive_texts, min(10, len(archive_texts)))

    return VoiceProfile(
        avg_sentence_length  = avg_sent_len,
        vocabulary_richness  = vocab_richness,
        preferred_structures = style_data.get("preferred_structures", []),
        forbidden_phrases    = style_data.get("forbidden_phrases", []),
        style_summary        = style_data.get("style_summary", ""),
        few_shot_examples    = few_shot,
        faiss_index_path     = index_path,
    )


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------

def get_voice_similarity(draft_text: str, profile: "VoiceProfile") -> float:
    """
    Cosine similarity between draft and VoiceProfile FAISS index.
    Returns mean of top-5 nearest-neighbour cosine similarities (0–1).
    Returns 0.5 as neutral fallback on any error.
    """
    if not profile or not os.path.exists(profile.faiss_index_path):
        return 0.5
    try:
        import faiss
        model     = _get_model()
        index     = faiss.read_index(profile.faiss_index_path)
        emb       = model.encode([draft_text])
        emb       = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        distances, _ = index.search(emb.astype(np.float32), k=min(5, index.ntotal))
        return float(np.mean(distances[0]))
    except Exception as exc:
        logger.warning("get_voice_similarity failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Incremental learning from human feedback
# ---------------------------------------------------------------------------

async def update_voice_profile_from_feedback(
    original_draft: str,
    approved_output: str,
    profile: "VoiceProfile",
    user_id: str,
) -> None:
    """
    1. Add approved_output to FAISS index as gold-label example.
    2. Persist updated index to disk.
    3. Check if rebuild threshold (20 new entries) is reached.
       If so: fetch all approved outputs for this user from Supabase,
       rebuild the full VoiceProfile, and persist it.
    """
    if not approved_output.strip():
        return

    try:
        import faiss
        model = _get_model()
        index = faiss.read_index(profile.faiss_index_path)
        emb   = model.encode([approved_output])
        emb   = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        index.add(emb.astype(np.float32))
        faiss.write_index(index, profile.faiss_index_path)
        new_total = index.ntotal
        logger.info("Voice profile updated for %s — index size: %d", user_id, new_total)

        # Rebuild threshold: every 20 new approved outputs
        if new_total % 20 == 0:
            logger.info("Rebuild threshold reached (%d) for %s — triggering full rebuild", new_total, user_id)
            await _rebuild_voice_profile(user_id, profile)

    except ImportError:
        logger.warning("faiss not available — skipping voice profile update")
    except Exception as exc:
        logger.warning("Voice profile update failed for %s: %s", user_id, exc)


async def _rebuild_voice_profile(user_id: str, current_profile: "VoiceProfile") -> None:
    """
    Fetch all approved outputs for this user from Supabase feedback table,
    rebuild the VoiceProfile from scratch, and persist it.
    """
    try:
        from memory.store import _get_client, save_voice_profile
        client = _get_client()

        # Fetch all approved/edited outputs
        resp = (
            client.table("feedback")
            .select("approved_output")
            .eq("user_id", user_id)
            .in_("decision", ["approve", "edit"])
            .execute()
        )
        approved_texts = [
            row["approved_output"] for row in (resp.data or [])
            if row.get("approved_output")
        ]

        if len(approved_texts) < 5:
            logger.info("Not enough approved outputs for full rebuild (%d) — skipping", len(approved_texts))
            return

        logger.info("Rebuilding voice profile for %s from %d approved outputs", user_id, len(approved_texts))
        new_profile = await build_voice_profile(approved_texts, user_id)
        await save_voice_profile(user_id, new_profile)
        logger.info("Voice profile rebuilt for %s", user_id)

    except Exception as exc:
        logger.warning("Full voice profile rebuild failed for %s: %s", user_id, exc)