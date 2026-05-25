"""
api/routers/memory.py — Voice archive upload, file parsing, share links.
"""
from __future__ import annotations

import io
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_txt(content: bytes) -> list[str]:
    text = content.decode("utf-8", errors="replace")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _extract_pdf(content: bytes) -> list[str]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        full = "\n\n".join(pages)
        return [p.strip() for p in full.split("\n\n") if len(p.strip()) > 40]
    except ImportError:
        logger.warning("PyMuPDF not installed — falling back to raw decode")
        return _extract_txt(content)
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        return []


def _extract_docx(content: bytes) -> list[str]:
    try:
        from docx import Document  # python-docx
        doc = Document(io.BytesIO(content))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return paragraphs
    except ImportError:
        logger.warning("python-docx not installed — treating as plain text")
        return _extract_txt(content)
    except Exception as exc:
        logger.error("DOCX extraction failed: %s", exc)
        return []


def _extract_json(content: bytes) -> list[str]:
    import json
    try:
        data = json.loads(content.decode("utf-8", errors="replace"))
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict):
            # Try common keys: posts, tweets, texts, content
            for key in ("posts", "tweets", "texts", "content", "samples"):
                if key in data and isinstance(data[key], list):
                    return [str(item).strip() for item in data[key] if str(item).strip()]
        return [json.dumps(data)]
    except Exception:
        return _extract_txt(content)


def parse_file(filename: str, content: bytes) -> list[str]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf(content)
    elif name.endswith(".docx"):
        return _extract_docx(content)
    elif name.endswith(".json"):
        return _extract_json(content)
    else:  # .txt, .csv, or unknown
        return _extract_txt(content)


# ---------------------------------------------------------------------------
# Voice archive upload
# ---------------------------------------------------------------------------

@router.post("/memory/voice-archive")
async def upload_voice_archive(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Accept PDF, DOCX, TXT, CSV, or JSON writing archive.
    Extracts text, builds VoiceProfile, persists to Supabase.
    """
    raw = await file.read()
    archive_texts = parse_file(file.filename or "", raw)

    if len(archive_texts) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Archive must contain at least 2 text samples. Got {len(archive_texts)}.",
        )

    try:
        from agent.voice_profile import build_voice_profile
        from memory.store import save_voice_profile
        profile = await build_voice_profile(archive_texts, user_id)
        await save_voice_profile(user_id, profile)
        return {
            "user_id": user_id,
            "samples_processed": len(archive_texts),
            "avg_sentence_length": round(profile.avg_sentence_length, 1),
            "vocabulary_richness": round(profile.vocabulary_richness, 3),
            "style_summary": profile.style_summary,
            "preferred_structures": profile.preferred_structures,
        }
    except Exception as exc:
        logger.error("Voice profile build failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/memory/voice-profile/{user_id}")
async def get_profile(user_id: str):
    try:
        from memory.store import get_voice_profile
        profile = await get_voice_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="No voice profile found")
        return profile.dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# General file upload for content generation (PDF/DOCX → text)
# ---------------------------------------------------------------------------

@router.post("/upload/content-file")
async def upload_content_file(file: UploadFile = File(...)):
    """
    Parse an uploaded file and return its extracted text.
    Frontend calls this before calling /api/generate with input_type=file_upload.
    Supports: PDF, DOCX, TXT, CSV, JSON.
    Max size: 10MB.
    """
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    raw = await file.read()
    if len(raw) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    texts = parse_file(file.filename or "", raw)
    if not texts:
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    combined = "\n\n".join(texts)
    return {
        "filename": file.filename,
        "text": combined[:20000],  # cap at 20k chars for context
        "paragraphs": len(texts),
        "chars": len(combined),
    }


# ---------------------------------------------------------------------------
# Share links — create a public read-only view of a completed run
# ---------------------------------------------------------------------------

@router.post("/share/{run_id}")
async def create_share_link(run_id: str):
    """
    Create a shareable, read-only link for an approved run.
    Returns: { share_id, share_url }
    """
    import uuid
    from memory.store import get_run
    from datetime import datetime

    run = await get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Run not yet complete")

    share_id = str(uuid.uuid4()).replace("-", "")[:16]

    try:
        from memory.store import _get_client
        client = _get_client()
        client.table("shared_runs").insert({
            "share_id": share_id,
            "run_id": run_id,
            "user_id": run.get("user_id", ""),
            "final_output": run.get("final_output", ""),
            "formatted_variants": run.get("formatted_variants"),
            "sources": run.get("trace"),  # reuse trace field for simplicity
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as exc:
        logger.warning("Could not persist share link: %s", exc)
        # Still return a link — it will just show final_output from the run

    return {
        "share_id": share_id,
        "run_id": run_id,
        "share_url": f"/share/{share_id}",
    }


@router.get("/share/{share_id}")
async def get_shared_run(share_id: str):
    """
    Public read-only endpoint. Returns final output + variants for a shared run.
    No auth required.
    """
    try:
        from memory.store import _get_client
        client = _get_client()
        resp = (
            client.table("shared_runs")
            .select("*")
            .eq("share_id", share_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            row = resp.data[0]
            return {
                "share_id": share_id,
                "final_output": row.get("final_output", ""),
                "formatted_variants": row.get("formatted_variants"),
                "created_at": row.get("created_at"),
            }
    except Exception as exc:
        logger.warning("Share lookup failed: %s", exc)

    # Fallback: try direct run lookup
    from memory.store import get_run
    run = await get_run(share_id)
    if run:
        return {
            "share_id": share_id,
            "final_output": run.get("final_output", ""),
            "formatted_variants": run.get("formatted_variants"),
        }

    raise HTTPException(status_code=404, detail="Shared run not found")
