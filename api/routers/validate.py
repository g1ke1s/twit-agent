"""
api/routers/validate.py — Human validator webhook: poll + submit decision + resume graph.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.graph import get_graph
from agent.schemas import ValidationDecision
from memory.store import get_run, update_run_status, save_feedback

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Poll endpoint
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}")
async def get_run_status(run_id: str):
    """
    Poll endpoint for the validator UI.
    Returns current run state, draft, critique, sources, and trace.
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        logger.warning("get_state failed for %s: %s", run_id[:8], exc)
        snapshot = None

    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    state      = snapshot.values or {}
    next_nodes = list(snapshot.next) if snapshot.next else []

    draft   = state.get("draft", [])
    critique = state.get("critique")
    sources  = state.get("sources", [])
    trace    = state.get("trace", [])
    claims   = state.get("claims", [])

    return {
        "run_id":              run_id,
        "status":              state.get("status", "unknown"),
        "next":                next_nodes,
        "awaiting_validation": "validator_gate" in next_nodes,
        "draft":               [t.dict() for t in draft] if isinstance(draft, list) else draft,
        "critique":            critique.dict() if critique else None,
        "sources":             [
            {"url": s.url, "title": s.title, "credibility_score": s.credibility_score}
            for s in sources
        ],
        "claims": [
            {"id": c.id, "text": c.text, "verified": c.verified,
             "surprise_score": c.surprise_score}
            for c in claims
        ],
        "trace":               [e.dict() for e in trace],
    }


# ---------------------------------------------------------------------------
# Submit validation decision
# ---------------------------------------------------------------------------

class ValidationRequest(BaseModel):
    decision:       str            # "approve" | "edit" | "reject"
    human_edits:    Optional[str] = None
    rejection_note: Optional[str] = None


@router.post("/validate/{run_id}")
async def submit_validation(run_id: str, req: ValidationRequest):
    """
    Receive human decision, update graph state, resume the pipeline.
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": run_id}}

    # Verify state exists and is paused at validator_gate
    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {exc}")

    if snapshot is None:
        raise HTTPException(status_code=404, detail="Run not found")

    next_nodes = list(snapshot.next) if snapshot.next else []
    if "validator_gate" not in next_nodes:
        raise HTTPException(
            status_code=409,
            detail=f"Run is not awaiting validation (next={next_nodes})"
        )

    # Build and validate decision
    try:
        decision = ValidationDecision(
            run_id=run_id,
            decision=req.decision,       # type: ignore
            human_edits=req.human_edits,
            rejection_note=req.rejection_note,
            timestamp=datetime.utcnow(),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info("[run:%s] Validation submitted: decision=%s", run_id[:8], req.decision)

    # Save human edit diff to feedback table
    if req.decision in ("approve", "edit"):
        state_vals    = snapshot.values or {}
        original_draft = state_vals.get("draft", [])
        original_text  = (
            " ".join(t.text for t in original_draft)
            if isinstance(original_draft, list)
            else str(original_draft)
        )
        approved_text = req.human_edits or original_text
        try:
            await save_feedback(
                run_id=run_id,
                user_id=state_vals.get("user_id", "default"),
                original_draft=original_text,
                approved_output=approved_text,
                decision=req.decision,
            )
        except Exception as exc:
            logger.warning("save_feedback failed: %s", exc)

        # Trigger voice profile update in background
        profile = state_vals.get("voice_profile")
        user_id = state_vals.get("user_id", "default")
        if profile and approved_text:
            asyncio.create_task(
                _update_voice_async(original_text, approved_text, profile, user_id)
            )

    # Inject state updates before resume
    state_updates: dict = {"validation": decision}

    if req.decision == "reject" and req.rejection_note:
        existing = (snapshot.values or {}).get("writer_instruction", "")
        state_updates["writer_instruction"] = (
            existing + f"\n\nREJECTION NOTE: {req.rejection_note}"
        ).strip()
        state_updates["iteration_count"] = 0
        logger.info("[run:%s] Rejection note injected into writer instruction", run_id[:8])

    if req.decision == "edit" and req.human_edits:
        existing_draft = (snapshot.values or {}).get("draft", [])
        if isinstance(existing_draft, list):
            from agent.schemas import Tweet
            lines = [l.strip() for l in req.human_edits.split("\n") if l.strip()]
            new_tweets = []
            for i, line in enumerate(lines):
                text = line.lstrip("0123456789").lstrip("/").strip()
                if text:
                    new_tweets.append(Tweet(
                        position=i + 1, text=text, char_count=len(text)
                    ))
            if new_tweets:
                state_updates["draft"] = new_tweets
        else:
            state_updates["draft"] = req.human_edits

    # Apply state updates
    graph.update_state(config, state_updates)

    # Resume graph asynchronously
    asyncio.create_task(_resume_graph(run_id, config))

    return {
        "run_id":   run_id,
        "decision": req.decision,
        "status":   "resuming",
        "poll_url": f"/api/runs/{run_id}",
    }


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------

async def _resume_graph(run_id: str, config: dict) -> None:
    """Resume the paused graph and persist final state."""
    graph = get_graph()
    try:
        logger.info("[run:%s] Resuming graph", run_id[:8])
        async for _ in graph.astream(None, config=config):
            pass  # consume all remaining events
        await update_run_status(run_id, "complete")
        logger.info("[run:%s] Graph resume complete", run_id[:8])
    except Exception as exc:
        logger.error("[run:%s] Graph resume failed: %s", run_id[:8], exc)
        await update_run_status(run_id, "error")


async def _update_voice_async(
    original_draft: str,
    approved_output: str,
    profile,
    user_id: str,
) -> None:
    try:
        from agent.voice_profile import update_voice_profile_from_feedback
        await update_voice_profile_from_feedback(
            original_draft, approved_output, profile, user_id
        )
    except Exception as exc:
        logger.warning("Background voice update failed: %s", exc)
