"""
agent/nodes/distributor.py — Layer 9: Persist run to Supabase and emit final SSE event.
"""
from __future__ import annotations

import logging
import time

from agent.schemas import AgentState, TraceEvent

logger = logging.getLogger(__name__)


async def distributor_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    run_id = state.get("run_id", "unknown")

    # Persist to Supabase
    try:
        from memory.store import save_run
        await save_run(
            run_id=run_id,
            user_id=state.get("user_id", "default"),
            final_output=state.get("final_output", ""),
            formatted_variants=state.get("formatted_variants"),
            trace=state.get("trace", []),
            validation=state.get("validation"),
        )
    except Exception as exc:
        logger.warning("Failed to persist run %s to Supabase: %s", run_id, exc)

    # Save trace to traces/ folder on disk
    try:
        import json
        from pathlib import Path
        from datetime import datetime

        traces_dir = Path("traces")
        traces_dir.mkdir(exist_ok=True)

        trace_data = {
            "run_id":       run_id,
            "user_id":      state.get("user_id", "default"),
            "timestamp":    datetime.utcnow().isoformat(),
            "input":        state.get("input_context", {}).raw_input if state.get("input_context") else "",
            "output_type":  state.get("input_context", {}).output_type.value if state.get("input_context") else "",
            "final_output": state.get("final_output", ""),
            "sources":      [{"url": s.url, "title": s.title, "credibility_score": s.credibility_score}
                             for s in state.get("sources", [])],
            "claims":       [{"id": c.id, "text": c.text, "verified": c.verified,
                              "surprise_score": c.surprise_score}
                             for c in state.get("claims", [])],
            "critique":     state.get("critique").model_dump(mode="json") if state.get("critique") else None,
            "validation":   state.get("validation").model_dump(mode="json") if state.get("validation") else None,
            "trace":        [e.model_dump(mode="json") for e in state.get("trace", [])],
        }

        trace_path = traces_dir / f"trace_{run_id[:8]}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, default=str)
        logger.info("Trace saved to %s", trace_path)
    except Exception as exc:
        logger.warning("Failed to save trace to disk: %s", exc)

    # Update voice profile from feedback if human made edits
    validation = state.get("validation")
    if validation and validation.decision == "edit":
        try:
            from agent.voice_profile import update_voice_profile_from_feedback
            profile = state.get("voice_profile")
            if profile:
                draft = state.get("draft", "")
                original_text = (
                    " ".join(t.text for t in draft) if isinstance(draft, list) else str(draft)
                )
                await update_voice_profile_from_feedback(
                    original_draft=original_text,
                    approved_output=validation.human_edits or "",
                    profile=profile,
                    user_id=state.get("user_id", "default"),
                )
        except Exception as exc:
            logger.warning("Voice profile update failed: %s", exc)

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="distributor",
        model="none",
        duration_ms=duration_ms,
        detail=f"run_id={run_id} persisted",
    )
    return {"status": "complete", "trace": [event]}