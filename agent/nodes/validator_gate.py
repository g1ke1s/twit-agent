"""
agent/nodes/validator_gate.py — Layer 7: Human-in-the-loop interrupt gate.
"""
from __future__ import annotations

import logging
import time

from agent.schemas import AgentState, TraceEvent, ValidationDecision

logger = logging.getLogger(__name__)


async def validator_gate_node(state: AgentState) -> dict:
    """
    This node is reached AFTER the graph resumes from interrupt_before.
    By the time this runs, state["validation"] has been populated by the
    /api/validate/{run_id} webhook. We simply pass through and return.
    """
    t0 = time.monotonic()
    validation: ValidationDecision | None = state.get("validation")

    if not validation:
        # Should not happen in normal flow — graph interrupted before this node
        logger.warning("validator_gate reached with no validation decision in state")
        return {"status": "awaiting_validation"}

    decision = validation.decision

    # If reject: inject rejection note into writer instruction
    updates: dict = {"status": "validated", "validation": validation}
    if decision == "reject" and validation.rejection_note:
        existing = state.get("writer_instruction", "")
        updates["writer_instruction"] = (
            existing + f"\n\nREJECTION NOTE: {validation.rejection_note}"
        ).strip()
        updates["iteration_count"] = 0

    # If edit: replace draft with human-edited version
    if decision == "edit" and validation.human_edits:
        existing_draft = state.get("draft")
        if isinstance(existing_draft, list):
            # Parse human_edits as newline-separated tweets
            lines = [l.strip() for l in validation.human_edits.split("\n") if l.strip()]
            from agent.schemas import Tweet
            new_tweets = []
            for i, line in enumerate(lines):
                # Strip leading "N/" numbering if present
                text = line.lstrip("0123456789").lstrip("/").strip()
                new_tweets.append(Tweet(
                    position=i + 1,
                    text=text,
                    char_count=len(text),
                ))
            updates["draft"] = new_tweets
        else:
            updates["draft"] = validation.human_edits

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="validator_gate",
        model="human",
        duration_ms=duration_ms,
        detail=f"decision={decision}",
    )
    updates["trace"] = [event]
    return updates


def route_after_validation(state: AgentState) -> str:
    validation: ValidationDecision | None = state.get("validation")
    if not validation:
        return "approve"
    if validation.decision == "reject":
        return "reject"
    return "approve"  # covers both "approve" and "edit"
