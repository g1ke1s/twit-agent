"""
api/routers/generate.py — /api/generate SSE endpoint with full node tracking.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import get_graph
from agent.schemas import AgentState, InputContext, InputType, OutputType
from memory.store import update_run_status

logger = logging.getLogger(__name__)
router = APIRouter()

# All pipeline node names — used for node_start/node_complete detection
PIPELINE_NODES = {
    "input_router", "voice_loader", "researcher", "analyst",
    "hook_generator", "writer", "critic", "validator_gate",
    "formatter", "distributor",
}


class GenerateRequest(BaseModel):
    raw_input: str
    input_type: str = "instruction"
    output_type: str = "x_thread"
    user_instruction: str | None = None
    uploaded_content: str | None = None
    target_tone: str | None = None
    user_id: str = "default"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_graph_stream(
    run_id: str,
    initial_state: AgentState,
) -> AsyncGenerator[str, None]:
    graph  = get_graph()
    config = {"configurable": {"thread_id": run_id}}

    yield _sse("start", {"run_id": run_id, "status": "starting"})
    logger.info("[run:%s] Pipeline started", run_id[:8])

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            # Node starting
            if kind == "on_chain_start" and name in PIPELINE_NODES:
                logger.debug("[run:%s] Node start: %s", run_id[:8], name)
                yield _sse("node_start", {"node": name})

            # Node completed
            elif kind == "on_chain_end" and name in PIPELINE_NODES:
                data    = event.get("data", {})
                output  = data.get("output") or {}
                status  = output.get("status", "")
                trace   = output.get("trace") or []
                # detail  = trace[0].get("detail", "") if trace else ""
                # duration = trace[0].get("duration_ms", 0) if trace else 0

                first    = trace[0] if trace else None
                detail   = (first.detail      if hasattr(first, "detail")      else first.get("detail", ""))      if first else ""
                duration = (first.duration_ms  if hasattr(first, "duration_ms") else first.get("duration_ms", 0)) if first else 0
                logger.debug("[run:%s] Node done: %s status=%s detail=%s", run_id[:8], name, status, detail)
                yield _sse("node_complete", {
                    "node": name, "status": status,
                    "detail": detail, "duration_ms": duration,
                })

            # Graph end
            elif kind == "on_chain_end" and name == "__end__":
                logger.debug("[run:%s] Graph __end__ reached", run_id[:8])
                break

        # ── Check interrupt state ────────────────────────────────────────
        snapshot   = graph.get_state(config)
        next_nodes = list(snapshot.next) if snapshot and snapshot.next else []
        logger.info("[run:%s] Graph paused, next=%s", run_id[:8], next_nodes)

        if "validator_gate" in next_nodes:
            state_vals = snapshot.values
            draft      = state_vals.get("draft", [])
            critique   = state_vals.get("critique")
            sources    = state_vals.get("sources", [])
            claims     = state_vals.get("claims", [])

            draft_payload = (
                [t.dict() for t in draft] if isinstance(draft, list) else draft
            )
            critique_payload = critique.dict() if critique else {}
            sources_payload  = [
                {"url": s.url, "title": s.title, "credibility_score": s.credibility_score}
                for s in sources
            ]
            claims_payload = [
                {"id": c.id, "text": c.text, "verified": c.verified,
                 "surprise_score": c.surprise_score, "source_url": c.source_url}
                for c in claims
            ]

            await update_run_status(run_id, "awaiting_validation", draft)

            yield _sse("node_start", {"node": "validator_gate"})
            yield _sse("awaiting_validation", {
                "run_id":       run_id,
                "draft":        draft_payload,
                "critique":     critique_payload,
                "sources":      sources_payload,
                "claims":       claims_payload,
                "validate_url": f"/validate/{run_id}",
            })
            logger.info("[run:%s] Awaiting human validation", run_id[:8])
        else:
            # Completed without interrupt (e.g. after resume)
            final_state  = graph.get_state(config)
            final_output = final_state.values.get("final_output", "") if final_state else ""
            await update_run_status(run_id, "complete")
            yield _sse("complete", {"run_id": run_id, "final_output": final_output})
            logger.info("[run:%s] Pipeline complete", run_id[:8])

    except Exception as exc:
        logger.exception("[run:%s] Graph run failed: %s", run_id[:8], exc)
        yield _sse("error", {"run_id": run_id, "error": str(exc)})


@router.post("/generate")
async def generate_content(req: GenerateRequest):
    run_id = str(uuid.uuid4())

    try:
        input_type  = InputType(req.input_type)
        output_type = OutputType(req.output_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    input_ctx = InputContext(
        raw_input=req.raw_input,
        input_type=input_type,
        output_type=output_type,
        user_instruction=req.user_instruction,
        uploaded_content=req.uploaded_content,
        target_tone=req.target_tone,
    )

    initial_state: AgentState = {
        "run_id":           run_id,
        "user_id":          req.user_id,
        "input_context":    input_ctx,
        "voice_profile":    None,
        "sources":          [],
        "claims":           [],
        "hook_variants":    [],
        "thesis_options":   [],
        "chosen_thesis":    "",
        "draft":            [],
        "critique":         None,
        "validation":       None,
        "formatted_variants": None,
        "final_output":     "",
        "iteration_count":  0,
        "writer_instruction": "",
        "trace":            [],
        "status":           "starting",
        "error":            None,
    }

    logger.info("[run:%s] New run user=%s output=%s", run_id[:8], req.user_id, req.output_type)

    return StreamingResponse(
        _run_graph_stream(run_id, initial_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "X-Run-Id":          run_id,
        },
    )
