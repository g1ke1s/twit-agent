"""
agent/nodes/input_router.py — Layer 0: Normalise any input into InputContext.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from agent.schemas import AgentState, InputContext, InputType, OutputType, TraceEvent
from agent.tools import jina_fetch, scrape_tweet

logger = logging.getLogger(__name__)

_TWITTER_RE = re.compile(r"https?://(www\.)?(twitter\.com|x\.com)/\w+/status/\d+")
_URL_RE = re.compile(r"https?://\S+")

OUTPUT_KEYWORDS: dict[OutputType, list[str]] = {
    OutputType.x_thread:             ["thread", "tweets", "twitter thread"],
    OutputType.quote_rt:             ["quote", "quote tweet", "quote rt", "quote retweet"],
    OutputType.essay:                ["essay", "article", "blog post", "long-form"],
    OutputType.analysis:             ["analysis", "analyze", "analyse", "report"],
    OutputType.strategic_narrative:  ["strategic narrative", "narrative", "strategy"],
}


def _detect_output_type(instruction: str) -> OutputType:
    lower = instruction.lower()
    for output_type, keywords in OUTPUT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return output_type
    return OutputType.x_thread


async def input_router_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    raw = state.get("input_context", {})
    if isinstance(raw, dict):
        raw_input = raw.get("raw_input", "")
        user_instruction = raw.get("user_instruction", "")
        uploaded_content = raw.get("uploaded_content")
    else:
        # InputContext already constructed upstream (API entry point)
        return {"status": "input_parsed"}

    raw_input = raw_input.strip()
    input_type: InputType
    content: str = raw_input
    original_tweet: str | None = None

    # --- Detect input type ---
    if _TWITTER_RE.match(raw_input):
        input_type = InputType.tweet_url
        content = await scrape_tweet(raw_input)
        original_tweet = content
        logger.info("Scraped tweet from %s", raw_input)

    elif _URL_RE.match(raw_input) and not raw_input.startswith("http") is False:
        # General URL
        input_type = InputType.web_url
        content = await jina_fetch(raw_input)
        logger.info("Fetched web URL %s via Jina", raw_input)

    elif uploaded_content:
        input_type = InputType.file_upload
        content = uploaded_content

    elif len(raw_input.split()) <= 40 and "\n" not in raw_input:
        # Short and no URL → treat as raw tweet or instruction
        input_type = InputType.raw_tweet if "#" in raw_input or "@" in raw_input else InputType.instruction

    else:
        input_type = InputType.instruction

    # --- Detect intended output type ---
    combined_instruction = f"{user_instruction} {raw_input}".lower()
    output_type = _detect_output_type(combined_instruction)

    input_ctx = InputContext(
        raw_input=raw_input,
        input_type=input_type,
        output_type=output_type,
        original_tweet=original_tweet,
        uploaded_content=uploaded_content,
        user_instruction=user_instruction or None,
    )

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="input_router",
        model="none",
        duration_ms=duration_ms,
        detail=f"Detected input_type={input_type.value}, output_type={output_type.value}",
    )

    return {
        "input_context": input_ctx,
        "status": "input_parsed",
        "trace": [event],
    }
