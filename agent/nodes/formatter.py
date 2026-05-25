"""
agent/nodes/formatter.py — Layer 8: Platform-native output formatting.
"""
from __future__ import annotations

import json
import logging
import time

from agent.schemas import AgentState, FormattedVariants, OutputType, Source, TraceEvent, Tweet

logger = logging.getLogger(__name__)


def _format_thread(tweets: list[Tweet]) -> str:
    lines = []
    for t in tweets:
        badge = ""
        if t.is_hook:
            badge = " [thread]"
        prefix = f"{t.position}/"
        lines.append(f"{prefix}{badge} {t.text}  ({t.char_count} chars)")
    return "\n\n".join(lines)


def _thread_copy_json(tweets: list[Tweet]) -> str:
    return json.dumps(
        [{"position": t.position, "text": t.text, "char_count": t.char_count} for t in tweets],
        indent=2,
    )


def _sources_section(sources: list[Source]) -> str:
    if not sources:
        return ""
    lines = ["## Sources"]
    for s in sources:
        lines.append(f"- [{s.title}]({s.url}) — credibility {s.credibility_score}/10")
    return "\n".join(lines)


def _word_count(text: str) -> int:
    return len(text.split())


async def _generate_linkedin_variant(content: str, output_type: OutputType) -> str:
    """Rewrite for LinkedIn using Mistral Small."""
    import os
    import httpx
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        return content

    prompt = (
        f"Rewrite this {'thread' if output_type in (OutputType.x_thread, OutputType.quote_rt) else 'essay'} "
        f"for LinkedIn. Keep the same ideas but make it more narrative and professional. "
        f"Remove Twitter-specific formatting. Max 1500 words.\n\n{content}"
    )
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("LinkedIn variant generation failed: %s", exc)
        return content[:500] + "..."


def _newsletter_paragraph(content: str) -> str:
    """Extract or generate a 3-sentence newsletter hook."""
    sentences = content.replace("\n", " ").split(". ")
    hook_sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:3]
    return ". ".join(hook_sentences) + ("." if hook_sentences else "")


async def formatter_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    draft = state.get("draft")
    validation = state.get("validation")
    output_type: OutputType = state["input_context"].output_type
    sources: list[Source] = state.get("sources", [])

    # Use human-edited draft if provided
    if validation and validation.decision == "edit" and validation.human_edits:
        if isinstance(draft, list):
            pass  # already replaced in validator_gate
        else:
            draft = validation.human_edits

    # --- Format platform-native output ---
    if isinstance(draft, list):
        # Thread / quote RT
        # Verify char counts
        for t in draft:
            if t.char_count > 280:
                t.text = t.text[:277] + "..."
                t.char_count = len(t.text)

        platform_native = _thread_copy_json(draft)
        full_text = _format_thread(draft)
        wc = None
        src_section = None
    else:
        # Essay / analysis / strategic narrative
        text = draft or ""

        # Add sources section
        src_section = _sources_section(sources)
        if output_type == OutputType.strategic_narrative:
            # Add key takeaways
            claims = state.get("claims", [])
            takeaways = "\n".join(f"- {c.text}" for c in claims[:5])
            text = text + f"\n\n## Key Takeaways\n{takeaways}"

        if src_section:
            text = text + f"\n\n{src_section}"

        platform_native = text
        full_text = text
        wc = _word_count(text)

    # Generate LinkedIn variant
    linkedin = await _generate_linkedin_variant(full_text[:3000], output_type)
    newsletter = _newsletter_paragraph(full_text)

    variants = FormattedVariants(
        platform_native=platform_native,
        linkedin_variant=linkedin,
        newsletter_paragraph=newsletter,
        word_count=wc,
        sources_section=src_section,
    )

    duration_ms = (time.monotonic() - t0) * 1000
    event = TraceEvent(
        node="formatter",
        model="mistral-small",
        duration_ms=duration_ms,
        detail=f"output_type={output_type.value} wc={wc}",
    )
    return {
        "formatted_variants": variants,
        "final_output": platform_native,
        "status": "formatted",
        "trace": [event],
    }
