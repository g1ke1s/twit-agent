"""
agent/schemas.py — All Pydantic v2 models for the Voice Agent pipeline.
"""
from __future__ import annotations

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class InputType(str, Enum):
    tweet_url = "tweet_url"
    raw_tweet = "raw_tweet"
    web_url = "web_url"
    file_upload = "file_upload"
    instruction = "instruction"
    voice_archive = "voice_archive"


class OutputType(str, Enum):
    x_thread = "x_thread"
    quote_rt = "quote_rt"
    essay = "essay"
    analysis = "analysis"
    strategic_narrative = "strategic_narrative"


# ---------------------------------------------------------------------------
# Input / context
# ---------------------------------------------------------------------------

class InputContext(BaseModel):
    raw_input: str
    input_type: InputType
    output_type: OutputType
    original_tweet: Optional[str] = None
    uploaded_content: Optional[str] = None
    user_instruction: Optional[str] = None
    target_tone: Optional[str] = None  # "analytical" | "provocative" | "educational"

    @field_validator("raw_input")
    @classmethod
    def raw_input_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw_input must not be empty")
        return v


# ---------------------------------------------------------------------------
# Voice profile
# ---------------------------------------------------------------------------

class VoiceProfile(BaseModel):
    avg_sentence_length: float
    vocabulary_richness: float
    preferred_structures: list[str]
    forbidden_phrases: list[str]
    style_summary: str
    few_shot_examples: list[str]
    faiss_index_path: str

    @field_validator("few_shot_examples")
    @classmethod
    def at_least_one_example(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("few_shot_examples must have at least one entry")
        return v

    @property
    def few_shot_examples_formatted(self) -> str:
        return "\n---\n".join(f"Example {i + 1}:\n{ex}" for i, ex in enumerate(self.few_shot_examples))

    @property
    def forbidden_phrases_formatted(self) -> str:
        return ", ".join(self.forbidden_phrases)


# ---------------------------------------------------------------------------
# Research artefacts
# ---------------------------------------------------------------------------

class Source(BaseModel):
    url: str
    title: str
    content: str  # Jina-cleaned markdown, max 4000 chars
    source_type: str  # web | arxiv | hn | exa | file
    credibility_score: float = Field(ge=0, le=10)
    claim_ids: list[str] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def truncate_content(cls, v: str) -> str:
        return v[:4000]


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    source_url: str
    verified: bool = False
    surprise_score: float = Field(ge=1, le=10)
    contradiction_flag: bool = False

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Claim text must not be empty")
        return v


# ---------------------------------------------------------------------------
# Hook variants
# ---------------------------------------------------------------------------

class HookVariant(BaseModel):
    text: str
    pattern: Literal["counterintuitive", "specific_number", "personal_story", "bold_prediction"]
    score: float = Field(ge=1, le=10)
    reasoning: str

    @property
    def formatted(self) -> str:
        return f"[{self.pattern}] (score: {self.score:.1f})\n{self.text}\nReasoning: {self.reasoning}"


# ---------------------------------------------------------------------------
# Draft content
# ---------------------------------------------------------------------------

class Tweet(BaseModel):
    position: int
    text: str
    char_count: int
    claim_ids: list[str] = Field(default_factory=list)
    is_hook: bool = False
    is_cta: bool = False

    @field_validator("char_count")
    @classmethod
    def char_count_within_limit(cls, v: int) -> int:
        if v > 280:
            raise ValueError(f"Tweet exceeds 280 chars ({v})")
        return v

    @field_validator("text")
    @classmethod
    def text_matches_char_count(cls, v: str) -> str:
        return v


# ---------------------------------------------------------------------------
# Critique
# ---------------------------------------------------------------------------

class CritiqueResult(BaseModel):
    hook_strength: float
    voice_match: float
    insight_density: float
    clarity: float
    hallucination_flags: list[str] = Field(default_factory=list)
    cliche_flags: list[str] = Field(default_factory=list)
    average_score: float
    rewrite_instructions: str
    verdict: Literal["approve", "rewrite"]

    @field_validator("average_score")
    @classmethod
    def compute_average(cls, v: float) -> float:
        return round(v, 2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationDecision(BaseModel):
    run_id: str
    decision: Literal["approve", "edit", "reject"]
    human_edits: Optional[str] = None
    rejection_note: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("human_edits")
    @classmethod
    def edits_required_when_edit(cls, v: Optional[str], info) -> Optional[str]:
        # Pydantic v2: info.data contains already-validated fields
        if info.data.get("decision") == "edit" and not v:
            raise ValueError("human_edits is required when decision is 'edit'")
        return v


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    node: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0
    detail: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Formatted output variants
# ---------------------------------------------------------------------------

class FormattedVariants(BaseModel):
    platform_native: str          # thread JSON / markdown essay
    linkedin_variant: str
    newsletter_paragraph: str
    word_count: Optional[int] = None
    sources_section: Optional[str] = None


# ---------------------------------------------------------------------------
# Master state (TypedDict for LangGraph)
# ---------------------------------------------------------------------------

from typing import TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    user_id: str
    input_context: InputContext
    voice_profile: Optional[VoiceProfile]
    sources: list[Source]
    claims: list[Claim]
    hook_variants: list[HookVariant]
    thesis_options: list[str]
    chosen_thesis: str
    draft: list[Tweet] | str
    critique: Optional[CritiqueResult]
    validation: Optional[ValidationDecision]
    formatted_variants: Optional[FormattedVariants]
    final_output: str
    iteration_count: int
    writer_instruction: str
    trace: Annotated[list[TraceEvent], operator.add]
    status: str
    error: Optional[str]
