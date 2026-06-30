"""
Pydantic models for request validation and response schema enforcement.

The response schema is the EXACT contract from the assignment — never deviate.
"""
from __future__ import annotations

from pydantic import BaseModel, field_validator
from typing import Literal


# ── Request Models ────────────────────────────────────────────────────────────

class Message(BaseModel):
    """A single message in the conversation history."""
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content must be a non-empty string")
        return v


class ChatRequest(BaseModel):
    """
    POST /chat request body.
    `messages` can be empty (= fresh conversation, no context yet).
    """
    messages: list[Message] = []


# ── Response Models ───────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """A single catalog item recommendation."""
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """
    POST /chat response body — EXACT shape, every field, every call.

    - `reply`: the agent's natural-language reply
    - `recommendations`: always a list (never null), 0–10 items
    - `end_of_conversation`: boolean, true only when final shortlist is delivered
    """
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Internal Models (not exposed on the wire) ─────────────────────────────────

class ExtractedConstraints(BaseModel):
    """Structured constraints extracted by the LLM from conversation history."""
    role: str = ""
    seniority: str = ""
    skills: list[str] = []
    test_types: list[str] = []
    languages: list[str] = []
    industry: str = ""
    specific_assessments: list[str] = []
    query_text: str = ""  # free-text search query derived from constraints
    include_personality: bool = True  # default: include OPQ32r per trace pattern


class LLMExtractionResult(BaseModel):
    """The structured output we ask the LLM to produce (intermediate, not wire format)."""
    intent: Literal["clarify", "recommend", "refine", "compare", "refuse"]
    constraints: ExtractedConstraints = ExtractedConstraints()
    draft_reply: str = ""
    clarifying_question: str = ""
    additions: list[str] = []       # for refine: items/skills to add
    removals: list[str] = []        # for refine: items/skills to remove
    compare_items: list[str] = []   # for compare: specific assessment names
    previous_shortlist_names: list[str] = []  # names already recommended
