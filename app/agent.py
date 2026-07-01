"""
Core agent logic — stateless conversation handler.

Every call reconstructs all context from the full message history.
Routes to clarify/recommend/refine/compare/refuse based on LLM-extracted intent.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from app.models import (
    ChatRequest, ChatResponse, Recommendation,
    ExtractedConstraints, LLMExtractionResult,
)
from app.catalog import (
    get_catalog, lookup_by_name, validate_url, item_to_recommendation,
    find_items_by_names,
)
from app.retrieval import search, lookup_items_for_compare, search_for_refinement
from app.llm_client import extract_intent, generate_reply, TOTAL_BUDGET_SECONDS
from app.prompts import (
    INTENT_EXTRACTION_SYSTEM_PROMPT,
    COMPARE_SYSTEM_PROMPT,
    REPLY_FORMATTING_PROMPT,
)

logger = logging.getLogger(__name__)

# ── Fallback responses (schema-valid, used when LLM fails) ───────────────────

FALLBACK_CLARIFY = ChatResponse(
    reply=(
        "I can help narrow that down. "
        "Could you tell me more about the role and what specific skills "
        "or competencies you need to evaluate?"
    ),
    recommendations=[],
    end_of_conversation=False,
)

FALLBACK_REFUSE = ChatResponse(
    reply=(
        "I specialize in selecting SHL assessments. "
        "I cannot assist with that request, but I'm ready to help you build "
        "an assessment shortlist. What role are you hiring for?"
    ),
    recommendations=[],
    end_of_conversation=False,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_previous_shortlist(messages: list[dict]) -> list[dict]:
    """
    Scan previous assistant messages for any recommendations that were given.
    Returns a list of recommendation dicts from the most recent recommending turn.
    """
    # Walk backwards through assistant messages looking for recommendation patterns
    # Since we're stateless, we need to parse this from the conversation text
    # The LLM extraction also provides previous_shortlist_names, but we can
    # double-check by scanning the conversation ourselves
    return []  # Will rely on LLM extraction for this


def _build_query_from_constraints(constraints: dict) -> str:
    """Build a search query string from extracted constraints."""
    parts = []
    if constraints.get("role"):
        parts.append(constraints["role"])
    if constraints.get("seniority"):
        parts.append(constraints["seniority"])
    if constraints.get("skills"):
        parts.extend(constraints["skills"])
    if constraints.get("industry"):
        parts.append(constraints["industry"])
    if constraints.get("specific_assessments"):
        parts.extend(constraints["specific_assessments"])
    if constraints.get("query_text"):
        parts.append(constraints["query_text"])
    return " ".join(parts)


def _validate_recommendations(recs: list[dict]) -> list[Recommendation]:
    """
    Validate every recommendation URL against the catalog.
    Drop any that aren't real catalog items. Never let hallucinated URLs through.
    """
    validated = []
    for rec in recs:
        if validate_url(rec.get("url", "")):
            validated.append(Recommendation(
                name=rec["name"],
                url=rec["url"],
                test_type=rec["test_type"],
                duration=rec.get("duration", "") or "",
                keys=rec.get("keys", "") or "",
                languages=rec.get("languages", "") or "",
            ))
        else:
            logger.warning(f"Dropped recommendation with invalid URL: {rec.get('name', '?')}")
    return validated


def _parse_extraction(raw: dict) -> LLMExtractionResult:
    """Parse the raw LLM JSON output into our structured model, with defaults."""
    try:
        constraints_raw = raw.get("constraints", {})
        constraints = ExtractedConstraints(
            role=constraints_raw.get("role", ""),
            seniority=constraints_raw.get("seniority", ""),
            skills=constraints_raw.get("skills", []) or [],
            test_types=constraints_raw.get("test_types", []) or [],
            languages=constraints_raw.get("languages", []) or [],
            industry=constraints_raw.get("industry", ""),
            specific_assessments=constraints_raw.get("specific_assessments", []) or [],
            query_text=constraints_raw.get("query_text", ""),
            include_personality=constraints_raw.get("include_personality", True),
        )

        intent = raw.get("intent", "clarify")
        if intent not in ("clarify", "recommend", "refine", "compare", "refuse"):
            intent = "clarify"

        return LLMExtractionResult(
            reasoning=raw.get("reasoning", ""),
            intent=intent,
            constraints=constraints,
            draft_reply=raw.get("draft_reply", ""),
            clarifying_question=raw.get("clarifying_question", ""),
            additions=raw.get("additions", []) or [],
            removals=raw.get("removals", []) or [],
            compare_items=raw.get("compare_items", []) or [],
            previous_shortlist_names=raw.get("previous_shortlist_names", []) or [],
            is_confirmation=raw.get("is_confirmation", False),
        )
    except Exception as e:
        logger.error(f"Failed to parse extraction result: {e}")
        return LLMExtractionResult(intent="clarify")


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_chat(request: ChatRequest) -> ChatResponse:
    """
    Main stateless chat handler. Reconstructs all context from message history.

    f(full_message_history) -> response
    """
    request_start = time.time()
    messages = request.messages

    # ── Edge case: empty messages → opening clarification ─────────────────
    if not messages:
        logger.info("Empty messages → opening clarification")
        return FALLBACK_CLARIFY

    # ── Prepare conversation for LLM ──────────────────────────────────────
    conv_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]

    # ── Step 1: LLM intent extraction ─────────────────────────────────────
    logger.info(f"Extracting intent from {len(conv_messages)} messages")
    raw_extraction = await extract_intent(
        system_prompt=INTENT_EXTRACTION_SYSTEM_PROMPT,
        conversation_messages=conv_messages,
        request_start_time=request_start,
    )

    if raw_extraction is None:
        # LLM failed — return a safe fallback instead of blind-searching
        # (blind search on off-topic text returns random results)
        logger.warning("LLM extraction failed → returning clarification fallback")
        return FALLBACK_CLARIFY

    extraction = _parse_extraction(raw_extraction)
    logger.info(f"Detected intent: {extraction.intent}")

    # ── Step 2: Route by intent ───────────────────────────────────────────

    if extraction.intent == "refuse":
        return await _handle_refuse(extraction, conv_messages, request_start)

    elif extraction.intent == "clarify":
        return await _handle_clarify(extraction)

    elif extraction.intent == "recommend":
        return await _handle_recommend(extraction, conv_messages, request_start)

    elif extraction.intent == "refine":
        return await _handle_refine(extraction, conv_messages, request_start)

    elif extraction.intent == "compare":
        return await _handle_compare(extraction, conv_messages, request_start)

    else:
        # Unknown intent → clarify
        return FALLBACK_CLARIFY


# ── Intent handlers ───────────────────────────────────────────────────────────

async def _handle_clarify(extraction: LLMExtractionResult) -> ChatResponse:
    """Handle clarification intent — ask a focused question, no recommendations."""
    reply = extraction.clarifying_question or extraction.draft_reply
    if not reply:
        reply = (
            "I can help narrow that down. "
            "Could you tell me more about the role and what specific competencies "
            "you need to evaluate?"
        )
    return ChatResponse(
        reply=reply,
        recommendations=[],
        end_of_conversation=False,
    )


async def _handle_recommend(
    extraction: LLMExtractionResult,
    conv_messages: list[dict],
    request_start: float,
) -> ChatResponse:
    """Handle recommend intent — search catalog and return shortlist."""
    constraints = extraction.constraints

    # Build search query from all extracted constraints
    query = _build_query_from_constraints(constraints.model_dump())
    if not query.strip():
        query = extraction.draft_reply  # fallback to draft as query

    logger.info(f"Searching catalog with query: {query[:100]}...")

    # Search with filters
    results = search(
        query=query,
        top_k=10,
        job_level=constraints.seniority,
        test_type_filter=constraints.test_types if constraints.test_types else None,
        include_personality=constraints.include_personality,
    )

    if not results:
        # No results — try broader search with just skills
        if constraints.skills:
            results = search(
                query=" ".join(constraints.skills),
                top_k=10,
                include_personality=constraints.include_personality,
            )

    # Validate all recommendations
    validated = _validate_recommendations(results)

    # Generate a polished reply
    reply = extraction.draft_reply
    if not reply or len(reply) < 20:
        # Generate reply via LLM if the draft is empty/short
        elapsed = time.time() - request_start
        if elapsed < TOTAL_BUDGET_SECONDS - 5:
            context = f"User needs: {query}\nNumber of results: {len(validated)}"
            result_summary = "\n".join(
                f"- {r.name} ({r.test_type})" for r in validated[:10]
            )
            formatted_prompt = REPLY_FORMATTING_PROMPT.format(
                context=context, results=result_summary
            )
            llm_reply = await generate_reply(
                system_prompt=formatted_prompt,
                context=f"Please recommend these assessments for: {query}",
                request_start_time=request_start,
            )
            if llm_reply:
                reply = llm_reply

        reply = "Here is the recommended assessment shortlist for those requirements."

    return ChatResponse(
        reply=reply,
        recommendations=validated[:10],
        end_of_conversation=extraction.is_confirmation,
    )


async def _handle_refine(
    extraction: LLMExtractionResult,
    conv_messages: list[dict],
    request_start: float,
) -> ChatResponse:
    """Handle refine intent — update existing shortlist with additions/removals."""
    constraints = extraction.constraints

    # Reconstruct previous shortlist from extraction
    prev_names = extraction.previous_shortlist_names
    prev_items = []
    if prev_names:
        for name in prev_names:
            item = lookup_by_name(name)
            if item:
                prev_items.append(item_to_recommendation(item))

    # Build query for finding new items
    query = _build_query_from_constraints(constraints.model_dump())

    # Apply refinements
    refined = search_for_refinement(
        query=query,
        existing_items=prev_items,
        additions=extraction.additions,
        removals=extraction.removals,
        top_k=10,
        job_level=constraints.seniority,
        include_personality=constraints.include_personality,
    )

    # Validate
    validated = _validate_recommendations(refined)

    # Use draft reply
    reply = extraction.draft_reply
    if not reply:
        added_desc = f", adding focus on {', '.join(extraction.additions)}" if extraction.additions else ""
        removed_desc = f", removing {', '.join(extraction.removals)}" if extraction.removals else ""
        reply = f"I've updated your assessment shortlist{added_desc}{removed_desc}."

    return ChatResponse(
        reply=reply,
        recommendations=validated[:10],
        end_of_conversation=extraction.is_confirmation,
    )


async def _handle_compare(
    extraction: LLMExtractionResult,
    conv_messages: list[dict],
    request_start: float,
) -> ChatResponse:
    """Handle compare intent — ground comparison in actual catalog data."""
    compare_names = extraction.compare_items

    # Look up the actual catalog items
    catalog_items = lookup_items_for_compare(compare_names)

    # Build catalog data string for the comparison prompt
    if catalog_items:
        catalog_data = json.dumps(catalog_items, indent=2, default=str)
    else:
        catalog_data = "No matching assessments found in the catalog."

    # Generate grounded comparison via LLM
    formatted_prompt = COMPARE_SYSTEM_PROMPT.format(catalog_data=catalog_data)
    not_found = [n for n in compare_names if not any(
        n.lower() in item.get("name", "").lower() for item in catalog_items
    )]

    context = f"Compare these assessments: {', '.join(compare_names)}"
    if not_found:
        context += f"\nNote: The following were not found in the catalog: {', '.join(not_found)}"

    reply = await generate_reply(
        system_prompt=formatted_prompt,
        context=context,
        request_start_time=request_start,
    )

    if not reply:
        reply = extraction.draft_reply or "I couldn't generate a comparison at this time."

    # Keep existing shortlist if one exists
    prev_names = extraction.previous_shortlist_names
    existing_recs = []
    if prev_names:
        for name in prev_names:
            item = lookup_by_name(name)
            if item:
                rec = item_to_recommendation(item)
                if validate_url(rec["url"]):
                    existing_recs.append(Recommendation(**rec))

    return ChatResponse(
        reply=reply,
        recommendations=existing_recs,
        end_of_conversation=False,
    )


async def _handle_refuse(
    extraction: LLMExtractionResult,
    conv_messages: list[dict],
    request_start: float,
) -> ChatResponse:
    """Handle refusal — off-topic, legal, hiring advice, prompt injection."""
    reply = extraction.draft_reply
    if not reply:
        reply = FALLBACK_REFUSE.reply

    # Keep existing shortlist if one was previously committed
    prev_names = extraction.previous_shortlist_names
    existing_recs = []
    if prev_names:
        for name in prev_names:
            item = lookup_by_name(name)
            if item:
                rec = item_to_recommendation(item)
                if validate_url(rec["url"]):
                    existing_recs.append(Recommendation(**rec))

    return ChatResponse(
        reply=reply,
        recommendations=existing_recs,
        end_of_conversation=False,
    )
