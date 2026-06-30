"""
Groq LLM client — async, with explicit timeout, bounded retry, and
schema-valid fallback on failure.

The LLM is used for two narrow jobs:
  1. Intent extraction + constraint parsing from conversation history
  2. Phrasing natural-language reply text

It does NOT decide which catalog items exist.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from groq import AsyncGroq

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Model: fast Groq-hosted model for low-latency inference
MODEL = "llama-3.3-70b-versatile"

# Timeout per LLM call (seconds) — leaves headroom inside the 30s budget
LLM_CALL_TIMEOUT = 10

# Max time budget for the entire /chat handler
TOTAL_BUDGET_SECONDS = 28  # leave 2s margin under the 30s hard limit

# ── Client singleton ──────────────────────────────────────────────────────────

_client: Optional[AsyncGroq] = None


def get_client() -> Optional[AsyncGroq]:
    """Get or create the AsyncGroq client. Returns None if API key is missing."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY environment variable is not set. "
                "Set it in .env or as a platform secret."
            )
            return None
        _client = AsyncGroq(api_key=api_key, timeout=LLM_CALL_TIMEOUT)
    return _client


def validate_api_key() -> None:
    """Validate that GROQ_API_KEY is set at startup. Warn if not."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning(
            "GROQ_API_KEY environment variable is not set. "
            "LLM calls will fail — set it in .env or as a platform secret."
        )
    else:
        logger.info("GROQ_API_KEY is configured")


# ── Core LLM call ────────────────────────────────────────────────────────────

async def chat_completion(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    request_start_time: Optional[float] = None,
    json_mode: bool = False,
) -> Optional[str]:
    """
    Make an async chat completion call to Groq with timeout and retry.

    Args:
        system_prompt: The system prompt
        messages: List of {"role": ..., "content": ...} dicts
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        request_start_time: When the /chat request started (for budget tracking)
        json_mode: Whether to request JSON output mode

    Returns:
        The response text, or None if all attempts fail.
    """
    client = get_client()
    if client is None:
        logger.warning("No LLM client available (missing API key). Returning None.")
        return None
        
    start = request_start_time or time.time()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    # Cap message history to avoid exceeding context limits
    # Keep system + last 16 messages (8 turns × 2 roles)
    if len(full_messages) > 17:
        full_messages = [full_messages[0]] + full_messages[-16:]

    kwargs = {
        "model": MODEL,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(2):  # max 1 retry
        elapsed = time.time() - start
        remaining = TOTAL_BUDGET_SECONDS - elapsed

        if remaining < 3:
            logger.warning(f"LLM call skipped: only {remaining:.1f}s remaining in budget")
            return None

        try:
            logger.info(f"LLM call attempt {attempt + 1}, {remaining:.1f}s remaining")
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            logger.info(f"LLM call succeeded in {time.time() - start - elapsed:.1f}s")
            return content

        except Exception as e:
            elapsed_after = time.time() - start
            remaining_after = TOTAL_BUDGET_SECONDS - elapsed_after
            logger.warning(
                f"LLM call attempt {attempt + 1} failed: {type(e).__name__}: {e} "
                f"({remaining_after:.1f}s remaining)"
            )
            if attempt == 0 and remaining_after > 5:
                logger.info("Retrying LLM call...")
                continue
            else:
                logger.error("No more retries or insufficient time budget")
                return None

    return None


# ── Specialized callers ───────────────────────────────────────────────────────

async def extract_intent(
    system_prompt: str,
    conversation_messages: list[dict],
    request_start_time: float,
) -> Optional[dict]:
    """
    Call the LLM to extract intent and constraints from conversation history.
    Returns parsed JSON dict, or None on failure.
    """
    result = await chat_completion(
        system_prompt=system_prompt,
        messages=conversation_messages,
        temperature=0.2,
        max_tokens=1500,
        request_start_time=request_start_time,
        json_mode=True,
    )

    if result is None:
        return None

    # Parse JSON response
    try:
        # Clean up common LLM quirks
        text = result.strip()
        if text.startswith("```"):
            # Strip markdown code fences
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}\nRaw: {result[:500]}")
        return None


async def generate_reply(
    system_prompt: str,
    context: str,
    request_start_time: float,
) -> Optional[str]:
    """
    Call the LLM to generate a natural-language reply.
    Returns the reply text, or None on failure.
    """
    messages = [{"role": "user", "content": context}]
    result = await chat_completion(
        system_prompt=system_prompt,
        messages=messages,
        temperature=0.4,
        max_tokens=800,
        request_start_time=request_start_time,
    )
    return result
