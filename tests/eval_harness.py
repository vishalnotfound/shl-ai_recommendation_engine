"""
Evaluation harness — replays conversation traces against the running /chat endpoint,
computes Recall@10 per trace, and reports mean Recall@10 across all traces.

Usage:
    python -m tests.eval_harness [--base-url http://localhost:8000] [--traces-dir traces/]

The traces directory should contain .md files with conversation transcripts.
Each trace file should contain alternating user/assistant messages that can be
parsed to reconstruct the conversation flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx


def parse_trace_file(filepath: str) -> dict | None:
    """
    Parse a trace .md file to extract:
    - conversation turns (user messages + expected assistant responses)
    - expected final shortlist (recommendations from the last assistant turn)

    Returns a dict with:
        {
            "name": filename,
            "user_messages": [str, ...],
            "expected_shortlist": [{"name": str, "url": str, "test_type": str}, ...],
            "full_conversation": [{"role": str, "content": str}, ...]
        }
    or None if parsing fails.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Try to extract structured conversation data
    # The trace files are .md files — we need to parse them flexibly
    # Look for patterns like "User:" / "Assistant:" or role-based formatting

    turns = []
    user_messages = []
    expected_shortlist = []

    # Strategy 1: Look for JSON-like message blocks
    # Strategy 2: Look for User:/Assistant: patterns
    # Strategy 3: Look for table-based recommendation lists

    # Try to find user/assistant turns
    # Common patterns in trace docs:
    # - "**User:**" or "### User" or "User:" followed by text
    # - "**Assistant:**" or "### Assistant" or "Assistant:" followed by text

    lines = content.split("\n")
    current_role = None
    current_content = []

    for line in lines:
        line_stripped = line.strip()

        # Detect role markers
        if re.match(r'^(?:\*\*)?(?:User|Human)(?:\*\*)?[:\s]', line_stripped, re.IGNORECASE):
            # Save previous turn
            if current_role and current_content:
                text = "\n".join(current_content).strip()
                if text:
                    turns.append({"role": current_role, "content": text})
                    if current_role == "user":
                        user_messages.append(text)
            current_role = "user"
            # Get content after the role marker
            after = re.sub(r'^(?:\*\*)?(?:User|Human)(?:\*\*)?[:\s]*', '', line_stripped, flags=re.IGNORECASE)
            current_content = [after] if after else []

        elif re.match(r'^(?:\*\*)?(?:Assistant|Agent|Bot)(?:\*\*)?[:\s]', line_stripped, re.IGNORECASE):
            if current_role and current_content:
                text = "\n".join(current_content).strip()
                if text:
                    turns.append({"role": current_role, "content": text})
                    if current_role == "user":
                        user_messages.append(text)
            current_role = "assistant"
            after = re.sub(r'^(?:\*\*)?(?:Assistant|Agent|Bot)(?:\*\*)?[:\s]*', '', line_stripped, flags=re.IGNORECASE)
            current_content = [after] if after else []

        elif current_role:
            current_content.append(line)

    # Save last turn
    if current_role and current_content:
        text = "\n".join(current_content).strip()
        if text:
            turns.append({"role": current_role, "content": text})
            if current_role == "user":
                user_messages.append(text)

    if not user_messages:
        print(f"  WARNING: Could not parse any user messages from {filepath}")
        return None

    # Try to extract expected shortlist from the last assistant turn
    # Look for recommendation tables or lists
    last_assistant_turns = [t for t in turns if t["role"] == "assistant"]
    if last_assistant_turns:
        last_turn = last_assistant_turns[-1]["content"]
        # Try to find URLs in the last assistant turn
        urls = re.findall(r'https://www\.shl\.com/[^\s\)\"\']+', last_turn)
        # Try to find assessment names (in bold, in table rows, or in lists)
        names = re.findall(r'\*\*([^*]+)\*\*', last_turn)
        if not names:
            # Try table row pattern: | Name | URL | Type |
            table_rows = re.findall(r'\|([^|]+)\|([^|]+)\|([^|]+)\|', last_turn)
            for row in table_rows:
                name = row[0].strip()
                if name and name != "Name" and name != "---":
                    names.append(name)

        for name in names:
            expected_shortlist.append({"name": name.strip()})

    return {
        "name": os.path.basename(filepath),
        "user_messages": user_messages,
        "expected_shortlist": expected_shortlist,
        "full_conversation": turns,
        "content_hash": hashlib.md5(content.encode()).hexdigest(),
    }


def compute_recall_at_k(predicted: list[dict], expected: list[dict], k: int = 10) -> float:
    """
    Compute Recall@K = (relevant items in top K) / (total relevant items).

    Matching is done by normalized name comparison.
    """
    if not expected:
        return 1.0  # no expected items → trivially correct

    # Normalize names for matching
    predicted_names = {r["name"].lower().strip() for r in predicted[:k]}
    expected_names = {r["name"].lower().strip() for r in expected}

    # Count how many expected items appear in predictions
    hits = len(predicted_names & expected_names)

    # Also try substring matching for partial name matches
    for expected_name in expected_names:
        if expected_name not in predicted_names:
            for predicted_name in predicted_names:
                if expected_name in predicted_name or predicted_name in expected_name:
                    hits += 1
                    break

    recall = min(hits / len(expected_names), 1.0)
    return recall


async def replay_trace(
    base_url: str,
    trace: dict,
    delay: float = 0.5,
) -> dict:
    """
    Replay a single trace against the /chat endpoint.

    Sends user messages one at a time, accumulating the conversation.
    Returns the final response's recommendations for Recall@10 computation.
    """
    async with httpx.AsyncClient(timeout=35.0) as client:
        conversation = []
        last_response = None
        schema_valid = True

        for i, user_msg in enumerate(trace["user_messages"]):
            conversation.append({"role": "user", "content": user_msg})

            try:
                resp = await client.post(
                    f"{base_url}/chat",
                    json={"messages": conversation},
                )
                resp.raise_for_status()
                data = resp.json()

                # Schema validation
                if "reply" not in data or "recommendations" not in data or "end_of_conversation" not in data:
                    schema_valid = False
                if not isinstance(data.get("recommendations"), list):
                    schema_valid = False
                if not isinstance(data.get("end_of_conversation"), bool):
                    schema_valid = False

                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": data["reply"]})
                last_response = data

                if data.get("end_of_conversation"):
                    break

            except Exception as e:
                print(f"  ERROR on turn {i + 1}: {type(e).__name__}: {e}")
                schema_valid = False
                break

            # Small delay between turns to avoid rate limiting
            if delay > 0:
                time.sleep(delay)

        return {
            "trace_name": trace["name"],
            "final_recommendations": last_response.get("recommendations", []) if last_response else [],
            "expected_shortlist": trace["expected_shortlist"],
            "turns_played": len(conversation) // 2,
            "schema_valid": schema_valid,
            "end_of_conversation": last_response.get("end_of_conversation", False) if last_response else False,
        }


async def run_evaluation(base_url: str, traces_dir: str, delay: float = 0.5):
    """Run the full evaluation across all traces."""
    traces_path = Path(traces_dir)

    if not traces_path.exists():
        print(f"ERROR: Traces directory not found: {traces_dir}")
        print("Please place your trace .md files in the traces/ directory.")
        sys.exit(1)

    trace_files = sorted(traces_path.glob("*.md"))
    if not trace_files:
        print(f"ERROR: No .md trace files found in {traces_dir}")
        sys.exit(1)

    print(f"Found {len(trace_files)} trace files")

    # Parse all traces
    traces = []
    seen_hashes = set()
    for f in trace_files:
        trace = parse_trace_file(str(f))
        if trace:
            # Deduplicate by content hash
            if trace["content_hash"] in seen_hashes:
                print(f"  SKIPPING duplicate: {f.name}")
                continue
            seen_hashes.add(trace["content_hash"])
            traces.append(trace)
        else:
            print(f"  SKIPPING unparseable: {f.name}")

    print(f"Parsed {len(traces)} unique traces")
    print()

    # Check health endpoint first
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{base_url}/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
            print("✓ Health endpoint OK")
        except Exception as e:
            print(f"✗ Health endpoint failed: {e}")
            print(f"  Is the server running at {base_url}?")
            sys.exit(1)

    # Replay each trace
    results = []
    for i, trace in enumerate(traces):
        print(f"\n{'='*60}")
        print(f"Trace {i+1}/{len(traces)}: {trace['name']}")
        print(f"  User messages: {len(trace['user_messages'])}")
        print(f"  Expected shortlist: {len(trace['expected_shortlist'])} items")

        result = await replay_trace(base_url, trace, delay=delay)
        results.append(result)

        # Compute Recall@10
        recall = compute_recall_at_k(
            result["final_recommendations"],
            result["expected_shortlist"],
        )

        print(f"  Turns played: {result['turns_played']}")
        print(f"  Final recommendations: {len(result['final_recommendations'])}")
        print(f"  Schema valid: {'✓' if result['schema_valid'] else '✗'}")
        print(f"  End of conversation: {result['end_of_conversation']}")
        print(f"  Recall@10: {recall:.2%}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    recalls = []
    schema_pass = 0
    for r in results:
        recall = compute_recall_at_k(r["final_recommendations"], r["expected_shortlist"])
        recalls.append(recall)
        if r["schema_valid"]:
            schema_pass += 1

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0

    print(f"Traces evaluated: {len(results)}")
    print(f"Schema compliance: {schema_pass}/{len(results)}")
    print(f"Mean Recall@10: {mean_recall:.2%}")
    print(f"Individual recalls: {[f'{r:.2%}' for r in recalls]}")

    if mean_recall < 0.5:
        print("\n⚠ WARNING: Mean Recall@10 is below 50% — retrieval may need tuning")
    if schema_pass < len(results):
        print("\n⚠ WARNING: Some traces had schema validation failures")


def main():
    import asyncio

    parser = argparse.ArgumentParser(description="SHL Chatbot Evaluation Harness")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running server")
    parser.add_argument("--traces-dir", default="traces", help="Directory containing trace .md files")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between turns (seconds)")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.base_url, args.traces_dir, args.delay))


if __name__ == "__main__":
    main()
