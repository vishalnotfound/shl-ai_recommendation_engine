"""
Catalog download script — fetches the SHL product catalog JSON and saves a
filtered copy (Individual Test Solutions only) to data/catalog.json.

Run once at build time:  python -m scripts.download_catalog
"""
import json
import urllib.request
import os
import sys

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json")

# ── Letter-code legend for test_type ──────────────────────────────────────────
KEYS_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def derive_test_type(keys: list[str]) -> str:
    """Map a catalog item's `keys` array to a comma-separated letter-code string."""
    letters = sorted(set(KEYS_TO_LETTER.get(k, "") for k in keys) - {""})
    return ",".join(letters) if letters else "K"  # fallback to K if no mapping


def is_job_solution(item: dict) -> bool:
    """Return True if the item is a pre-packaged Job Solution (should be excluded)."""
    name = item.get("name", "")
    # All observed Job Solutions have "Solution" in the name
    # e.g. "Customer Service Phone Solution", "Entry Level Cashier Solution"
    return "Solution" in name


def process_catalog(raw_items: list[dict]) -> list[dict]:
    """Filter and transform the raw catalog into our internal format."""
    processed = []
    for item in raw_items:
        if is_job_solution(item):
            continue
        if item.get("status") != "ok":
            continue

        processed.append({
            "entity_id": item.get("entity_id", ""),
            "name": item.get("name", ""),
            "url": item.get("link", ""),  # rename link -> url
            "description": item.get("description", ""),
            "test_type": derive_test_type(item.get("keys", [])),
            "keys": item.get("keys", []),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": item.get("duration", ""),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
        })
    return processed


def main():
    print(f"Downloading catalog from {CATALOG_URL} ...")
    try:
        req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": "SHL-Chatbot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_text = resp.read().decode("utf-8")
        # The catalog JSON contains literal control characters (\r\n) inside
        # string values — use strict=False to accept them
        raw_data = json.loads(raw_text, strict=False)
    except Exception as e:
        print(f"ERROR: Failed to download catalog: {e}")
        sys.exit(1)

    print(f"Downloaded {len(raw_data)} raw items")

    processed = process_catalog(raw_data)
    print(f"After filtering: {len(processed)} Individual Test Solutions")

    # Count excluded
    excluded = [item for item in raw_data if is_job_solution(item)]
    print(f"Excluded {len(excluded)} Job Solutions: {[i['name'] for i in excluded]}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"Saved processed catalog to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
