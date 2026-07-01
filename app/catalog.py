"""
Catalog loader — reads the pre-processed catalog JSON from disk and provides
lookup utilities. Loaded once at startup, held in memory.
"""
from __future__ import annotations

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Letter-code legend (same as download script, kept here for runtime use) ───
KEYS_TO_LETTER: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

LETTER_TO_FULL: dict[str, str] = {v: k for k, v in KEYS_TO_LETTER.items()}

# ── Catalog singleton ────────────────────────────────────────────────────────

_catalog: list[dict] = []
_catalog_by_name: dict[str, dict] = {}
_catalog_urls: set[str] = set()


def load_catalog(path: Optional[str] = None) -> list[dict]:
    """Load the processed catalog JSON into memory. Idempotent."""
    global _catalog, _catalog_by_name, _catalog_urls

    if _catalog:
        return _catalog

    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Catalog file not found at {path}. "
            "Run `python -m scripts.download_catalog` first."
        )

    with open(path, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    # Build lookup indices
    _catalog_by_name = {}
    _catalog_urls = set()
    for item in _catalog:
        name_lower = item["name"].lower().strip()
        _catalog_by_name[name_lower] = item
        _catalog_urls.add(item["url"])

    logger.info(f"Loaded {len(_catalog)} catalog items")
    return _catalog


def get_catalog() -> list[dict]:
    """Get the loaded catalog (must call load_catalog first)."""
    if not _catalog:
        load_catalog()
    return _catalog


def get_catalog_by_name() -> dict[str, dict]:
    """Get the name→item lookup dict."""
    if not _catalog_by_name:
        load_catalog()
    return _catalog_by_name


def get_catalog_urls() -> set[str]:
    """Get the set of all valid catalog URLs."""
    if not _catalog_urls:
        load_catalog()
    return _catalog_urls


def lookup_by_name(name: str) -> Optional[dict]:
    """Look up a catalog item by exact or close name match."""
    by_name = get_catalog_by_name()
    key = name.lower().strip()

    # Exact match
    if key in by_name:
        return by_name[key]

    # Substring match (e.g. "OPQ32r" should find "Occupational Personality Questionnaire OPQ32r")
    for catalog_name, item in by_name.items():
        if key in catalog_name or catalog_name in key:
            return item

    return None


def find_items_by_names(names: list[str]) -> list[dict]:
    """Look up multiple catalog items by name. Returns only found items."""
    results = []
    seen_urls = set()
    for name in names:
        item = lookup_by_name(name)
        if item and item["url"] not in seen_urls:
            results.append(item)
            seen_urls.add(item["url"])
    return results


def validate_url(url: str) -> bool:
    """Check if a URL exists in the catalog. Prevents hallucinated URLs."""
    return url in get_catalog_urls()


def item_to_recommendation(item: dict) -> dict:
    """Convert a catalog item dict to the recommendation response shape."""
    # Format languages: show first 4, then (+N more)
    langs = item.get("languages", [])
    if len(langs) > 4:
        lang_str = ", ".join(langs[:4]) + f" (+{len(langs) - 4} more)"
    elif langs:
        lang_str = ", ".join(langs)
    else:
        lang_str = "—"

    # Format keys (category names)
    keys_list = item.get("keys", [])
    keys_str = ", ".join(keys_list) if keys_list else "—"

    return {
        "name": item["name"],
        "url": item["url"],
        "test_type": item["test_type"],
        "duration": item.get("duration", "") or "—",
        "keys": keys_str,
        "languages": lang_str,
    }
