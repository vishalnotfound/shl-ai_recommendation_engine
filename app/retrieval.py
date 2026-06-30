"""
Retrieval layer — TF-IDF + keyword matching over the catalog.

Built at startup, queried per request. The retrieval layer is the ONLY source
of truth for what goes into `recommendations`. The LLM never invents items.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import get_catalog, lookup_by_name, item_to_recommendation

logger = logging.getLogger(__name__)

# ── Module-level index state ──────────────────────────────────────────────────

_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None
_catalog_items: list[dict] = []

# The OPQ32r entity — near-default inclusion for hiring contexts (§10c)
OPQ_NAME = "Occupational Personality Questionnaire OPQ32r"


def build_index() -> None:
    """Build the TF-IDF index over the catalog. Call once at startup."""
    global _vectorizer, _tfidf_matrix, _catalog_items

    _catalog_items = get_catalog()
    if not _catalog_items:
        raise RuntimeError("Cannot build index: catalog is empty")

    # Build corpus: each document = name + description + keys + job_levels
    corpus = []
    for item in _catalog_items:
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
            " ".join(item.get("languages", [])),
        ]
        corpus.append(" ".join(parts).lower())

    _vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),  # unigrams + bigrams for better phrase matching
        sublinear_tf=True,
    )
    _tfidf_matrix = _vectorizer.fit_transform(corpus)
    logger.info(f"TF-IDF index built: {_tfidf_matrix.shape[0]} docs, {_tfidf_matrix.shape[1]} features")


def _apply_hard_filters(
    items: list[dict],
    scores: list[float],
    job_level: str = "",
    test_type_filter: list[str] = None,
) -> tuple[list[dict], list[float]]:
    """Apply hard filters to narrow candidates by job level and test type."""
    if not job_level and not test_type_filter:
        return items, scores

    filtered_items = []
    filtered_scores = []

    for item, score in zip(items, scores):
        # Job level filter
        if job_level:
            item_levels = [l.lower() for l in item.get("job_levels", [])]
            level_lower = job_level.lower()
            # Check if any item level contains or is contained by the filter
            level_match = any(
                level_lower in il or il in level_lower
                for il in item_levels
            )
            if not level_match and item_levels:
                # Soft penalty instead of hard filter — reduce score
                score *= 0.5

        # Test type filter (e.g. user wants "personality" → filter for P)
        if test_type_filter:
            item_types = set(item.get("test_type", "").split(","))
            if not item_types.intersection(set(test_type_filter)):
                score *= 0.3

        filtered_items.append(item)
        filtered_scores.append(score)

    return filtered_items, filtered_scores


def search(
    query: str,
    top_k: int = 10,
    job_level: str = "",
    test_type_filter: list[str] = None,
    include_personality: bool = True,
    exclude_names: list[str] = None,
) -> list[dict]:
    """
    Search the catalog using TF-IDF similarity + optional hard filters.

    Returns up to top_k catalog items as recommendation dicts.
    """
    if _vectorizer is None or _tfidf_matrix is None:
        build_index()

    if not query.strip():
        return []

    # Transform query
    query_vec = _vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    # Pair items with scores
    scored = list(zip(_catalog_items, similarities.tolist()))

    # Apply hard filters
    items_list = [s[0] for s in scored]
    scores_list = [s[1] for s in scored]
    items_list, scores_list = _apply_hard_filters(
        items_list, scores_list, job_level, test_type_filter
    )

    # Re-pair and sort by score descending
    scored = list(zip(items_list, scores_list))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Exclude specific names (for refine removals)
    if exclude_names:
        exclude_lower = {n.lower().strip() for n in exclude_names}
        scored = [
            (item, score) for item, score in scored
            if item["name"].lower().strip() not in exclude_lower
        ]

    # Take top_k with minimum score threshold
    min_score = 0.02  # very low threshold to be inclusive
    results = []
    seen_urls = set()
    for item, score in scored:
        if score < min_score:
            break
        if item["url"] not in seen_urls:
            results.append(item)
            seen_urls.add(item["url"])
        if len(results) >= top_k:
            break

    # OPQ32r soft prior: include if hiring context and not already present (§10c)
    if include_personality and results:
        opq_present = any("opq" in r["name"].lower() for r in results)
        if not opq_present:
            opq_item = lookup_by_name(OPQ_NAME)
            if opq_item and len(results) < top_k:
                results.append(opq_item)
            elif opq_item and len(results) >= top_k:
                # Replace the lowest-scored item
                results[-1] = opq_item

    return [item_to_recommendation(r) for r in results]


def lookup_items_for_compare(names: list[str]) -> list[dict]:
    """
    Look up specific catalog items by name for comparison.
    Returns full catalog item dicts (not recommendation shape) for grounding.
    """
    results = []
    for name in names:
        item = lookup_by_name(name)
        if item:
            results.append(item)
        else:
            # Try fuzzy substring matching
            catalog = get_catalog()
            name_lower = name.lower().strip()
            for cat_item in catalog:
                if name_lower in cat_item["name"].lower():
                    results.append(cat_item)
                    break
    return results


def search_for_refinement(
    query: str,
    existing_items: list[dict],
    additions: list[str],
    removals: list[str],
    top_k: int = 10,
    job_level: str = "",
    include_personality: bool = True,
) -> list[dict]:
    """
    Refine an existing shortlist by adding/removing items.

    - `existing_items`: the current shortlist (as recommendation dicts)
    - `additions`: skills/items to add
    - `removals`: items/skills to remove
    """
    # Start with existing items, minus removals
    removal_lower = {r.lower().strip() for r in removals}
    kept = [
        item for item in existing_items
        if not any(rem in item["name"].lower() for rem in removal_lower)
    ]

    # Search for additions
    if additions:
        add_query = " ".join(additions) + " " + query
        new_items = search(
            add_query,
            top_k=top_k,
            job_level=job_level,
            include_personality=False,  # don't double-add OPQ
            exclude_names=[item["name"] for item in kept],
        )
        # Add new items up to the limit
        for item in new_items:
            if len(kept) >= top_k:
                break
            if not any(k["url"] == item["url"] for k in kept):
                kept.append(item)

    # OPQ soft prior
    if include_personality and kept:
        opq_present = any("opq" in r["name"].lower() for r in kept)
        if not opq_present and len(kept) < top_k:
            opq_item = lookup_by_name(OPQ_NAME)
            if opq_item:
                kept.append(item_to_recommendation(opq_item))

    return kept[:top_k]
