"""
similarity_engine.py
Patient similarity using weighted Jaccard similarity.
Cardiac history is given higher weight as a critical risk factor.
"""

from __future__ import annotations

from typing import Any

CARDIAC_WEIGHT = 3.0   # cardiac event counts as N regular conditions
CONDITION_WEIGHT = 1.0
TREATMENT_WEIGHT = 0.5  # treatments less diagnostic than conditions


def _weighted_condition_set(entity: dict[str, Any]) -> dict[str, float]:
    """Build a weighted feature dict for Jaccard computation."""
    features: dict[str, float] = {}

    for cond in entity.get("conditions", []):
        features[f"cond:{cond}"] = CONDITION_WEIGHT

    for tx in entity.get("treatments", []):
        features[f"tx:{tx}"] = TREATMENT_WEIGHT

    if entity.get("prior_cardiac_event"):
        features["risk:cardiac"] = CARDIAC_WEIGHT

    return features


def weighted_jaccard(
    a: dict[str, float], b: dict[str, float]
) -> float:
    """
    Weighted Jaccard similarity: sum(min(a,b)) / sum(max(a,b))
    Returns 0.0 if both sets are empty.
    """
    all_keys = set(a) | set(b)
    if not all_keys:
        return 0.0

    numerator = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in all_keys)
    denominator = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in all_keys)
    return numerator / denominator if denominator > 0 else 0.0


def find_similar_patients(
    target: dict[str, Any],
    all_entities: list[dict[str, Any]],
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[tuple[str, float]]:
    """
    Return top-k most similar patients to target (excluding itself).
    Each result is (patient_id, similarity_score).
    """
    target_feats = _weighted_condition_set(target)
    scores: list[tuple[str, float]] = []

    for entity in all_entities:
        if entity["patient_id"] == target["patient_id"]:
            continue
        feats = _weighted_condition_set(entity)
        sim = weighted_jaccard(target_feats, feats)
        if sim >= min_similarity:
            scores.append((entity["patient_id"], sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def add_similarity_edges(
    graph,
    entities: list[dict[str, Any]],
    top_k: int = 3,
    min_similarity: float = 0.15,
) -> None:
    """
    Add SIMILAR_TO edges between patients in the graph.
    Only adds edges above min_similarity threshold.
    """
    for entity in entities:
        similar = find_similar_patients(
            entity, entities, top_k=top_k, min_similarity=min_similarity
        )
        for sim_pid, score in similar:
            graph.add_edge(
                entity["patient_id"],
                sim_pid,
                "SIMILAR_TO",
                weight=round(score, 4),
            )

