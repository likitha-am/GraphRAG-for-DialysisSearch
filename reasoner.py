"""
reasoner.py
Multi-hop graph reasoning engine.
Supports: BFS traversal, cycle prevention, constraint relaxation,
          missing link inference, and full explainability output.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from graph_engine import KnowledgeGraph
from query_engine import StructuredQuery
from similarity_engine import find_similar_patients


# ── Data structures ───────────────────────────────────────────────────────────

class ReasoningResult:
    def __init__(self) -> None:
        self.answer: str = ""
        self.reasoning_steps: list[str] = []
        self.traversal_paths: list[list[str]] = []
        self.matched_entities: list[str] = []
        self.inferred_links: list[str] = []
        self.final_decision: str = ""
        self.confidence: float = 0.0
        self.fallback_used: bool = False
        self.fallback_explanation: str = ""
        self.treatment_outcomes: dict[str, dict[str, int]] = {}  # tx -> {outcome: count}


# ── Patient matching ──────────────────────────────────────────────────────────

def _match_patients(
    sq: StructuredQuery,
    entities: list[dict[str, Any]],
    graph: KnowledgeGraph,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Find patients matching the structured query.
    Returns (matched_list, fallback_used).
    Relaxes constraints iteratively if no strict matches found.
    """
    def _score(entity: dict[str, Any], strict: bool) -> float:
        conditions = set(entity.get("conditions", []))
        score = 0.0
        matched_required = 0

        for req in sq.conditions_required:
            if req in conditions:
                matched_required += 1
                score += 2.0
            elif not strict:
                score += 0.0

        if strict and matched_required < len(sq.conditions_required):
            return -1.0  # disqualify

        for opt in sq.conditions_optional:
            if opt in conditions:
                score += 1.0

        if sq.require_cardiac_history:
            if entity.get("prior_cardiac_event"):
                score += 3.0
            elif strict:
                return -1.0

        if sq.outcome_filter and entity.get("outcome") == sq.outcome_filter:
            score += 1.5

        return score

    # Strict pass
    strict_matches = [
        (e, _score(e, strict=True))
        for e in entities
        if _score(e, strict=True) >= 0
    ]
    strict_matches = [(e, s) for e, s in strict_matches if s > 0]
    strict_matches.sort(key=lambda x: x[1], reverse=True)

    if strict_matches:
        return [e for e, _ in strict_matches], False

    # Relaxed pass — drop cardiac requirement, keep conditions
    relaxed: list[tuple[dict[str, Any], float]] = []
    for e in entities:
        conds = set(e.get("conditions", []))
        req_matched = sum(1 for r in sq.conditions_required if r in conds)
        if req_matched > 0:
            relaxed.append((e, float(req_matched)))

    relaxed.sort(key=lambda x: x[1], reverse=True)
    if relaxed:
        return [e for e, _ in relaxed], True

    # Last resort: nearest neighbor by similarity
    if entities:
        # Synthesize a pseudo-entity from the query for similarity
        pseudo = {
            "patient_id": "__query__",
            "conditions": sq.conditions_required + sq.conditions_optional,
            "treatments": sq.treatments_of_interest,
            "prior_cardiac_event": sq.require_cardiac_history,
        }
        similar = find_similar_patients(pseudo, entities, top_k=3, min_similarity=0.0)
        ids = {pid for pid, _ in similar}
        nearest = [e for e in entities if e["patient_id"] in ids]
        return nearest, True

    return [], True


# ── Multi-hop reasoning ───────────────────────────────────────────────────────

def _gather_treatments_and_outcomes(
    patient_ids: list[str],
    graph: KnowledgeGraph,
    result: ReasoningResult,
    step_offset: int = 0,
) -> dict[str, dict[str, int]]:
    """
    Hop from patients → treatments → outcomes.
    Populates result.traversal_paths and returns treatment_outcomes dict.
    """
    tx_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hop_descriptions = {
        "RECEIVED_TREATMENT": "received treatment",
        "TREATMENT_LED_TO": "led to outcome",
    }

    for pid in patient_ids:
        path = [pid]

        # Hop 1: patient → treatment
        treatment_edges = graph.neighbors(pid, "RECEIVED_TREATMENT")
        for tx_id, _, _ in treatment_edges:
            tx_path = path + [tx_id]

            # Hop 2: treatment → outcome
            outcome_edges = graph.neighbors(tx_id, "TREATMENT_LED_TO")
            for out_id, _, weight in outcome_edges:
                full_path = tx_path + [out_id]
                result.traversal_paths.append(full_path)

                tx_name = tx_id.replace("tx:", "")
                out_name = out_id.replace("outcome:", "")
                tx_outcomes[tx_name][out_name] += 1

    return {k: dict(v) for k, v in tx_outcomes.items()}


def _rank_treatments(
    tx_outcomes: dict[str, dict[str, int]]
) -> list[tuple[str, float, dict[str, int]]]:
    """
    Score treatments by efficacy: improved > stable > declined.
    Returns sorted list of (treatment, score, outcome_counts).
    """
    WEIGHTS = {"improved": 2.0, "stable": 1.0, "declined": -1.0}
    ranked: list[tuple[str, float, dict[str, int]]] = []

    for tx, outcomes in tx_outcomes.items():
        total = sum(outcomes.values())
        if total == 0:
            continue
        score = sum(WEIGHTS.get(o, 0) * c for o, c in outcomes.items()) / total
        ranked.append((tx, round(score, 3), outcomes))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _compute_confidence(
    matched: list[dict[str, Any]],
    fallback: bool,
    ranked_treatments: list[tuple[str, float, dict[str, int]]],
    sq: StructuredQuery,
) -> float:
    """
    Heuristic confidence score [0.0, 1.0].
    Based on match quality, number of supporting patients, and evidence strength.
    """
    base = 0.0

    if not fallback:
        base += 0.4
    else:
        base += 0.15

    # Evidence volume
    n = min(len(matched), 10)
    base += n * 0.04

    # Treatment evidence quality
    if ranked_treatments:
        top_score = ranked_treatments[0][1]
        base += max(0.0, top_score) * 0.1

    # Cardiac specificity bonus
    if sq.require_cardiac_history and not fallback:
        cardiac_matched = sum(
            1 for p in matched if p.get("prior_cardiac_event")
        )
        frac = cardiac_matched / max(len(matched), 1)
        base += frac * 0.15

    return min(round(base, 3), 1.0)


# ── Main reasoning entry point ────────────────────────────────────────────────

def reason(
    sq: StructuredQuery,
    entities: list[dict[str, Any]],
    graph: KnowledgeGraph,
) -> ReasoningResult:
    """
    Full multi-hop reasoning pipeline.
    Returns a complete ReasoningResult with answer, paths, confidence, and explainability.
    """
    result = ReasoningResult()
    result.reasoning_steps.append(
        f"Step 1 — Query understood: conditions={sq.conditions_required}, "
        f"cardiac_required={sq.require_cardiac_history}"
    )

    # Step 1: Find matching patients
    matched, fallback = _match_patients(sq, entities, graph)
    result.fallback_used = fallback
    result.matched_entities = [e["patient_id"] for e in matched]

    if fallback:
        result.fallback_explanation = (
            "No patients matched all required constraints. "
            "Constraints were relaxed: cardiac history requirement dropped "
            "and/or only partial condition matches considered. "
            "Results represent nearest clinical neighbors."
        )
        result.inferred_links.append(
            "Inferred connection via condition overlap without full constraint satisfaction"
        )

    result.reasoning_steps.append(
        f"Step 2 — Patient matching: found {len(matched)} patients "
        f"({'strict' if not fallback else 'relaxed/fallback'} criteria)"
    )

    if not matched:
        result.answer = (
            "No relevant patients found in the knowledge graph for this query."
        )
        result.final_decision = "insufficient_data"
        result.confidence = 0.0
        return result

    # Step 2: Find similar patients (hop 1)
    similar_pool: set[str] = set(result.matched_entities)
    for entity in matched:
        similar = find_similar_patients(entity, entities, top_k=3, min_similarity=0.1)
        for sim_pid, sim_score in similar:
            similar_pool.add(sim_pid)
            result.inferred_links.append(
                f"Similarity link: {entity['patient_id']} ↔ {sim_pid} "
                f"(score={sim_score:.3f})"
            )

    result.reasoning_steps.append(
        f"Step 3 — Similarity expansion: pool grew to {len(similar_pool)} patients"
    )

    # Step 3: Multi-hop treatment→outcome traversal
    all_pool_ids = list(similar_pool)
    tx_outcomes = _gather_treatments_and_outcomes(all_pool_ids, graph, result)

    result.reasoning_steps.append(
        f"Step 4 — Treatment traversal: found {len(tx_outcomes)} distinct treatments "
        f"across {len(result.traversal_paths)} traversal paths"
    )

    # Step 4: Rank treatments
    ranked = _rank_treatments(tx_outcomes)
    result.treatment_outcomes = {tx: oc for tx, _, oc in ranked}

    result.reasoning_steps.append(
        f"Step 5 — Treatment ranking: top treatment = "
        f"'{ranked[0][0] if ranked else 'none'}'"
    )

    # Step 5: Compute confidence
    result.confidence = _compute_confidence(matched, fallback, ranked, sq)

    # Step 6: Build final answer
    if not ranked:
        result.answer = (
            "Matching patients were found but no treatment-outcome data "
            "could be extracted from the graph."
        )
        result.final_decision = "no_treatment_data"
        return result

    # Compose human-readable answer
    lines: list[str] = []
    cond_list = ", ".join(sq.conditions_required) or "the specified conditions"
    lines.append(
        f"For dialysis patients with {cond_list}"
        + (" and prior cardiac events" if sq.require_cardiac_history else "")
        + f", evidence from {len(matched)} matched patients"
        + (f" (+ {len(similar_pool) - len(matched)} similar)" if len(similar_pool) > len(matched) else "")
        + " suggests the following treatments:"
    )
    lines.append("")

    for i, (tx, score, outcomes) in enumerate(ranked[:5], 1):
        outcome_summary = ", ".join(
            f"{o}: {c} patient{'s' if c != 1 else ''}"
            for o, c in sorted(outcomes.items(), key=lambda x: x[1], reverse=True)
        )
        efficacy = "high" if score >= 1.5 else "moderate" if score >= 0.5 else "low"
        lines.append(
            f"  {i}. {tx} — efficacy score {score:.2f} ({efficacy}) | {outcome_summary}"
        )

    lines.append("")
    lines.append(f"Confidence: {result.confidence:.0%}")

    if fallback:
        lines.append(f"⚠ Fallback used: {result.fallback_explanation}")

    result.answer = "\n".join(lines)
    result.final_decision = f"top_treatment={ranked[0][0]}"

    return result

