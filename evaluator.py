"""
evaluator.py
RAGAS-inspired evaluation metrics for the GraphRAG system.
Computes: answer_relevance, faithfulness, context_precision.
Uses LLM scoring with rule-based fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ollama_client import query_llm
from reasoner import ReasoningResult


@dataclass
class EvaluationResult:
    answer_relevance: float      # 0.0–1.0: does answer address the query?
    faithfulness: float          # 0.0–1.0: is answer grounded in retrieved evidence?
    context_precision: float     # 0.0–1.0: are context nodes actually relevant?
    overall: float               # weighted average
    notes: list[str]


def _rule_based_evaluate(
    query: str,
    result: ReasoningResult,
    entities: list[dict[str, Any]],
) -> EvaluationResult:
    """Deterministic RAGAS-style evaluation without LLM."""
    notes: list[str] = []

    # Answer relevance: does answer mention conditions/terms from query?
    query_terms = set(query.lower().split())
    answer_terms = set(result.answer.lower().split())
    overlap = query_terms & answer_terms
    relevance_score = min(len(overlap) / max(len(query_terms), 1), 1.0)
    notes.append(
        f"Answer relevance: {len(overlap)}/{len(query_terms)} query terms present"
    )

    # Faithfulness: are treatments cited actually present in matched patients?
    matched_treatments: set[str] = set()
    matched_ids = set(result.matched_entities)
    for e in entities:
        if e["patient_id"] in matched_ids:
            for tx in e.get("treatments", []):
                matched_treatments.add(tx.lower())

    cited_treatments = set(result.treatment_outcomes.keys())
    grounded = cited_treatments & {t.lower() for t in matched_treatments}
    faithful_score = (
        len(grounded) / len(cited_treatments) if cited_treatments else 0.0
    )
    notes.append(
        f"Faithfulness: {len(grounded)}/{len(cited_treatments)} cited treatments "
        f"grounded in matched patients"
    )

    # Context precision: do matched patients actually have required conditions?
    required = []
    # Extract required conditions from reasoning steps
    for step in result.reasoning_steps:
        if "conditions=" in step:
            m = re.search(r"conditions=\[([^\]]*)\]", step)
            if m:
                raw = m.group(1)
                required = [c.strip().strip("'\"") for c in raw.split(",") if c.strip()]

    if required and result.matched_entities:
        relevant_count = 0
        for e in entities:
            if e["patient_id"] in set(result.matched_entities):
                conds = set(e.get("conditions", []))
                if any(r in conds for r in required):
                    relevant_count += 1
        precision = relevant_count / len(result.matched_entities)
    else:
        precision = 1.0 if result.matched_entities else 0.0

    notes.append(
        f"Context precision: {precision:.2f} of matched patients have required conditions"
    )

    # Penalise fallback
    if result.fallback_used:
        relevance_score *= 0.8
        faithful_score *= 0.85
        precision *= 0.85
        notes.append("Scores penalised because fallback was used")

    overall = (relevance_score * 0.35 + faithful_score * 0.40 + precision * 0.25)

    return EvaluationResult(
        answer_relevance=round(relevance_score, 3),
        faithfulness=round(faithful_score, 3),
        context_precision=round(precision, 3),
        overall=round(overall, 3),
        notes=notes,
    )


def _llm_evaluate(
    query: str,
    result: ReasoningResult,
) -> EvaluationResult | None:
    """LLM-based evaluation. Returns None on failure."""
    prompt = f"""You are a clinical RAG system evaluator. Score this QA result on three metrics.

Query: "{query}"

Answer:
{result.answer}

Reasoning steps:
{chr(10).join(result.reasoning_steps)}

Score each metric from 0.0 to 1.0. Return ONLY valid JSON (no markdown):
{{
  "answer_relevance": float,
  "faithfulness": float,
  "context_precision": float,
  "notes": ["brief note per metric"]
}}

Definitions:
- answer_relevance: does the answer directly address the query?
- faithfulness: are claims in the answer supported by the evidence?
- context_precision: are the retrieved patients actually relevant to the query?"""

    raw = query_llm(prompt)
    if not raw:
        return None

    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
        ar = float(data.get("answer_relevance", 0.0))
        f = float(data.get("faithfulness", 0.0))
        cp = float(data.get("context_precision", 0.0))
        overall = ar * 0.35 + f * 0.40 + cp * 0.25
        return EvaluationResult(
            answer_relevance=round(ar, 3),
            faithfulness=round(f, 3),
            context_precision=round(cp, 3),
            overall=round(overall, 3),
            notes=data.get("notes", []),
        )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def evaluate(
    query: str,
    result: ReasoningResult,
    entities: list[dict[str, Any]],
) -> EvaluationResult:
    """
    Main evaluation entry point.
    Tries LLM evaluation first; falls back to rule-based.
    """
    eval_result = _llm_evaluate(query, result)
    if eval_result is None:
        eval_result = _rule_based_evaluate(query, result, entities)
    return eval_result

