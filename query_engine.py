"""
query_engine.py
Converts natural language queries into structured query objects.
Uses LLM for rich parsing; falls back to rule-based keyword extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ollama_client import query_llm

CONDITION_KEYWORDS = {
    "fluid overload": ["fluid overload", "fluid retention", "volume overload", "edema"],
    "hypertension": ["hypertension", "high blood pressure", "elevated bp"],
    "heart failure": ["heart failure", "cardiac failure", "chf"],
    "coronary artery disease": ["coronary artery disease", "cad", "coronary disease"],
    "atrial fibrillation": ["atrial fibrillation", "afib", "a-fib"],
    "diabetes mellitus": ["diabetes", "diabetic", "dm"],
    "anemia": ["anemia", "anaemia", "low hemoglobin"],
}

TREATMENT_KEYWORDS = {
    "ultrafiltration": ["ultrafiltration", "fluid removal"],
    "ACE inhibitors": ["ace inhibitors", "ace-i", "lisinopril", "enalapril"],
    "loop diuretics": ["loop diuretics", "furosemide", "lasix"],
    "beta blockers": ["beta blockers", "metoprolol", "carvedilol"],
}


@dataclass
class StructuredQuery:
    original: str
    conditions_required: list[str] = field(default_factory=list)
    conditions_optional: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    treatments_of_interest: list[str] = field(default_factory=list)
    outcome_filter: Optional[str] = None
    require_cardiac_history: bool = False
    relaxed: bool = False  # True when constraints were relaxed for fallback


def _rule_based_parse(query: str) -> StructuredQuery:
    """Keyword-based query parser as fallback."""
    q = query.lower()
    sq = StructuredQuery(original=query)

    for canonical, aliases in CONDITION_KEYWORDS.items():
        if any(a in q for a in aliases):
            sq.conditions_required.append(canonical)

    for canonical, aliases in TREATMENT_KEYWORDS.items():
        if any(a in q for a in aliases):
            sq.treatments_of_interest.append(canonical)

    cardiac_phrases = [
        "cardiac", "heart", "cardiac event", "cardiac history",
        "prior cardiac", "cardiac events", "myocardial",
    ]
    if any(p in q for p in cardiac_phrases):
        sq.require_cardiac_history = True
        sq.risk_factors.append("prior cardiac event")

    if "improved" in q:
        sq.outcome_filter = "improved"
    elif "stable" in q:
        sq.outcome_filter = "stable"
    elif "declined" in q or "worsened" in q:
        sq.outcome_filter = "declined"

    return sq


def _llm_parse(query: str) -> StructuredQuery | None:
    """LLM-based query parsing. Returns None on failure."""
    prompt = f"""You are a clinical query parser for a dialysis patient knowledge graph.

Parse this query into structured components. Return ONLY valid JSON (no markdown):
{{
  "conditions_required": ["list of conditions that MUST be present"],
  "conditions_optional": ["list of conditions that are optional/preferred"],
  "risk_factors": ["list of risk factors mentioned"],
  "treatments_of_interest": ["treatments mentioned or implied"],
  "outcome_filter": null or "improved" or "stable" or "declined",
  "require_cardiac_history": true or false
}}

Query: "{query}"

Map clinical concepts to these known values:
Conditions: fluid overload, hypertension, anemia, diabetes mellitus, heart failure,
            coronary artery disease, atrial fibrillation, peripheral neuropathy,
            peripheral vascular disease, lupus nephritis, secondary hyperparathyroidism, malnutrition
Risk factors: prior cardiac event, diabetes, hypertension, advanced age"""

    raw = query_llm(prompt)
    if not raw:
        return None

    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
        sq = StructuredQuery(
            original=query,
            conditions_required=data.get("conditions_required", []),
            conditions_optional=data.get("conditions_optional", []),
            risk_factors=data.get("risk_factors", []),
            treatments_of_interest=data.get("treatments_of_interest", []),
            outcome_filter=data.get("outcome_filter"),
            require_cardiac_history=bool(data.get("require_cardiac_history", False)),
        )
        return sq
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def parse_query(query: str) -> StructuredQuery:
    """
    Main query parsing entry point.
    Tries LLM first; uses rule-based fallback on failure.
    """
    result = _llm_parse(query)
    if result is None:
        result = _rule_based_parse(query)
    return result

