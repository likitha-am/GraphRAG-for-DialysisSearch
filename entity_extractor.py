"""
entity_extractor.py
Hybrid entity extraction: tries LLM first, auto-switches to rule-based on failure.
Extracts: patients, conditions, treatments, outcomes, risk factors.
"""

import json
import re
from typing import Any

from ollama_client import query_llm

KNOWN_CONDITIONS = {
    "fluid overload", "hypertension", "anemia", "diabetes mellitus",
    "peripheral neuropathy", "heart failure", "coronary artery disease",
    "atrial fibrillation", "peripheral vascular disease", "lupus nephritis",
    "secondary hyperparathyroidism", "malnutrition",
}

KNOWN_TREATMENTS = {
    "ultrafiltration", "ACE inhibitors", "erythropoietin", "insulin therapy",
    "gabapentin", "loop diuretics", "beta blockers", "calcium channel blockers",
    "nitrates", "anticoagulation", "rate control", "antiplatelet therapy",
    "phosphate binders", "vitamin D analogs", "ARBs", "digoxin",
    "immunosuppressants", "cinacalcet", "nutritional supplementation",
}

KNOWN_OUTCOMES = {"improved", "stable", "declined"}


def _rule_based_extract(patient: dict[str, Any]) -> dict[str, Any]:
    """Deterministic extraction directly from structured patient dict."""
    conditions = [c.lower().strip() for c in patient.get("conditions", [])]
    treatments = [t.strip() for t in patient.get("treatments", [])]
    outcome = patient.get("outcome", "unknown").lower().strip()
    has_cardiac = bool(patient.get("prior_cardiac_event", False))

    risk_factors = []
    if has_cardiac:
        risk_factors.append("prior cardiac event")
    if "diabetes mellitus" in conditions:
        risk_factors.append("diabetes")
    if "hypertension" in conditions:
        risk_factors.append("hypertension")
    if patient.get("age", 0) > 65:
        risk_factors.append("advanced age")

    return {
        "patient_id": patient["patient_id"],
        "age": patient.get("age"),
        "conditions": conditions,
        "treatments": treatments,
        "outcome": outcome,
        "prior_cardiac_event": has_cardiac,
        "risk_factors": risk_factors,
        "extraction_method": "rule_based",
    }


def _llm_extract(patient: dict[str, Any]) -> dict[str, Any] | None:
    """
    Attempt LLM-assisted extraction for richer entity normalization.
    Returns None if LLM is unavailable or response is unparseable.
    """
    prompt = f"""You are a clinical NLP system. Given this dialysis patient record, extract entities.

Patient record:
{json.dumps(patient, indent=2)}

Return ONLY valid JSON with these exact keys (no markdown, no explanation):
{{
  "patient_id": string,
  "age": integer,
  "conditions": [list of condition strings],
  "treatments": [list of treatment strings],
  "outcome": string,
  "prior_cardiac_event": boolean,
  "risk_factors": [list of risk factor strings]
}}"""

    raw = query_llm(prompt)
    if not raw:
        return None

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
        required = {"patient_id", "conditions", "treatments", "outcome", "prior_cardiac_event"}
        if not required.issubset(data.keys()):
            return None
        data["extraction_method"] = "llm"
        return data
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def extract_entities(patient: dict[str, Any]) -> dict[str, Any]:
    """
    Main extraction entry point. Tries LLM first; falls back to rules on failure.
    Always returns a complete entity dict.
    """
    result = _llm_extract(patient)
    if result is None:
        result = _rule_based_extract(patient)
    return result


def extract_all(patients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract entities from all patients. Logs per-patient method used."""
    extracted = []
    for p in patients:
        e = extract_entities(p)
        extracted.append(e)
    return extracted

