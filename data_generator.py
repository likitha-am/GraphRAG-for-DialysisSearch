"""
data_generator.py
Generates >= 20 synthetic dialysis patient profiles with realistic clinical combinations.
Guarantees >= 5 patients with fluid overload + prior cardiac event.
"""

import json
import os
import random
from typing import Any

PATIENTS: list[dict[str, Any]] = [
    {
        "patient_id": "P001",
        "age": 67,
        "conditions": ["fluid overload", "hypertension", "anemia"],
        "treatments": ["ultrafiltration", "ACE inhibitors", "erythropoietin"],
        "outcome": "improved",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P002",
        "age": 72,
        "conditions": ["fluid overload", "diabetes mellitus", "peripheral neuropathy"],
        "treatments": ["ultrafiltration", "insulin therapy", "gabapentin"],
        "outcome": "stable",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P003",
        "age": 58,
        "conditions": ["fluid overload", "heart failure", "anemia"],
        "treatments": ["ultrafiltration", "loop diuretics", "beta blockers", "erythropoietin"],
        "outcome": "improved",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P004",
        "age": 65,
        "conditions": ["fluid overload", "coronary artery disease", "hypertension"],
        "treatments": ["ultrafiltration", "nitrates", "calcium channel blockers", "ACE inhibitors"],
        "outcome": "improved",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P005",
        "age": 70,
        "conditions": ["fluid overload", "atrial fibrillation", "diabetes mellitus"],
        "treatments": ["ultrafiltration", "anticoagulation", "rate control", "insulin therapy"],
        "outcome": "stable",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P006",
        "age": 55,
        "conditions": ["fluid overload", "hypertension", "secondary hyperparathyroidism"],
        "treatments": ["ultrafiltration", "ACE inhibitors", "phosphate binders", "vitamin D analogs"],
        "outcome": "stable",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P007",
        "age": 48,
        "conditions": ["hypertension", "anemia", "secondary hyperparathyroidism"],
        "treatments": ["ACE inhibitors", "erythropoietin", "phosphate binders"],
        "outcome": "improved",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P008",
        "age": 61,
        "conditions": ["diabetes mellitus", "peripheral vascular disease", "anemia"],
        "treatments": ["insulin therapy", "antiplatelet therapy", "erythropoietin"],
        "outcome": "stable",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P009",
        "age": 74,
        "conditions": ["fluid overload", "heart failure", "coronary artery disease", "diabetes mellitus"],
        "treatments": ["ultrafiltration", "loop diuretics", "beta blockers", "antiplatelet therapy", "insulin therapy"],
        "outcome": "declined",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P010",
        "age": 53,
        "conditions": ["hypertension", "diabetes mellitus", "peripheral neuropathy"],
        "treatments": ["ARBs", "insulin therapy", "gabapentin"],
        "outcome": "stable",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P011",
        "age": 66,
        "conditions": ["fluid overload", "atrial fibrillation", "heart failure"],
        "treatments": ["ultrafiltration", "anticoagulation", "digoxin", "loop diuretics"],
        "outcome": "improved",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P012",
        "age": 59,
        "conditions": ["anemia", "secondary hyperparathyroidism", "malnutrition"],
        "treatments": ["erythropoietin", "phosphate binders", "nutritional supplementation"],
        "outcome": "improved",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P013",
        "age": 78,
        "conditions": ["fluid overload", "coronary artery disease", "hypertension", "anemia"],
        "treatments": ["ultrafiltration", "beta blockers", "ACE inhibitors", "erythropoietin"],
        "outcome": "stable",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P014",
        "age": 45,
        "conditions": ["hypertension", "lupus nephritis", "anemia"],
        "treatments": ["ACE inhibitors", "immunosuppressants", "erythropoietin"],
        "outcome": "stable",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P015",
        "age": 63,
        "conditions": ["diabetes mellitus", "peripheral vascular disease", "hypertension"],
        "treatments": ["insulin therapy", "antiplatelet therapy", "ARBs"],
        "outcome": "stable",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P016",
        "age": 71,
        "conditions": ["fluid overload", "heart failure", "atrial fibrillation", "anemia"],
        "treatments": ["ultrafiltration", "beta blockers", "anticoagulation", "erythropoietin"],
        "outcome": "improved",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P017",
        "age": 50,
        "conditions": ["secondary hyperparathyroidism", "hypertension", "malnutrition"],
        "treatments": ["cinacalcet", "ACE inhibitors", "nutritional supplementation"],
        "outcome": "improved",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P018",
        "age": 68,
        "conditions": ["fluid overload", "diabetes mellitus", "coronary artery disease"],
        "treatments": ["ultrafiltration", "insulin therapy", "beta blockers", "antiplatelet therapy"],
        "outcome": "stable",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P019",
        "age": 57,
        "conditions": ["anemia", "peripheral neuropathy", "malnutrition"],
        "treatments": ["erythropoietin", "gabapentin", "nutritional supplementation"],
        "outcome": "improved",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P020",
        "age": 76,
        "conditions": ["fluid overload", "atrial fibrillation", "peripheral vascular disease"],
        "treatments": ["ultrafiltration", "anticoagulation", "antiplatelet therapy"],
        "outcome": "declined",
        "prior_cardiac_event": True,
    },
    {
        "patient_id": "P021",
        "age": 62,
        "conditions": ["hypertension", "secondary hyperparathyroidism", "anemia"],
        "treatments": ["calcium channel blockers", "phosphate binders", "erythropoietin"],
        "outcome": "stable",
        "prior_cardiac_event": False,
    },
    {
        "patient_id": "P022",
        "age": 69,
        "conditions": ["fluid overload", "heart failure", "diabetes mellitus", "anemia"],
        "treatments": ["ultrafiltration", "loop diuretics", "insulin therapy", "erythropoietin"],
        "outcome": "improved",
        "prior_cardiac_event": True,
    },
]


def generate_patients(output_path: str = "patients.json") -> list[dict[str, Any]]:
    """Write patient profiles to JSON and return the list."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(PATIENTS, f, indent=2)

    cardiac_fluid = [
        p for p in PATIENTS
        if "fluid overload" in p["conditions"] and p["prior_cardiac_event"]
    ]
    assert len(cardiac_fluid) >= 5, (
        f"Data integrity error: only {len(cardiac_fluid)} patients have "
        "fluid overload + prior cardiac event (need >= 5)"
    )
    return PATIENTS


def load_patients(path: str = "patients.json") -> list[dict[str, Any]]:
    """Load patient profiles from JSON; generate if missing."""
    if not os.path.exists(path):
        generate_patients(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    pts = generate_patients()
    print(f"Generated {len(pts)} patients.")
    n = sum(1 for p in pts if "fluid overload" in p["conditions"] and p["prior_cardiac_event"])
    print(f"Patients with fluid overload + cardiac history: {n}")

