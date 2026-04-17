"""
app.py
CLI entry point for the GraphRAG Dialysis Patient Search system.

Usage:
    python app.py                  # runs default query
    python app.py "custom query"   # runs custom query
    python app.py --stats          # print graph statistics only
"""

from __future__ import annotations

import sys
import time
from typing import Any

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

from data_generator import generate_patients, load_patients
from entity_extractor import extract_all
from evaluator import evaluate
from graph_engine import build_graph
from ollama_client import is_ollama_available
from query_engine import parse_query
from reasoner import reason
from similarity_engine import add_similarity_edges

DEFAULT_QUERY = (
    "What treatments worked for patients with fluid overload "
    "who also had prior cardiac events?"
)

DIVIDER = "─" * 70


def _color(text: str, color_code: str) -> str:
    if HAS_COLOR:
        return f"{color_code}{text}{Style.RESET_ALL}"
    return text


def _header(text: str) -> str:
    return _color(f"\n{DIVIDER}\n  {text}\n{DIVIDER}", Fore.CYAN if HAS_COLOR else "")


def _ok(text: str) -> str:
    return _color(f"  ✓ {text}", Fore.GREEN if HAS_COLOR else "")


def _warn(text: str) -> str:
    return _color(f"  ⚠ {text}", Fore.YELLOW if HAS_COLOR else "")


def _err(text: str) -> str:
    return _color(f"  ✗ {text}", Fore.RED if HAS_COLOR else "")


def _bold(text: str) -> str:
    return _color(text, Style.BRIGHT if HAS_COLOR else "")


def run_pipeline(query: str, verbose: bool = True) -> dict[str, Any]:
    """
    Execute the full GraphRAG pipeline for a given query.
    Returns a structured output dict regardless of failures.
    """
    t0 = time.time()
    output: dict[str, Any] = {
        "query": query,
        "answer": "",
        "reasoning_steps": [],
        "traversal_paths": [],
        "matched_entities": [],
        "inferred_links": [],
        "final_decision": "",
        "confidence": 0.0,
        "fallback_used": False,
        "fallback_explanation": "",
        "evaluation": {},
        "graph_stats": {},
        "errors": [],
    }

    # ── Step 0: Check Ollama ───────────────────────────────────────────────
    if verbose:
        print(_header("GraphRAG Dialysis Patient Search"))
        llm_status = is_ollama_available()
        if llm_status:
            print(_ok("Ollama is available — LLM extraction active"))
        else:
            print(_warn("Ollama unavailable — using rule-based fallback throughout"))

    # ── Step 1: Load/generate patients ────────────────────────────────────
    try:
        patients = load_patients("patients.json")
        if verbose:
            print(_ok(f"Loaded {len(patients)} patient profiles"))
    except Exception as e:
        output["errors"].append(f"Patient load failed: {e}")
        try:
            patients = generate_patients("patients.json")
            if verbose:
                print(_warn(f"Regenerated {len(patients)} patients after load failure"))
        except Exception as e2:
            output["errors"].append(f"Patient generation also failed: {e2}")
            if verbose:
                print(_err("Cannot proceed without patient data"))
            return output

    # ── Step 2: Entity extraction ─────────────────────────────────────────
    try:
        entities = extract_all(patients)
        methods = set(e.get("extraction_method", "unknown") for e in entities)
        if verbose:
            print(_ok(f"Extracted entities from {len(entities)} patients (methods: {methods})"))
    except Exception as e:
        output["errors"].append(f"Entity extraction failed: {e}")
        # Fallback: use raw patient dicts as entities
        entities = [
            {
                "patient_id": p["patient_id"],
                "age": p.get("age"),
                "conditions": [c.lower() for c in p.get("conditions", [])],
                "treatments": p.get("treatments", []),
                "outcome": p.get("outcome", "unknown"),
                "prior_cardiac_event": p.get("prior_cardiac_event", False),
                "risk_factors": [],
                "extraction_method": "raw_fallback",
            }
            for p in patients
        ]
        if verbose:
            print(_warn("Used raw patient data as entity fallback"))

    # ── Step 3: Build knowledge graph ─────────────────────────────────────
    try:
        graph = build_graph(entities)
        add_similarity_edges(graph, entities, top_k=3, min_similarity=0.15)
        stats = graph.stats()
        output["graph_stats"] = stats
        if verbose:
            print(_ok(
                f"Knowledge graph built: {stats['nodes']} nodes, "
                f"{stats['edges']} edges"
            ))
    except Exception as e:
        output["errors"].append(f"Graph construction failed: {e}")
        if verbose:
            print(_err(f"Graph build failed: {e}"))
        return output

    # ── Step 4: Parse query ───────────────────────────────────────────────
    try:
        sq = parse_query(query)
        if verbose:
            print(_ok(
                f"Query parsed: conditions={sq.conditions_required}, "
                f"cardiac_required={sq.require_cardiac_history}"
            ))
    except Exception as e:
        output["errors"].append(f"Query parsing failed: {e}")
        # Minimal fallback query
        from query_engine import StructuredQuery
        sq = StructuredQuery(
            original=query,
            conditions_required=["fluid overload"],
            require_cardiac_history=True,
        )
        if verbose:
            print(_warn("Using minimal fallback query parameters"))

    # ── Step 5: Multi-hop reasoning ───────────────────────────────────────
    try:
        result = reason(sq, entities, graph)
        output.update({
            "answer": result.answer,
            "reasoning_steps": result.reasoning_steps,
            "traversal_paths": result.traversal_paths[:20],  # cap for display
            "matched_entities": result.matched_entities,
            "inferred_links": result.inferred_links[:10],
            "final_decision": result.final_decision,
            "confidence": result.confidence,
            "fallback_used": result.fallback_used,
            "fallback_explanation": result.fallback_explanation,
        })
        if verbose:
            print(_ok(
                f"Reasoning complete: {len(result.matched_entities)} matched, "
                f"confidence={result.confidence:.0%}"
            ))
    except Exception as e:
        output["errors"].append(f"Reasoning failed: {e}")
        output["answer"] = f"Reasoning error: {e}"
        if verbose:
            print(_err(f"Reasoning failed: {e}"))
        return output

    # ── Step 6: Evaluation ────────────────────────────────────────────────
    try:
        eval_result = evaluate(query, result, entities)
        output["evaluation"] = {
            "answer_relevance": eval_result.answer_relevance,
            "faithfulness": eval_result.faithfulness,
            "context_precision": eval_result.context_precision,
            "overall": eval_result.overall,
            "notes": eval_result.notes,
        }
        if verbose:
            print(_ok(f"Evaluation complete: overall score={eval_result.overall:.2f}"))
    except Exception as e:
        output["errors"].append(f"Evaluation failed: {e}")

    output["elapsed_seconds"] = round(time.time() - t0, 2)
    return output


def print_output(output: dict[str, Any]) -> None:
    """Pretty-print the pipeline output to stdout."""

    print(_header("QUERY"))
    print(f"  {output['query']}")

    print(_header("ANSWER"))
    for line in output["answer"].splitlines():
        print(f"  {line}")

    print(_header("REASONING STEPS"))
    for step in output["reasoning_steps"]:
        print(f"  {step}")

    print(_header("MATCHED ENTITIES"))
    entities_str = ", ".join(output["matched_entities"]) or "(none)"
    print(f"  {entities_str}")

    if output["inferred_links"]:
        print(_header("INFERRED LINKS"))
        for link in output["inferred_links"][:8]:
            print(f"  {link}")

    print(_header("TRAVERSAL PATHS (sample)"))
    paths = output["traversal_paths"]
    for path in paths[:6]:
        print(f"  {' → '.join(path)}")
    if len(paths) > 6:
        print(f"  ... and {len(paths) - 6} more paths")

    print(_header("CONFIDENCE"))
    conf = output["confidence"]
    bar_len = int(conf * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  [{bar}] {conf:.0%}")

    if output["fallback_used"]:
        print(_header("FALLBACK EXPLANATION"))
        print(f"  {output['fallback_explanation']}")

    if output.get("evaluation"):
        print(_header("RAGAS-STYLE EVALUATION"))
        ev = output["evaluation"]
        print(f"  Answer relevance : {ev['answer_relevance']:.3f}")
        print(f"  Faithfulness     : {ev['faithfulness']:.3f}")
        print(f"  Context precision: {ev['context_precision']:.3f}")
        overall_str = f"{ev['overall']:.3f}"
        print(f"  Overall          : {_bold(overall_str)}")
        for note in ev.get("notes", []):
            print(f"    • {note}")

    if output.get("graph_stats"):
        print(_header("GRAPH STATISTICS"))
        for k, v in output["graph_stats"].items():
            print(f"  {k}: {v}")

    if output.get("errors"):
        print(_header("ERRORS / WARNINGS"))
        for err in output["errors"]:
            print(_warn(f"  {err}"))

    elapsed = output.get("elapsed_seconds", 0)
    print(_header("COMPLETE"))
    print(f"  Total time: {elapsed:.2f}s\n")


def main() -> None:
    args = sys.argv[1:]

    if "--stats" in args:
        # Print graph stats only
        patients = load_patients("patients.json")
        entities = extract_all(patients)
        graph = build_graph(entities)
        add_similarity_edges(graph, entities)
        stats = graph.stats()
        print("Graph statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args and not args[0].startswith("--"):
        query = " ".join(args)
    else:
        query = DEFAULT_QUERY

    # Ensure patient data exists
    try:
        generate_patients("patients.json")
    except Exception:
        pass

    output = run_pipeline(query, verbose=True)
    print_output(output)

    if output["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

