"""
streamlit_app.py
Streamlit demonstration UI for the GraphRAG Dialysis Patient Search system.
Features: file upload (JSON patient data), custom query, graph stats,
          reasoning path visualization, RAGAS evaluation display.

Run with: streamlit run streamlit_app.py
"""

import json
import tempfile
import os
import sys
import time
from typing import Any

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG · Dialysis Patient Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.card-accent {
    background: #0d1117;
    border: 1px solid #30363d;
    border-left: 3px solid #3fb950;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #7ee787;
}

.card-warn {
    border-left: 3px solid #d29922;
    color: #e3b341;
}

.card-info {
    border-left: 3px solid #388bfd;
    color: #79c0ff;
}

/* Header */
.hero {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
}

.hero h1 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: #e6edf3;
    margin: 0;
    letter-spacing: -0.5px;
}

.hero p {
    color: #7d8590;
    font-size: 0.9rem;
    margin: 0.3rem 0 0 0;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 6px;
}

.badge-green  { background: #1a4a1a; color: #3fb950; border: 1px solid #2ea043; }
.badge-yellow { background: #3d2e00; color: #e3b341; border: 1px solid #9e6a03; }
.badge-blue   { background: #0c2a4a; color: #79c0ff; border: 1px solid #1f6feb; }
.badge-purple { background: #2b1a4a; color: #d2a8ff; border: 1px solid #6e40c9; }
.badge-red    { background: #3d0c0c; color: #ff7b72; border: 1px solid #da3633; }

/* Stat tiles */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.stat-tile {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #3fb950;
    line-height: 1;
}

.stat-label {
    font-size: 0.72rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* Confidence bar */
.conf-bar-bg {
    background: #21262d;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}

.conf-bar-fill {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #238636, #3fb950);
    transition: width 0.6s ease;
}

/* Traversal path chips */
.path-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
}

.chip {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px 8px;
    color: #c9d1d9;
}

.chip-patient  { border-color: #388bfd; color: #79c0ff; background: #0c2a4a; }
.chip-tx       { border-color: #2ea043; color: #7ee787; background: #0d2818; }
.chip-outcome  { border-color: #6e40c9; color: #d2a8ff; background: #1e1133; }
.chip-arrow    { color: #484f58; background: none; border: none; padding: 0 2px; }

/* Metric gauges */
.metric-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    gap: 10px;
}

.metric-label {
    font-size: 0.8rem;
    color: #8b949e;
    width: 160px;
    flex-shrink: 0;
}

.metric-bar-bg {
    flex: 1;
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}

.metric-bar-fill {
    height: 6px;
    border-radius: 4px;
}

.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #e6edf3;
    width: 40px;
    text-align: right;
}

/* Step list */
.step-item {
    display: flex;
    gap: 12px;
    margin-bottom: 10px;
    align-items: flex-start;
}

.step-num {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #3fb950;
    flex-shrink: 0;
    margin-top: 1px;
}

.step-text {
    font-size: 0.85rem;
    color: #8b949e;
    font-family: 'DM Mono', monospace;
    line-height: 1.5;
}

/* Treatment table */
.tx-row {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid #21262d;
    gap: 12px;
    font-size: 0.85rem;
}

.tx-row:last-child { border-bottom: none; }
.tx-rank { color: #484f58; font-family: 'DM Mono', monospace; width: 20px; }
.tx-name { flex: 1; color: #e6edf3; }
.tx-score { font-family: 'DM Mono', monospace; color: #3fb950; width: 50px; text-align: right; }

/* Streamlit element overrides */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

div[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background: #2ea043 !important;
}

div[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}

.stSpinner > div { color: #3fb950 !important; }

/* Section headers */
.sec-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #484f58;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chip(text: str, kind: str = "") -> str:
    css = f"chip chip-{kind}" if kind else "chip"
    return f'<span class="{css}">{text}</span>'


def _arrow() -> str:
    return '<span class="chip chip-arrow">→</span>'


def _badge(text: str, color: str = "green") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


def _metric_bar(label: str, value: float, color: str = "#3fb950") -> str:
    pct = int(value * 100)
    return f"""
    <div class="metric-row">
        <div class="metric-label">{label}</div>
        <div class="metric-bar-bg">
            <div class="metric-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
        <div class="metric-val">{value:.2f}</div>
    </div>"""


def _path_html(path: list[str]) -> str:
    """Render a traversal path as colored chips."""
    parts = []
    for i, node in enumerate(path):
        if node.startswith("tx:"):
            parts.append(_chip(node.replace("tx:", ""), "tx"))
        elif node.startswith("outcome:"):
            parts.append(_chip(node.replace("outcome:", ""), "outcome"))
        elif node.startswith("cond:"):
            parts.append(_chip(node.replace("cond:", ""), ""))
        else:
            parts.append(_chip(node, "patient"))
        if i < len(path) - 1:
            parts.append(_arrow())
    return f'<div class="path-row">{"".join(parts)}</div>'


# ── Import pipeline modules ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _import_modules():
    """Import pipeline modules once and cache."""
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import load_patients, generate_patients, PATIENTS
    from entity_extractor import extract_all
    from graph_engine import build_graph
    from similarity_engine import add_similarity_edges
    from query_engine import parse_query
    from reasoner import reason
    from evaluator import evaluate
    from ollama_client import is_ollama_available
    return {
        "load_patients": load_patients,
        "generate_patients": generate_patients,
        "PATIENTS": PATIENTS,
        "extract_all": extract_all,
        "build_graph": build_graph,
        "add_similarity_edges": add_similarity_edges,
        "parse_query": parse_query,
        "reason": reason,
        "evaluate": evaluate,
        "is_ollama_available": is_ollama_available,
    }


try:
    mods = _import_modules()
except Exception as e:
    st.error(f"Failed to import pipeline modules: {e}")
    st.info("Make sure streamlit_app.py is in the same folder as the GraphRAG modules.")
    st.stop()


# ── Session state defaults ────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None
if "graph_stats" not in st.session_state:
    st.session_state.graph_stats = None
if "matched_entities" not in st.session_state:
    st.session_state.matched_entities = []
if "custom_patients" not in st.session_state:
    st.session_state.custom_patients = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ── Hero header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🧬 GraphRAG · Dialysis Patient Search</h1>
  <p>Multi-hop knowledge graph reasoning over dialysis patient data · Powered by Ollama llama3</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sec-header">System Status</div>', unsafe_allow_html=True)

    ollama_ok = mods["is_ollama_available"]()
    if ollama_ok:
        st.markdown(_badge("Ollama Online", "green") + " LLM active", unsafe_allow_html=True)
    else:
        st.markdown(_badge("Ollama Offline", "yellow") + " Rule-based fallback", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec-header">Patient Data Source</div>', unsafe_allow_html=True)

    data_source = st.radio(
        "Source",
        ["Built-in (22 patients)", "Upload JSON"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    if data_source == "Upload JSON":
        uploaded_file = st.file_uploader(
            "Upload patients JSON",
            type=["json"],
            help="JSON array of patient objects. Each must have: patient_id, age, conditions (list), treatments (list), outcome, prior_cardiac_event (bool).",
        )

        with st.expander("📋 Required JSON schema"):
            st.code("""[
  {
    "patient_id": "P001",
    "age": 67,
    "conditions": ["fluid overload", "hypertension"],
    "treatments": ["ultrafiltration", "ACE inhibitors"],
    "outcome": "improved",
    "prior_cardiac_event": true
  }
]""", language="json")

    st.markdown("---")
    st.markdown('<div class="sec-header">Query Examples</div>', unsafe_allow_html=True)

    example_queries = [
        "What treatments worked for patients with fluid overload who also had prior cardiac events?",
        "Which treatments helped diabetic patients with anemia?",
        "What outcomes did patients with heart failure receive from beta blockers?",
        "Best treatments for hypertensive patients with secondary hyperparathyroidism?",
    ]

    selected_example = st.selectbox(
        "Load an example",
        ["(custom query)"] + example_queries,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="sec-header">Reasoning Settings</div>', unsafe_allow_html=True)
    top_k_similar = st.slider("Similar patients (top-k)", 1, 8, 3)
    min_similarity = st.slider("Min similarity threshold", 0.0, 0.5, 0.15, 0.05)
    show_all_paths = st.checkbox("Show all traversal paths", value=False)


# ── Main layout ───────────────────────────────────────────────────────────────

col_query, col_results = st.columns([1, 1.4], gap="large")

with col_query:
    st.markdown('<div class="sec-header">Query</div>', unsafe_allow_html=True)

    default_query = (
        selected_example
        if selected_example != "(custom query)"
        else "What treatments worked for patients with fluid overload who also had prior cardiac events?"
    )

    query_text = st.text_area(
        "Natural language query",
        value=default_query,
        height=100,
        label_visibility="collapsed",
        placeholder="Ask a clinical question about dialysis patients...",
    )

    run_btn = st.button("⚡ Run GraphRAG Query", use_container_width=True)

    # File upload handling
    if uploaded_file is not None:
        try:
            raw = json.load(uploaded_file)
            if not isinstance(raw, list):
                st.error("JSON must be a top-level array of patient objects.")
            else:
                required_keys = {"patient_id", "conditions", "treatments", "outcome", "prior_cardiac_event"}
                missing = [
                    p.get("patient_id", f"index {i}")
                    for i, p in enumerate(raw)
                    if not required_keys.issubset(p.keys())
                ]
                if missing:
                    st.warning(f"Some records missing required keys: {missing[:3]}")
                else:
                    st.session_state.custom_patients = raw
                    st.markdown(
                        f'<div class="card-accent">'
                        f'✓ Loaded {len(raw)} patients from upload — '
                        f'{sum(1 for p in raw if "fluid overload" in p.get("conditions",[]) and p.get("prior_cardiac_event"))} with fluid overload + cardiac history'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
    elif data_source == "Built-in (22 patients)":
        st.session_state.custom_patients = None

    # Data preview
    with st.expander("👁 Preview patient data"):
        if st.session_state.custom_patients:
            patients_preview = st.session_state.custom_patients
        else:
            patients_preview = mods["PATIENTS"]

        st.dataframe(
            [
                {
                    "ID": p["patient_id"],
                    "Age": p.get("age", "?"),
                    "Conditions": ", ".join(p.get("conditions", [])),
                    "Treatments": ", ".join(p.get("treatments", [])),
                    "Outcome": p.get("outcome", "?"),
                    "Cardiac Hx": "✓" if p.get("prior_cardiac_event") else "✗",
                }
                for p in patients_preview
            ],
            use_container_width=True,
            hide_index=True,
        )


# ── Run pipeline ──────────────────────────────────────────────────────────────

if run_btn and query_text.strip():
    with st.spinner("Running multi-hop reasoning..."):
        t0 = time.time()

        # Load patients
        if st.session_state.custom_patients:
            patients = st.session_state.custom_patients
        else:
            try:
                patients = mods["load_patients"]("patients.json")
            except Exception:
                mods["generate_patients"]("patients.json")
                patients = mods["load_patients"]("patients.json")

        # Extract entities
        entities = mods["extract_all"](patients)

        # Build graph
        graph = mods["build_graph"](entities)
        mods["add_similarity_edges"](graph, entities, top_k=top_k_similar, min_similarity=min_similarity)
        stats = graph.stats()
        st.session_state.graph_stats = stats

        # Parse + reason
        sq = mods["parse_query"](query_text)
        result = mods["reason"](sq, entities, graph)
        eval_result = mods["evaluate"](query_text, result, entities)

        st.session_state.result = result
        st.session_state.eval_result = eval_result
        st.session_state.matched_entities = result.matched_entities
        st.session_state.last_query = query_text
        st.session_state.elapsed = round(time.time() - t0, 2)


# ── Results panel ─────────────────────────────────────────────────────────────

with col_results:
    result = st.session_state.result
    eval_result = st.session_state.eval_result
    stats = st.session_state.graph_stats

    if result is None:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 1.5rem;color:#484f58;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🔬</div>
            <div style="font-size:0.9rem;">Run a query to see multi-hop reasoning results</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Graph stat tiles
        if stats:
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-tile">
                    <div class="stat-num">{stats.get('nodes', 0)}</div>
                    <div class="stat-label">Nodes</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-num">{stats.get('edges', 0)}</div>
                    <div class="stat-label">Edges</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-num">{stats.get('patient', 0)}</div>
                    <div class="stat-label">Patients</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-num">{len(result.matched_entities)}</div>
                    <div class="stat-label">Matched</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Status badges
        badges = []
        if result.fallback_used:
            badges.append(_badge("Fallback Used", "yellow"))
        else:
            badges.append(_badge("Strict Match", "green"))
        badges.append(_badge(f"{len(result.traversal_paths)} Paths", "blue"))
        badges.append(_badge(f"{result.confidence:.0%} Confidence", "purple"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Answer
        st.markdown('<div class="sec-header">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{result.answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        # Confidence bar
        conf = result.confidence
        conf_color = "#3fb950" if conf >= 0.7 else "#e3b341" if conf >= 0.4 else "#f85149"
        st.markdown(f"""
        <div style="margin-bottom:1.25rem;">
            <div style="font-size:0.8rem;color:#8b949e;margin-bottom:4px;">Confidence</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{int(conf*100)}%;background:{conf_color};"></div>
            </div>
            <div style="font-size:0.75rem;color:{conf_color};margin-top:4px;font-family:'DM Mono',monospace;">{conf:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for details
        tab_reasoning, tab_paths, tab_eval, tab_raw = st.tabs(
            ["🧠 Reasoning", "🗺 Paths", "📊 Evaluation", "📄 Raw JSON"]
        )

        with tab_reasoning:
            st.markdown('<div class="sec-header" style="margin-top:0.5rem">Reasoning Steps</div>', unsafe_allow_html=True)
            for i, step in enumerate(result.reasoning_steps, 1):
                st.markdown(f"""
                <div class="step-item">
                    <div class="step-num">{i}</div>
                    <div class="step-text">{step}</div>
                </div>""", unsafe_allow_html=True)

            if result.treatment_outcomes:
                st.markdown('<div class="sec-header" style="margin-top:1rem">Treatment Rankings</div>', unsafe_allow_html=True)
                WEIGHTS = {"improved": 2.0, "stable": 1.0, "declined": -1.0}
                ranked = sorted(
                    result.treatment_outcomes.items(),
                    key=lambda x: sum(WEIGHTS.get(o, 0) * c for o, c in x[1].items()) / max(sum(x[1].values()), 1),
                    reverse=True,
                )
                st.markdown('<div class="card" style="padding:0.5rem 0;">', unsafe_allow_html=True)
                for i, (tx, outcomes) in enumerate(ranked[:8], 1):
                    total = max(sum(outcomes.values()), 1)
                    score = sum(WEIGHTS.get(o, 0) * c for o, c in outcomes.items()) / total
                    outcome_str = " · ".join(
                        f'<span style="color:{"#3fb950" if o=="improved" else "#e3b341" if o=="stable" else "#f85149"}">{o}: {c}</span>'
                        for o, c in sorted(outcomes.items(), key=lambda x: x[1], reverse=True)
                    )
                    st.markdown(f"""
                    <div class="tx-row">
                        <div class="tx-rank">#{i}</div>
                        <div class="tx-name">{tx}</div>
                        <div style="flex:1;font-size:0.78rem;color:#8b949e;">{outcome_str}</div>
                        <div class="tx-score">{score:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if result.fallback_used and result.fallback_explanation:
                st.markdown(f'<div class="card-accent card-warn">⚠ {result.fallback_explanation}</div>', unsafe_allow_html=True)

            if result.inferred_links:
                with st.expander(f"🔗 Inferred links ({len(result.inferred_links)})"):
                    for link in result.inferred_links:
                        st.markdown(f'<div class="card-accent">{link}</div>', unsafe_allow_html=True)

        with tab_paths:
            paths = result.traversal_paths
            display_paths = paths if show_all_paths else paths[:12]
            st.markdown(
                f'<div style="font-size:0.8rem;color:#8b949e;margin-bottom:10px;">'
                f'Showing {len(display_paths)} of {len(paths)} traversal paths</div>',
                unsafe_allow_html=True,
            )
            for path in display_paths:
                st.markdown(_path_html(path), unsafe_allow_html=True)

            if not show_all_paths and len(paths) > 12:
                st.caption(f"Enable 'Show all traversal paths' in sidebar to see all {len(paths)} paths.")

            st.markdown('<div class="sec-header" style="margin-top:1rem">Matched Patients</div>', unsafe_allow_html=True)
            chips = " ".join(_chip(pid, "patient") for pid in result.matched_entities)
            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{chips}</div>', unsafe_allow_html=True)

        with tab_eval:
            if eval_result:
                st.markdown('<div class="sec-header" style="margin-top:0.5rem">RAGAS-Style Metrics</div>', unsafe_allow_html=True)
                overall_color = "#3fb950" if eval_result.overall >= 0.7 else "#e3b341" if eval_result.overall >= 0.4 else "#f85149"
                st.markdown(
                    _metric_bar("Answer Relevance", eval_result.answer_relevance, "#79c0ff") +
                    _metric_bar("Faithfulness", eval_result.faithfulness, "#3fb950") +
                    _metric_bar("Context Precision", eval_result.context_precision, "#d2a8ff") +
                    _metric_bar(f"Overall", eval_result.overall, overall_color),
                    unsafe_allow_html=True,
                )
                if eval_result.notes:
                    st.markdown('<div class="sec-header" style="margin-top:1rem">Evaluator Notes</div>', unsafe_allow_html=True)
                    for note in eval_result.notes:
                        st.markdown(f'<div class="card-accent card-info">ℹ {note}</div>', unsafe_allow_html=True)

        with tab_raw:
            raw_output = {
                "query": st.session_state.last_query,
                "answer": result.answer,
                "confidence": result.confidence,
                "fallback_used": result.fallback_used,
                "fallback_explanation": result.fallback_explanation,
                "matched_entities": result.matched_entities,
                "reasoning_steps": result.reasoning_steps,
                "traversal_paths": result.traversal_paths[:20],
                "inferred_links": result.inferred_links,
                "final_decision": result.final_decision,
                "treatment_outcomes": result.treatment_outcomes,
                "evaluation": {
                    "answer_relevance": eval_result.answer_relevance if eval_result else None,
                    "faithfulness": eval_result.faithfulness if eval_result else None,
                    "context_precision": eval_result.context_precision if eval_result else None,
                    "overall": eval_result.overall if eval_result else None,
                } if eval_result else {},
                "graph_stats": stats,
            }
            st.download_button(
                "⬇ Download result JSON",
                data=json.dumps(raw_output, indent=2),
                file_name="graphrag_result.json",
                mime="application/json",
            )
            st.code(json.dumps(raw_output, indent=2), language="json")

        elapsed = st.session_state.get("elapsed", 0)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#484f58;margin-top:0.5rem;font-family:\'DM Mono\',monospace;">'
            f'⏱ completed in {elapsed}s</div>',
            unsafe_allow_html=True,
        )