"""
scoring_app.py — KPI tier scoring interface · AutoAlchemy design system.
Run with:  streamlit run app/scoring_app.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_MODEL  = _ROOT / "models" / "model_kpi_a.joblib"
_SCHEMA = _ROOT / "models" / "schema_kpi.json"

import sys
sys.path.insert(0, str(_ROOT))        # repo root — needed for Streamlit Cloud
sys.path.insert(0, str(_ROOT / "src"))

from app.predictor    import load_artifact, predict_one, predict_batch
from app.schema_utils import get_field_specs, validate_and_coerce, get_csv_template

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_BADGE = {
    "At-Risk":        ("tier-badge--crit", "⚠",  "var(--red)"),
    "Developing":     ("tier-badge--warn", "⚡", "var(--amber)"),
    "High Performer": ("tier-badge--ok",   "★",  "var(--green)"),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KPI Performance Predictor · AutoAlchemy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── AutoAlchemy CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Doto:wght@400;500;700;900&display=swap');

:root {
  --orange: #FF6B1A; --orange-2: #E85B10; --orange-ink: #B0470C;
  --orange-soft: rgba(255,107,26,0.12); --orange-tint: rgba(255,107,26,0.07);
  --paper-0: #FBF9F4; --paper-1: #F2EFE8; --paper-2: #E8E3D8; --paper-3: #DED8CB;
  --ink: #1C1A16; --ink-2: #57534A; --ink-3: #8B8678; --line: #DED8CB;
  --green: #5B7A3F; --green-tint: rgba(91,122,63,0.12);
  --amber: #B8801F; --amber-tint: rgba(184,128,31,0.12);
  --red: #C0432C;   --red-tint: rgba(192,67,44,0.12);
  --shadow-sm: 3px 3px 0 0 var(--ink); --shadow: 4px 4px 0 0 var(--ink);
}

/* ── Base ── */
html, body, .stApp {
  font-family: 'Outfit', system-ui, sans-serif !important;
  background-color: var(--paper-0) !important;
}
/* Dot texture */
.stApp::before {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(rgba(28,26,22,0.13) 1px, transparent 1.5px);
  background-size: 6px 6px; opacity: 0.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background-color: var(--paper-1) !important;
}
[data-testid="stSidebar"] {
  border-right: 1.5px solid var(--ink) !important;
}
[data-testid="stSidebar"] * { font-family: 'Outfit', sans-serif !important; }

/* ── Typography ── */
h1 { font-size: 30px !important; font-weight: 700 !important;
  letter-spacing: -0.015em !important; line-height: 1.15 !important;
  color: var(--ink) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1.5px solid var(--ink) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Outfit', sans-serif !important; font-size: 11px !important;
  font-weight: 600 !important; letter-spacing: 0.14em !important;
  text-transform: uppercase !important; color: var(--ink-3) !important;
  background: transparent !important; border: none !important;
  padding: 10px 18px !important;
}
.stTabs [aria-selected="true"][data-baseweb="tab"] { color: var(--orange-ink) !important; }
.stTabs [data-baseweb="tab-highlight"] {
  background-color: var(--orange) !important; height: 2.5px !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
  border: 1.5px solid var(--ink) !important; border-radius: 3px !important;
  background: var(--paper-1) !important; box-shadow: var(--shadow-sm) !important;
  margin-bottom: 8px !important; overflow: hidden !important;
}
[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
  font-family: 'Outfit', sans-serif !important; font-size: 11px !important;
  font-weight: 600 !important; letter-spacing: 0.14em !important;
  text-transform: uppercase !important; color: var(--ink) !important;
  background: var(--paper-1) !important; padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover,
.streamlit-expanderHeader:hover { background: var(--paper-2) !important; }

/* ── Checkboxes ── */
[data-testid="stCheckbox"] {
  border: 1px solid var(--line) !important; border-radius: 2px !important;
  background: var(--paper-0) !important; padding: 6px 10px !important;
  transition: border-color 0.15s, background 0.15s;
}
[data-testid="stCheckbox"]:hover {
  border-color: var(--ink) !important; background: var(--paper-2) !important;
}
[data-testid="stCheckbox"] label {
  font-family: 'Outfit', sans-serif !important; font-size: 12px !important;
  color: var(--ink-2) !important;
}
[data-testid="stCheckbox"] input[type="checkbox"] { accent-color: var(--orange) !important; }

/* ── Inputs ── */
.stNumberInput input, .stTextInput input {
  font-family: 'Outfit', sans-serif !important; font-size: 14px !important;
  border: 1.5px solid var(--ink) !important; border-radius: 2px !important;
  background: var(--paper-1) !important; color: var(--ink) !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
  border-color: var(--orange) !important; box-shadow: none !important;
}
/* Selectbox */
[data-baseweb="select"] > div:first-child {
  font-family: 'Outfit', sans-serif !important; font-size: 14px !important;
  border: 1.5px solid var(--ink) !important; border-radius: 2px !important;
  background: var(--paper-1) !important; color: var(--ink) !important;
}
[data-baseweb="select"] > div:first-child:hover {
  border-color: var(--orange) !important;
}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {
  font-family: 'Outfit', sans-serif !important; font-size: 13px !important;
}

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
  font-family: 'Outfit', sans-serif !important; font-size: 13px !important;
  font-weight: 600 !important; border-radius: 2px !important;
  border: 1.5px solid var(--ink) !important;
  background: var(--paper-0) !important; color: var(--ink) !important;
  box-shadow: var(--shadow-sm) !important;
  transition: transform 0.11s ease, box-shadow 0.11s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translate(-1px,-1px) !important; box-shadow: var(--shadow) !important;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translate(1px,1px) !important; box-shadow: 1px 1px 0 0 var(--ink) !important;
}
/* Primary */
.stButton > button[kind="primary"] {
  background: var(--orange) !important; border-color: var(--orange) !important;
  color: #fff !important;
}
.stButton > button[kind="primary"]:hover { background: var(--orange-2) !important; }
/* Download = primary-style */
.stDownloadButton > button {
  background: var(--paper-0) !important; border-color: var(--ink) !important;
  color: var(--ink) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  border: 1.5px dashed var(--ink) !important; border-radius: 3px !important;
  background: var(--paper-1) !important; padding: 8px !important;
}
[data-testid="stFileUploaderDropzone"] {
  font-family: 'Outfit', sans-serif !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1.5px solid var(--ink) !important; border-radius: 3px !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stDataFrame"] table {
  font-family: 'Outfit', sans-serif !important; font-size: 12px !important;
}

/* ── Alert/info boxes ── */
[data-testid="stAlert"] {
  border-radius: 2px !important; font-family: 'Outfit', sans-serif !important;
  font-size: 13px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { font-family: 'Outfit', sans-serif !important; }

/* ── Scrollbar ── */
* { scrollbar-width: thin; scrollbar-color: var(--ink-3) transparent; }
*::-webkit-scrollbar { width: 8px; }
*::-webkit-scrollbar-thumb {
  background: var(--ink-3); border: 2px solid transparent; background-clip: content-box;
}
*::-webkit-scrollbar-thumb:hover { background: var(--orange); background-clip: content-box; }

/* ── Custom HTML components ── */
.aa-eyebrow {
  font-family: 'Outfit', sans-serif; font-size: 12px; font-weight: 700;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-3);
  margin-bottom: 6px;
}
.aa-page-title {
  font-family: 'Outfit', sans-serif; font-size: 30px; font-weight: 700;
  letter-spacing: -0.015em; line-height: 1.15; color: var(--ink); margin: 0;
}
.aa-kpi-rail {
  display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr;
  gap: 14px; margin-bottom: 20px;
}
.aa-kpi {
  background: var(--paper-1); border: 1.5px solid var(--ink);
  border-radius: 3px; padding: 16px 18px;
  box-shadow: var(--shadow-sm); display: flex; flex-direction: column; gap: 8px;
}
.aa-kpi--hero { background: var(--paper-2); box-shadow: none; border-color: var(--line); }
.aa-kpi__label {
  font-family: 'Outfit', sans-serif; font-size: 10px; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-3);
}
.aa-kpi__value {
  font-family: 'Doto', monospace; font-size: 30px; font-weight: 700;
  line-height: 1.05; color: var(--ink); font-variant-numeric: tabular-nums;
}
.aa-kpi--accent .aa-kpi__value { color: var(--orange-ink); }
.aa-kpi__foot { display: flex; align-items: center; gap: 6px; }
.aa-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: 'Outfit', sans-serif; font-size: 10px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; padding: 3px 8px;
  border: 1px solid var(--ink); border-radius: 2px;
  color: var(--ink-2); white-space: nowrap;
}
.aa-pill--hot { background: var(--orange-soft); color: var(--orange-ink); border-color: var(--orange); }
.aa-pill--ok  { color: var(--green); border-color: var(--green); }
.aa-pill--warn { color: var(--amber); border-color: var(--amber); }
.aa-pill--crit { color: var(--red); border-color: var(--red); }
.aa-dot {
  width: 6px; height: 6px; border-radius: 1px; display: inline-block;
  background: var(--ink-3); flex-shrink: 0;
}
.aa-dot--accent { background: var(--orange); }
.aa-dot--ok     { background: var(--green); }
.aa-dot--warn   { background: var(--amber); }
.aa-dot--crit   { background: var(--red); }

/* Section label (sidebar) */
.aa-slabel {
  font-family: 'Outfit', sans-serif; font-size: 10px; font-weight: 600;
  letter-spacing: 0.20em; text-transform: uppercase; color: var(--ink-3);
  padding: 12px 0 6px; margin-top: 8px; border-top: 1px solid var(--line);
}
.aa-slabel:first-child { border-top: none; margin-top: 0; padding-top: 0; }
.aa-skpi-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 7px 0; border-bottom: 1px solid var(--line);
  font-family: 'Outfit', sans-serif;
}
.aa-skpi-row:last-child { border-bottom: none; }
.aa-skpi-name { font-size: 10px; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.08em; }
.aa-skpi-val { font-family: 'Doto', monospace; font-size: 15px; font-weight: 700; color: var(--orange-ink); }
.aa-cut-row {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 0; font-size: 12px; color: var(--ink-2);
  border-bottom: 1px solid var(--line); font-family: 'Outfit', sans-serif;
}
.aa-cut-row:last-of-type { border-bottom: none; }
.aa-cut-dot { width: 7px; height: 7px; border-radius: 1px; flex-shrink: 0; }
.aa-train-row {
  display: flex; justify-content: space-between;
  padding: 4px 0; font-size: 11px; font-family: 'Outfit', sans-serif;
  border-bottom: 1px solid var(--line);
}
.aa-train-row:last-child { border-bottom: none; }
.aa-train-key { color: var(--ink-3); }
.aa-train-val { font-weight: 600; color: var(--ink); }

/* Result card */
.aa-card {
  background: var(--paper-1); border: 1.5px solid var(--ink);
  border-radius: 3px; padding: 20px; box-shadow: var(--shadow-sm);
  margin-bottom: 14px;
}
.aa-card--flat { background: var(--paper-0); border-color: var(--line); box-shadow: none; }
.aa-card__head {
  display: flex; align-items: center; gap: 8px;
  font-family: 'Outfit', sans-serif; font-size: 10px; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-3);
  margin-bottom: 14px;
}
.aa-card__head::before { content: ""; width: 7px; height: 7px; background: var(--orange); flex: none; }
.result-grid { display: grid; grid-template-columns: 170px 1fr; gap: 20px; align-items: start; }
.tier-badge {
  border: 1.5px solid; border-radius: 3px; padding: 20px 14px; text-align: center;
}
.tier-badge__icon { font-size: 1.8rem; margin-bottom: 8px; }
.tier-badge__name {
  font-family: 'Doto', monospace; font-size: 18px; font-weight: 700;
  line-height: 1.1; margin-bottom: 4px;
}
.tier-badge__sub {
  font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ink-3); font-family: 'Outfit', sans-serif;
}
.tier-badge--crit { background: var(--red-tint); border-color: var(--red); }
.tier-badge--crit .tier-badge__name { color: var(--red); }
.tier-badge--warn { background: var(--amber-tint); border-color: var(--amber); }
.tier-badge--warn .tier-badge__name { color: var(--amber); }
.tier-badge--ok   { background: var(--green-tint); border-color: var(--green); }
.tier-badge--ok   .tier-badge__name { color: var(--green); }
.prob-row { margin-bottom: 12px; }
.prob-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 5px; font-size: 12px; font-family: 'Outfit', sans-serif;
}
.prob-header .name { font-weight: 600; }
.prob-pct { font-family: 'Doto', monospace; font-size: 14px; font-weight: 700; }
.prob-track {
  height: 5px; background: var(--paper-2); border: 1px solid var(--line);
  border-radius: 0; overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 0; }

/* Batch metric row */
.aa-metric-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 16px; }
.aa-mblock {
  background: var(--paper-1); border: 1.5px solid var(--ink);
  border-radius: 3px; padding: 14px 16px; box-shadow: var(--shadow-sm);
}
.aa-mblock__label {
  font-family: 'Outfit', sans-serif; font-size: 10px; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 4px;
}
.aa-mblock__val { font-family: 'Doto', monospace; font-size: 28px; font-weight: 700; line-height: 1; }
.aa-mblock--crit .aa-mblock__val { color: var(--red); }
.aa-mblock--warn .aa-mblock__val { color: var(--amber); }
.aa-mblock--ok   .aa-mblock__val { color: var(--green); }

@media (max-width: 900px) {
  .aa-kpi-rail { grid-template-columns: 1fr 1fr; }
  .result-grid { grid-template-columns: 1fr; }
  .aa-metric-row { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def _load():
    return load_artifact(_MODEL, _SCHEMA)

artifact, schema = _load()
field_specs  = get_field_specs(schema)
tier_labels  = artifact["kpi_tier_labels"]
cutpoints    = artifact["tier_info"]["cutpoints"]
n_labeled    = artifact["tier_info"]["n_labeled"]
pct_labeled  = artifact["tier_info"]["pct_labeled"]
auc_val      = artifact["metrics"].get("auc_ovr_macro") or 0.0
accuracy     = artifact["metrics"]["accuracy"]
baseline     = artifact["majority_baseline_acc"]
lift         = artifact["metrics"]["lift_vs_baseline"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:4px 0 16px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px">
        <div style="font-family:'Outfit',sans-serif;font-size:16px;font-weight:700;color:var(--ink)">
          Auto<span style="font-family:'Doto',monospace;letter-spacing:0.04em">Alchemy</span>
        </div>
        <div class="aa-pill aa-pill--hot">
          <span class="aa-dot aa-dot--accent"></span>Live
        </div>
      </div>

      <div class="aa-slabel" style="border-top:none;margin-top:0;padding-top:0">Model</div>
      <div style="margin-bottom:14px">
        <div class="aa-skpi-row">
          <span class="aa-skpi-name">AUC</span>
          <span class="aa-skpi-val">{auc_val:.3f}</span>
        </div>
        <div class="aa-skpi-row">
          <span class="aa-skpi-name">Accuracy</span>
          <span class="aa-skpi-val">{accuracy:.1%}</span>
        </div>
        <div class="aa-skpi-row">
          <span class="aa-skpi-name">Baseline</span>
          <span class="aa-skpi-val">{baseline:.1%}</span>
        </div>
        <div class="aa-skpi-row" style="border-bottom:none">
          <span class="aa-skpi-name">Lift</span>
          <span class="aa-skpi-val" style="color:var(--green)">+{lift*100:.0f}pp</span>
        </div>
      </div>

      <div class="aa-slabel">KPI Cutpoints</div>
      <div style="margin-bottom:12px">
        <div class="aa-cut-row">
          <div class="aa-cut-dot" style="background:var(--red)"></div>
          <div><strong>At-Risk</strong> — PTG &lt; {cutpoints['p33']:.1f}%</div>
        </div>
        <div class="aa-cut-row">
          <div class="aa-cut-dot" style="background:var(--amber)"></div>
          <div><strong>Developing</strong> — {cutpoints['p33']:.1f}–{cutpoints['p67']:.1f}%</div>
        </div>
        <div class="aa-cut-row" style="border-bottom:none">
          <div class="aa-cut-dot" style="background:var(--green)"></div>
          <div><strong>High Performer</strong> — PTG &gt; {cutpoints['p67']:.1f}%</div>
        </div>
        <div style="margin-top:10px;font-size:11px;color:var(--ink-3);line-height:1.7;font-family:'Outfit',sans-serif">
          {n_labeled:,} labeled agents<br>Tercile-based · each tier ≈ 33%
        </div>
      </div>

      <div class="aa-slabel">Training</div>
      <div>
        <div class="aa-train-row"><span class="aa-train-key">Algorithm</span><span class="aa-train-val">LightGBM</span></div>
        <div class="aa-train-row"><span class="aa-train-key">Features</span><span class="aa-train-val">122 pre-hire</span></div>
        <div class="aa-train-row"><span class="aa-train-key">Classes</span><span class="aa-train-val">3 tiers</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:20px">
  <div class="aa-eyebrow">Concentrix PH · Advisor Performance Intelligence</div>
  <h1 class="aa-page-title">KPI Performance Predictor.</h1>
</div>
""", unsafe_allow_html=True)

# ── KPI Rail ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="aa-kpi-rail">
  <div class="aa-kpi aa-kpi--hero aa-kpi--accent">
    <span class="aa-kpi__label">Model Accuracy</span>
    <span class="aa-kpi__value">{accuracy*100:.1f}<span style="font-size:0.4em;color:var(--ink-3);margin-left:3px">%</span></span>
    <div class="aa-kpi__foot">
      <span class="aa-pill aa-pill--hot"><span class="aa-dot aa-dot--accent"></span>+{lift*100:.0f}pp lift</span>
    </div>
  </div>
  <div class="aa-kpi">
    <span class="aa-kpi__label">Labeled Agents</span>
    <span class="aa-kpi__value">{n_labeled/1000:.1f}<span style="font-size:0.5em;color:var(--ink-3);margin-left:3px">k</span></span>
    <div class="aa-kpi__foot"><span class="aa-pill aa-pill--ok">CDM · {pct_labeled:.1%}</span></div>
  </div>
  <div class="aa-kpi">
    <span class="aa-kpi__label">Features</span>
    <span class="aa-kpi__value">122</span>
    <div class="aa-kpi__foot"><span class="aa-pill">Pre-hire only</span></div>
  </div>
  <div class="aa-kpi">
    <span class="aa-kpi__label">Macro AUC</span>
    <span class="aa-kpi__value">{auc_val:.3f}</span>
    <div class="aa-kpi__foot"><span class="aa-pill aa-pill--ok">3-class</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_form, tab_batch = st.tabs(["Score Candidate", "Batch Upload"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Score Candidate
# ═══════════════════════════════════════════════════════════════════════════════
with tab_form:
    col_form, col_result = st.columns([1.6, 1])

    with col_form:
        groups: dict[str, list[dict]] = {}
        for spec in field_specs:
            groups.setdefault(spec["group"], []).append(spec)

        form_values: dict = {}

        flag_groups = [
            "Leadership Background", "Support Type Experience", "Channel Experience",
            "Certifications & Licences", "Transportation", "Internet Service Provider",
            "Industry Vertical Experience",
        ]
        for group_name in flag_groups:
            if group_name not in groups:
                continue
            with st.expander(group_name, expanded=False):
                cols = st.columns(4)
                for i, spec in enumerate(groups[group_name]):
                    with cols[i % 4]:
                        form_values[spec["name"]] = int(
                            st.checkbox(spec["name"], value=bool(spec["default"]),
                                        key=f"form_{spec['name']}")
                        )

        demo_group = groups.get("Demographics & Org Context", [])
        if demo_group:
            with st.expander("Demographics & Org Context", expanded=True):
                c1, c2 = st.columns(2)
                for i, spec in enumerate(demo_group):
                    with (c1 if i % 2 == 0 else c2):
                        if spec["type"] == "categorical":
                            opts = spec["options"] or []
                            form_values[spec["name"]] = st.selectbox(
                                spec["name"], opts or ["—"],
                                index=0, key=f"form_{spec['name']}"
                            ) if opts else None
                        elif spec["type"] == "numeric":
                            form_values[spec["name"]] = st.number_input(
                                spec["name"],
                                min_value=spec["min"], max_value=spec["max"],
                                value=spec["default"], key=f"form_{spec['name']}"
                            )
                        else:
                            form_values[spec["name"]] = int(
                                st.checkbox(spec["name"], value=False,
                                            key=f"form_{spec['name']}")
                            )

        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            coerced, errors = validate_and_coerce(form_values, schema)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                result   = predict_one(artifact, coerced)
                tier     = result["tier"]
                badge_cls, icon, color = TIER_BADGE[tier]

                prob_html = ""
                for i, lbl in enumerate(tier_labels):
                    p  = result["proba"][i]
                    _, _, c = TIER_BADGE[lbl]
                    prob_html += (
                        f'<div class="prob-row">'
                        f'<div class="prob-header">'
                        f'<span class="name" style="color:{c}">&#9642; {lbl}</span>'
                        f'<span class="prob-pct" style="color:{c}">{p:.1%}</span>'
                        f'</div>'
                        f'<div class="prob-track">'
                        f'<div class="prob-fill" style="width:{p*100:.1f}%;background:{c}"></div>'
                        f'</div>'
                        f'</div>'
                    )

                prob_label = '<div style="font-family:\'Outfit\',sans-serif;font-size:10px;font-weight:600;letter-spacing:0.16em;text-transform:uppercase;color:var(--ink-3);margin-bottom:12px">Probability Breakdown</div>'
                st.markdown(
                    f'<div class="aa-card">'
                    f'<div class="aa-card__head">Prediction Output</div>'
                    f'<div class="result-grid">'
                    f'<div class="tier-badge {badge_cls}">'
                    f'<div class="tier-badge__icon">{icon}</div>'
                    f'<div class="tier-badge__name">{tier}</div>'
                    f'<div class="tier-badge__sub">Predicted KPI Tier</div>'
                    f'</div>'
                    f'<div>{prob_label}{prob_html}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("""
            <div class="aa-card aa-card--flat" style="text-align:center;padding:44px 20px">
              <div style="font-family:'Doto',monospace;font-size:36px;color:var(--orange-ink);
                margin-bottom:12px">?</div>
              <div style="font-family:'Outfit',sans-serif;font-size:10px;font-weight:600;
                letter-spacing:0.16em;text-transform:uppercase;color:var(--ink-3);
                margin-bottom:8px">No prediction yet</div>
              <div style="font-size:13px;color:var(--ink-3);font-family:'Outfit',sans-serif;
                line-height:1.6">Fill in the candidate profile<br>and click Run Prediction.</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Upload
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("""
    <div style="margin-bottom:18px">
      <div class="aa-eyebrow" style="margin-bottom:4px">Batch Scoring</div>
      <p style="font-size:13px;color:var(--ink-2);margin:0;font-family:'Outfit',sans-serif">
        Upload a filled CSV to score multiple candidates at once.
      </p>
    </div>
    """, unsafe_allow_html=True)

    tmpl = get_csv_template(schema)
    st.download_button(
        "⬇ Download CSV Template",
        data=tmpl.to_csv(index=False).encode("utf-8"),
        file_name="kpi_scoring_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload filled CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded, dtype=str)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_raw = None

        if df_raw is not None:
            st.markdown(
                f'<p style="font-size:13px;color:var(--ink-2);font-family:\'Outfit\',sans-serif;'
                f'margin-bottom:8px"><strong style="color:var(--ink)">{len(df_raw):,} candidates</strong>'
                f' detected · preview:</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(df_raw.head(5), use_container_width=True)

            score_btn = st.button("Score All Candidates", type="primary")
            if score_btn:
                with st.spinner("Scoring…"):
                    coerced_rows = []
                    for _, row in df_raw.iterrows():
                        coerced, _ = validate_and_coerce(row.to_dict(), schema)
                        coerced_rows.append(coerced)
                    df_results = predict_batch(artifact, pd.DataFrame(coerced_rows))

                tier_counts = df_results["predicted_tier"].value_counts()
                at_n  = tier_counts.get("At-Risk", 0)
                dev_n = tier_counts.get("Developing", 0)
                hp_n  = tier_counts.get("High Performer", 0)

                st.markdown(f"""
                <div class="aa-metric-row">
                  <div class="aa-mblock aa-mblock--crit">
                    <div class="aa-mblock__label">▪ At-Risk</div>
                    <div class="aa-mblock__val">{at_n}</div>
                  </div>
                  <div class="aa-mblock aa-mblock--warn">
                    <div class="aa-mblock__label">▪ Developing</div>
                    <div class="aa-mblock__val">{dev_n}</div>
                  </div>
                  <div class="aa-mblock aa-mblock--ok">
                    <div class="aa-mblock__label">▪ High Performer</div>
                    <div class="aa-mblock__val">{hp_n}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                display_cols = ["predicted_tier", "p_at_risk", "p_developing", "p_high_performer"]
                df_show = (
                    df_results[display_cols]
                    .sort_values("p_at_risk", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(
                    df_show.style.format({
                        "p_at_risk":        "{:.1%}",
                        "p_developing":     "{:.1%}",
                        "p_high_performer": "{:.1%}",
                    }),
                    use_container_width=True,
                    height=400,
                )
                st.download_button(
                    "⬇ Download Scored Results",
                    data=df_results.to_csv(index=False).encode("utf-8"),
                    file_name="kpi_scored_results.csv",
                    mime="text/csv",
                )
