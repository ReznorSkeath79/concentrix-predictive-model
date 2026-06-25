# Phase 2 — KPI Scoring App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app that loads `model_kpi_a.joblib` + `schema_kpi.json`, accepts pre-hire candidate data (form or CSV), and returns a predicted KPI tier (At-Risk / Developing / High Performer) with probability scores.

**Architecture:** Three focused Python files under `app/`: `predictor.py` owns model loading and inference (no UI dependency), `schema_utils.py` owns schema parsing and input validation, and `scoring_app.py` wires them together into a Streamlit UI with two tabs — single-candidate form and CSV batch upload. The inference path mirrors what `train_kpi_model.py` does at evaluation time.

**Tech Stack:** Streamlit, pandas, numpy, joblib, LightGBM (already installed). Pure Python — no frontend framework, no API server.

## Global Constraints

- Python 3.12
- Working directory for all commands: `E:\work\concentrix\PredictiveModel`
- `app/` lives inside the project root (same repo)
- `predictor.py` must work without Streamlit installed (pure inference logic)
- All imports from `src/` use `sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))`
- Model artifact: `models/model_kpi_a.joblib`; schema: `models/schema_kpi.json`
- Tier labels: `["At-Risk", "Developing", "High Performer"]` (from artifact, not hardcoded)
- Probability columns in batch output: `p_at_risk`, `p_developing`, `p_high_performer`
- Streamlit version: latest (`pip install streamlit`) — add to `requirements.txt`
- No modification to any existing `src/` files

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `app/__init__.py` | Empty (makes `app` a package) |
| Create | `app/predictor.py` | Load artifact; `predict_one()`, `predict_batch()` |
| Create | `app/schema_utils.py` | Parse schema; `get_field_specs()`; `validate_and_coerce()` |
| Create | `app/scoring_app.py` | Streamlit UI — form tab + CSV tab |
| Modify | `requirements.txt` | Add `streamlit>=1.35.0` |

---

## Task 1: app/predictor.py — Model loading and inference

**Files:**
- Create: `app/__init__.py`
- Create: `app/predictor.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `load_artifact(model_path, schema_path) -> tuple[dict, dict]`
- Produces: `predict_one(artifact, input_dict) -> dict` with keys `tier_idx`, `tier`, `proba`, `labels`
- Produces: `predict_batch(artifact, df) -> pd.DataFrame` with columns `predicted_tier`, `p_at_risk`, `p_developing`, `p_high_performer` appended

- [ ] **Step 1: Add streamlit to requirements.txt**

Append one line to `requirements.txt`:
```
streamlit>=1.35.0
```

Then install it:
```bash
pip install streamlit
```

Expected: `Successfully installed streamlit-...` (or "already satisfied")

- [ ] **Step 2: Create `app/__init__.py`**

Create an empty file at `app/__init__.py`. No content needed.

- [ ] **Step 3: Create `app/predictor.py`**

```python
"""
predictor.py — Model loading and KPI tier inference.
No Streamlit dependency — pure Python so it's testable independently.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from features import coerce_dtypes


def load_artifact(
    model_path: str | Path = "models/model_kpi_a.joblib",
    schema_path: str | Path = "models/schema_kpi.json",
) -> tuple[dict, dict]:
    """
    Load model artifact and schema from disk.

    Returns
    -------
    artifact : dict  — model, encoder, feature_cols, kpi_tier_labels, tier_info, …
    schema   : dict  — kpi_output, features dict
    """
    artifact = joblib.load(model_path)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return artifact, schema


def predict_one(artifact: dict, input_dict: dict) -> dict:
    """
    Score a single candidate.

    Parameters
    ----------
    artifact   : dict   — loaded via load_artifact()
    input_dict : dict   — {feature_name: value} — missing keys become NaN

    Returns
    -------
    dict with keys:
        tier_idx : int              — 0=At-Risk, 1=Developing, 2=High Performer
        tier     : str              — human label
        proba    : list[float]      — [P(At-Risk), P(Developing), P(High Performer)]
        labels   : list[str]        — ["At-Risk", "Developing", "High Performer"]
    """
    feature_cols   = artifact["feature_cols"]
    high_card_cols = artifact["high_card_cols"]
    encoder        = artifact["encoder"]
    model          = artifact["model"]
    tier_labels    = artifact["kpi_tier_labels"]

    X = pd.DataFrame([input_dict])
    X = X.reindex(columns=feature_cols, fill_value=np.nan)
    X_enc = coerce_dtypes(encoder.transform(X.copy()), high_card_cols)
    for col in X_enc.select_dtypes(include=["object"]).columns:
        X_enc[col] = X_enc[col].astype("category")

    tier_idx = int(model.predict(X_enc)[0])
    proba    = model.predict_proba(X_enc)[0].tolist()

    return {
        "tier_idx": tier_idx,
        "tier":     tier_labels[tier_idx],
        "proba":    proba,
        "labels":   tier_labels,
    }


def predict_batch(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of candidates.
    Appends columns: predicted_tier, p_at_risk, p_developing, p_high_performer.
    """
    results = [predict_one(artifact, row.to_dict()) for _, row in df.iterrows()]
    out = df.copy()
    out["predicted_tier"] = [r["tier"] for r in results]
    out["p_at_risk"]        = [r["proba"][0] for r in results]
    out["p_developing"]     = [r["proba"][1] for r in results]
    out["p_high_performer"] = [r["proba"][2] for r in results]
    return out
```

- [ ] **Step 4: Verify predictor imports and runs correctly**

```bash
cd "E:\work\concentrix\PredictiveModel"
python -c "
from app.predictor import load_artifact, predict_one

artifact, schema = load_artifact()
print('Artifact keys:', list(artifact.keys()))
print('Tier labels:', artifact['kpi_tier_labels'])
print('Tier cutpoints:', artifact['tier_info']['cutpoints'])

# Score a dummy candidate (all zeros / first categorical value from schema)
dummy = {}
for name, info in schema['features'].items():
    if info['type'] == 'binary':
        dummy[name] = 0
    elif info['type'] == 'numeric':
        dummy[name] = info['range'][0]
    else:
        dummy[name] = info['values'][0] if info['values'] else None

result = predict_one(artifact, dummy)
print(f'Predicted tier: {result[\"tier\"]} (idx={result[\"tier_idx\"]})')
print(f'Probabilities: {[round(p, 3) for p in result[\"proba\"]]}')
prob_sum = sum(result['proba'])
assert abs(prob_sum - 1.0) < 0.001, f'Probabilities do not sum to 1: {prob_sum}'
print('PASS: probabilities sum to 1.0')
"
```

Expected output:
```
Artifact keys: ['model', 'encoder', 'feature_cols', 'flag_cols', 'high_card_cols', 'tier_info', 'kpi_tier_labels', 'best_params', 'metrics', 'majority_baseline_acc']
Tier labels: ['At-Risk', 'Developing', 'High Performer']
Tier cutpoints: {'p33': 91.65..., 'p67': 102.97...}
Predicted tier: [one of At-Risk/Developing/High Performer] (idx=[0/1/2])
Probabilities: [three floats summing to ~1.0]
PASS: probabilities sum to 1.0
```

- [ ] **Step 5: Verify predict_batch**

```bash
python -c "
import pandas as pd
from app.predictor import load_artifact, predict_batch

artifact, schema = load_artifact()

# Build 3 dummy rows
rows = []
for _ in range(3):
    dummy = {}
    for name, info in schema['features'].items():
        if info['type'] == 'binary':       dummy[name] = 0
        elif info['type'] == 'numeric':    dummy[name] = info['range'][0]
        else:                              dummy[name] = info['values'][0] if info['values'] else None
    rows.append(dummy)

df = pd.DataFrame(rows)
result = predict_batch(artifact, df)
print('Output columns added:', ['predicted_tier', 'p_at_risk', 'p_developing', 'p_high_performer'])
assert 'predicted_tier' in result.columns
assert 'p_at_risk' in result.columns
assert len(result) == 3
print('PASS: predict_batch returns 3 rows with correct columns')
print(result[['predicted_tier', 'p_at_risk', 'p_developing', 'p_high_performer']])
"
```

Expected: 3 rows, all four new columns present, no crash.

- [ ] **Step 6: Commit**

```bash
git add app/__init__.py app/predictor.py requirements.txt
git commit -m "feat: app/predictor.py — KPI tier inference; add streamlit to requirements"
```

---

## Task 2: app/schema_utils.py — Schema parsing and input validation

**Files:**
- Create: `app/schema_utils.py`

**Interfaces:**
- Consumes: `schema: dict` — output of `json.load(schema_kpi.json)`
- Produces: `get_field_specs(schema) -> list[dict]` — ordered field specs for form generation
  - Each spec: `{'name': str, 'type': 'binary'|'categorical'|'numeric', 'default': Any, 'options': list|None, 'min': float|None, 'max': float|None}`
- Produces: `validate_and_coerce(user_input, schema) -> tuple[dict, list[str]]`
  - Returns `(coerced_values: dict, errors: list[str])`
- Produces: `get_csv_template(schema) -> pd.DataFrame` — one-row DataFrame with all feature columns (NaN values), for CSV template download

- [ ] **Step 1: Create `app/schema_utils.py`**

```python
"""
schema_utils.py — Schema parsing, field spec generation, and input validation.
All logic here is pure Python; no Streamlit dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Field groups (for UI section headers) ────────────────────────────────────
# Prefix → display group name. Unmatched fields go to "Other".
_PREFIX_GROUPS = {
    "L-":   "Leadership Background",
    "S-":   "Support Type Experience",
    "ChS-": "Channel Experience",
    "CL-":  "Certifications & Licences",
    "Tr-":  "Transportation",
    "ISP":  "Internet Service Provider",
    "V-":   "Industry Vertical Experience",
}


def _field_group(name: str) -> str:
    for prefix, label in _PREFIX_GROUPS.items():
        if name.startswith(prefix):
            return label
    return "Demographics & Org Context"


def get_field_specs(schema: dict) -> list[dict]:
    """
    Build ordered field spec list from schema['features'].

    Each spec dict:
        name     : str
        type     : 'binary' | 'categorical' | 'numeric'
        group    : str           — UI section header
        default  : Any           — safe default value
        options  : list | None   — for categorical only
        min      : float | None  — for numeric only
        max      : float | None  — for numeric only
    """
    specs = []
    for name, info in schema["features"].items():
        spec: dict = {"name": name, "type": info["type"], "group": _field_group(name)}
        if info["type"] == "binary":
            spec["default"] = 0
            spec["options"] = None
            spec["min"]     = None
            spec["max"]     = None
        elif info["type"] == "numeric":
            spec["default"] = float(info["range"][0])
            spec["options"] = None
            spec["min"]     = float(info["range"][0])
            spec["max"]     = float(info["range"][1])
        else:
            opts = info.get("values", [])
            spec["default"] = opts[0] if opts else ""
            spec["options"] = opts
            spec["min"]     = None
            spec["max"]     = None
        specs.append(spec)
    return specs


def validate_and_coerce(user_input: dict, schema: dict) -> tuple[dict, list[str]]:
    """
    Validate and coerce raw user input (from form or CSV row) against the schema.

    Returns
    -------
    coerced : dict        — {feature_name: coerced_value} for all schema features
    errors  : list[str]   — human-readable validation messages (empty = valid)
    """
    coerced: dict = {}
    errors:  list = []

    for name, info in schema["features"].items():
        val = user_input.get(name)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            coerced[name] = np.nan
            continue

        if info["type"] == "binary":
            coerced[name] = float(bool(val))

        elif info["type"] == "numeric":
            try:
                coerced[name] = float(val)
            except (ValueError, TypeError):
                coerced[name] = np.nan
                errors.append(f"'{name}': expected a number, got {val!r}")

        else:  # categorical
            coerced[name] = str(val) if val != "" else np.nan

    return coerced, errors


def get_csv_template(schema: dict) -> pd.DataFrame:
    """
    Return a single-row DataFrame with all feature columns set to NaN.
    Used to generate a downloadable CSV template for batch scoring.
    """
    cols = list(schema["features"].keys())
    return pd.DataFrame([{c: np.nan for c in cols}])
```

- [ ] **Step 2: Verify field specs**

```bash
cd "E:\work\concentrix\PredictiveModel"
python -c "
import json
from app.schema_utils import get_field_specs, validate_and_coerce, get_csv_template

schema = json.loads(open('models/schema_kpi.json').read())
specs = get_field_specs(schema)

print(f'Total fields: {len(specs)}')

groups = {}
for s in specs:
    groups.setdefault(s[\"group\"], []).append(s[\"name\"])
print('Groups:')
for g, names in sorted(groups.items()):
    print(f'  {g}: {len(names)} fields')

# Test validate_and_coerce — all zeros/defaults
raw = {s[\"name\"]: s[\"default\"] for s in specs}
coerced, errors = validate_and_coerce(raw, schema)
print(f'Errors on all-default input: {errors}')
assert errors == [], f'Unexpected errors: {errors}'

# Test template
tmpl = get_csv_template(schema)
assert len(tmpl.columns) == len(specs)
print(f'CSV template shape: {tmpl.shape}')
print('PASS: schema_utils all checks passed')
"
```

Expected:
```
Total fields: [N]
Groups:
  Certifications & Licences: [N] fields
  Channel Experience: [N] fields
  Demographics & Org Context: [N] fields
  ...
Errors on all-default input: []
CSV template shape: (1, [N])
PASS: schema_utils all checks passed
```

- [ ] **Step 3: Verify validate_and_coerce catches a bad numeric**

```bash
python -c "
import json
from app.schema_utils import validate_and_coerce

schema = json.loads(open('models/schema_kpi.json').read())
bad_input = {'Site Distance': 'not_a_number'}
coerced, errors = validate_and_coerce(bad_input, schema)
assert len(errors) == 1
assert 'Site Distance' in errors[0]
print('PASS: bad numeric caught:', errors[0])
"
```

Expected: `PASS: bad numeric caught: 'Site Distance': expected a number, got 'not_a_number'`

- [ ] **Step 4: Commit**

```bash
git add app/schema_utils.py
git commit -m "feat: app/schema_utils.py — schema parsing, field specs, input validation"
```

---

## Task 3: app/scoring_app.py — Streamlit scoring UI

**Files:**
- Create: `app/scoring_app.py`

**Interfaces:**
- Consumes: `load_artifact()`, `predict_one()`, `predict_batch()` from `app.predictor`
- Consumes: `get_field_specs()`, `validate_and_coerce()`, `get_csv_template()` from `app.schema_utils`
- Produces: Streamlit app runnable with `streamlit run app/scoring_app.py`

**UI spec:**
- Page title: "KPI Performance Predictor"
- Sidebar: model metadata (tier cutpoints, labeled count, AUC)
- Two tabs: "Score Candidate" (form) and "Batch Upload" (CSV)
- Form tab: fields grouped by section in `st.expander`, predict button, result card
- Batch tab: file uploader, preview, score button, sortable results, CSV download
- Color coding: At-Risk = red (#ef4444), Developing = amber (#f59e0b), High Performer = green (#22c55e)

- [ ] **Step 1: Create `app/scoring_app.py`**

```python
"""
scoring_app.py — Streamlit KPI tier scoring interface.

Run with:
    streamlit run app/scoring_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path resolution ───────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_ROOT     = _HERE.parent
_MODEL    = _ROOT / "models" / "model_kpi_a.joblib"
_SCHEMA   = _ROOT / "models" / "schema_kpi.json"

# Add src/ to path (predictor.py does the same internally)
import sys
sys.path.insert(0, str(_ROOT / "src"))

from app.predictor    import load_artifact, predict_one, predict_batch
from app.schema_utils import get_field_specs, validate_and_coerce, get_csv_template

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "At-Risk":          "#ef4444",
    "Developing":       "#f59e0b",
    "High Performer":   "#22c55e",
}
TIER_ICONS = {
    "At-Risk":          "🔴",
    "Developing":       "🟡",
    "High Performer":   "🟢",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KPI Performance Predictor",
    page_icon="📊",
    layout="wide",
)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def _load():
    return load_artifact(_MODEL, _SCHEMA)

artifact, schema = _load()
field_specs      = get_field_specs(schema)
tier_labels      = artifact["kpi_tier_labels"]
cutpoints        = artifact["tier_info"]["cutpoints"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 KPI Predictor")
    st.caption("Concentrix PH — Advisor Performance")
    st.divider()
    st.subheader("Model Info")
    st.metric("AUC",      f"{artifact['metrics']['auc']:.3f}")
    st.metric("Accuracy", f"{artifact['metrics']['accuracy']:.1%}")
    st.metric("Baseline", f"{artifact['majority_baseline_acc']:.1%}")
    st.metric("Lift",     f"+{artifact['metrics']['lift_vs_baseline']:.1%}")
    st.divider()
    st.subheader("KPI Tier Cutpoints")
    st.info(
        f"**At-Risk** — PTG < {cutpoints['p33']:.1f}%  \n"
        f"**Developing** — {cutpoints['p33']:.1f}% – {cutpoints['p67']:.1f}%  \n"
        f"**High Performer** — PTG > {cutpoints['p67']:.1f}%"
    )
    st.caption(
        f"Trained on {artifact['tier_info']['n_labeled']:,} labeled agents  \n"
        f"Tercile-based: each tier ≈ 33% of the population"
    )

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_form, tab_batch = st.tabs(["Score Candidate", "Batch Upload"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single candidate form
# ═══════════════════════════════════════════════════════════════════════════════
with tab_form:
    st.header("Score a Candidate")
    st.caption("Fill in the pre-hire details below. Leave unknowns at their default.")

    # Group fields by section
    groups: dict[str, list[dict]] = {}
    for spec in field_specs:
        groups.setdefault(spec["group"], []).append(spec)

    form_values: dict = {}

    # Flags section: binary checkboxes in 4-column grid
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
                        st.checkbox(spec["name"], value=bool(spec["default"]), key=f"form_{spec['name']}")
                    )

    # Categorical / numeric in Demographics & Org Context
    demo_group = groups.get("Demographics & Org Context", [])
    if demo_group:
        with st.expander("Demographics & Org Context", expanded=True):
            col1, col2 = st.columns(2)
            for i, spec in enumerate(demo_group):
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    if spec["type"] == "categorical":
                        opts = spec["options"] or []
                        default_idx = 0
                        form_values[spec["name"]] = st.selectbox(
                            spec["name"], opts or ["—"],
                            index=default_idx, key=f"form_{spec['name']}"
                        ) if opts else None
                    elif spec["type"] == "numeric":
                        form_values[spec["name"]] = st.number_input(
                            spec["name"],
                            min_value=spec["min"],
                            max_value=spec["max"],
                            value=spec["default"],
                            key=f"form_{spec['name']}"
                        )
                    else:
                        form_values[spec["name"]] = int(
                            st.checkbox(spec["name"], value=False, key=f"form_{spec['name']}")
                        )

    st.divider()
    predict_btn = st.button("Predict KPI Tier", type="primary", use_container_width=True)

    if predict_btn:
        coerced, errors = validate_and_coerce(form_values, schema)
        if errors:
            for e in errors:
                st.error(e)
        else:
            result = predict_one(artifact, coerced)
            tier   = result["tier"]
            color  = TIER_COLORS[tier]
            icon   = TIER_ICONS[tier]

            st.divider()
            st.subheader("Prediction Result")

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown(
                    f"""<div style="background:{color}22;border:2px solid {color};border-radius:12px;
                    padding:24px;text-align:center;">
                    <div style="font-size:2.5rem;">{icon}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{color};">{tier}</div>
                    <div style="color:#888;font-size:0.85rem;">Predicted KPI Tier</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown("**Probability Breakdown**")
                for i, label in enumerate(tier_labels):
                    p = result["proba"][i]
                    bar_color = TIER_COLORS[label]
                    st.markdown(f"{TIER_ICONS[label]} **{label}**")
                    st.progress(p)
                    st.caption(f"{p:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV upload
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch Score from CSV")

    tmpl = get_csv_template(schema)
    st.download_button(
        "Download CSV Template",
        data=tmpl.to_csv(index=False).encode("utf-8"),
        file_name="kpi_scoring_template.csv",
        mime="text/csv",
        help="Download a blank CSV with all feature columns — fill it in and upload below.",
    )

    uploaded = st.file_uploader("Upload filled CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded, dtype=str)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_raw = None

        if df_raw is not None:
            st.write(f"**{len(df_raw):,} candidates** detected. Preview:")
            st.dataframe(df_raw.head(5), use_container_width=True)

            score_btn = st.button("Score All Candidates", type="primary")
            if score_btn:
                with st.spinner("Scoring…"):
                    # Coerce each row; skip validation errors (fill NaN)
                    coerced_rows = []
                    for _, row in df_raw.iterrows():
                        coerced, _ = validate_and_coerce(row.to_dict(), schema)
                        coerced_rows.append(coerced)
                    df_coerced = pd.DataFrame(coerced_rows)
                    df_results = predict_batch(artifact, df_coerced)

                st.success(f"Scored {len(df_results):,} candidates.")

                # Show results sorted by at-risk probability descending
                display_cols = ["predicted_tier", "p_at_risk", "p_developing", "p_high_performer"]
                df_show = df_results[display_cols].copy()
                df_show = df_show.sort_values("p_at_risk", ascending=False).reset_index(drop=True)

                # Tier summary
                tier_counts = df_show["predicted_tier"].value_counts()
                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 At-Risk",        tier_counts.get("At-Risk", 0))
                m2.metric("🟡 Developing",     tier_counts.get("Developing", 0))
                m3.metric("🟢 High Performer", tier_counts.get("High Performer", 0))

                st.dataframe(
                    df_show.style.format({
                        "p_at_risk":        "{:.1%}",
                        "p_developing":     "{:.1%}",
                        "p_high_performer": "{:.1%}",
                    }),
                    use_container_width=True,
                    height=400,
                )

                # Download results
                full_result_csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Scored Results",
                    data=full_result_csv,
                    file_name="kpi_scored_results.csv",
                    mime="text/csv",
                )
```

- [ ] **Step 2: Verify the app starts without errors**

```bash
cd "E:\work\concentrix\PredictiveModel"
streamlit run app/scoring_app.py --server.headless true --server.port 8501 &
```

Wait 5 seconds, then:
```bash
python -c "
import urllib.request, time
time.sleep(5)
resp = urllib.request.urlopen('http://localhost:8501/healthz')
print('Status:', resp.status)
assert resp.status == 200, f'Expected 200, got {resp.status}'
print('PASS: app running at http://localhost:8501')
"
```

Expected: `Status: 200` and `PASS: app running at http://localhost:8501`

Stop the background server after verification:
```bash
pkill -f "streamlit run app/scoring_app.py" 2>/dev/null || taskkill /f /im streamlit.exe 2>/dev/null || true
```

- [ ] **Step 3: Verify predict_one is called correctly from the form flow**

```bash
python -c "
import json
from app.predictor    import load_artifact, predict_one
from app.schema_utils import get_field_specs, validate_and_coerce

artifact, schema = load_artifact()
specs = get_field_specs(schema)

# Simulate a form submission with all defaults
form_values = {s['name']: s['default'] for s in specs}
coerced, errors = validate_and_coerce(form_values, schema)
assert errors == [], f'Unexpected errors: {errors}'

result = predict_one(artifact, coerced)
assert result['tier'] in ['At-Risk', 'Developing', 'High Performer']
assert abs(sum(result['proba']) - 1.0) < 0.001
print(f'Full form flow: PASS — predicted {result[\"tier\"]}')
"
```

Expected: `Full form flow: PASS — predicted [tier name]`

- [ ] **Step 4: Commit**

```bash
git add app/scoring_app.py
git commit -m "feat: app/scoring_app.py — Streamlit KPI scoring UI with form + batch CSV tabs"
```

---

## Self-Review

**Spec coverage:**
- [x] Load `model_kpi_a.joblib` + `schema_kpi.json` on startup — `load_artifact()` in Task 1
- [x] Accept pre-hire data via form — Tab 1 in Task 3
- [x] Accept pre-hire data via CSV — Tab 2 in Task 3
- [x] Return KPI tier + probability scores — `predict_one()` returns both
- [x] No modification to `src/` files — all new code in `app/`
- [x] `predictor.py` works without Streamlit — pure Python, tested separately in Task 1
- [x] CSV template download — `get_csv_template()` + download button in Task 3
- [x] Color coding (At-Risk=red, Developing=amber, High Performer=green) — `TIER_COLORS` in Task 3
- [x] Batch results sorted by at-risk probability — `sort_values("p_at_risk")` in Task 3

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `predict_one()` returns `dict` with keys `tier_idx`, `tier`, `proba`, `labels` — used correctly in `scoring_app.py`
- `get_field_specs()` returns `list[dict]` with key `group` — used in `scoring_app.py` to build `groups` dict
- `validate_and_coerce()` returns `(dict, list[str])` — unpacked correctly in both form and batch tabs
- `predict_batch()` adds columns `predicted_tier`, `p_at_risk`, `p_developing`, `p_high_performer` — referenced by exact name in `scoring_app.py`
