"""
scoring_app.py — Streamlit KPI tier scoring interface.

Run with:
    streamlit run app/scoring_app.py
"""

from __future__ import annotations

from pathlib import Path

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
    # Groups whose fields are binary flags — rendered as 4-column checkbox grids.
    # If the schema gains a new prefix-based group, add its name here.
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
