"""
export_for_viz.py — Export model predictions + SHAP group contributions to JS.

Loads the saved model_a_plus.joblib artifact, rebuilds the test set with the
same random split, computes SHAP on the full test set, groups features by
category, and writes reports/model_viz_data.js for the interactive HTML simulator.

Output JS file defines window.MODEL_DATA = { meta, groups, correlations, samples }
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import shap

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RANDOM_STATE, TEST_SIZE, STAR_MAP, STAR_LABELS, FLAG_PREFIXES,
    PREHIRE_CAT_COLS, PREHIRE_NUM_COLS, LEAKAGE_COLS, ALWAYS_DROP,
)
from features import (
    clean_df, detect_emp_col, detect_star_col, get_flag_cols,
    coerce_flags, engineer_features, get_prehire_features,
    FoldTargetEncoder, coerce_dtypes,
)
from train_model_a import load_sdd1, load_cdm, build_master_roster

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature group definitions
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS = [
    {
        "key":   "career",
        "label": "Career Track & Job Grade",
        "desc":  "Job Grade and Management Level — seniority and organizational track at the time of hire.",
        "cols":  ["Job Grade", "Management Level Description", "MSA Fusion"],
    },
    {
        "key":   "wfh",
        "label": "Work-From-Home Setup",
        "desc":  "Whether the agent is WFH-enabled and what type of WFH arrangement they have.",
        "cols":  ["Work At Home Status", "Work At Home Sub Status"],
    },
    {
        "key":   "certifications",
        "label": "Professional Certifications",
        "desc":  "CL-prefix flags: PRC license, CISCO, Salesforce, Six Sigma, Domo, BI, Public Service credentials.",
        "prefix": "CL-",
    },
    {
        "key":   "vertical",
        "label": "Industry Vertical Background",
        "desc":  "V-prefix flags: previous work industry — Healthcare, Insurance, Technical, Consumer, Utilities.",
        "prefix": "V-",
    },
    {
        "key":   "channel",
        "label": "Channel Experience",
        "desc":  "ChS-prefix flags: prior experience in voice inbound, chat, back office, email, escalations.",
        "prefix": "ChS-",
    },
    {
        "key":   "languages",
        "label": "Language & Specialization",
        "desc":  "L-prefix flags: language skills and specialized support roles (SME, Quality, HR, Supervisor).",
        "prefix": "L-",
    },
    {
        "key":   "skills",
        "label": "Support Skills",
        "desc":  "S-prefix flags: specific support skill types — retention, escalations, KYC, TS, specialized.",
        "prefix": "S-",
    },
    {
        "key":   "internet",
        "label": "Internet Setup (ISP)",
        "desc":  "ISP-prefix flags: internet provider type — Converge, PLDT, Globe, Red Fiber, etc.",
        "prefix": "ISP",
    },
    {
        "key":   "transport",
        "label": "Commute & Transport",
        "desc":  "Tr-prefix flags: transportation mode — bus, jeepney, tricycle, Grab, fastcraft.",
        "prefix": "Tr-",
    },
    {
        "key":   "education",
        "label": "Education Level",
        "desc":  "GC Education field from the pre-hire survey — highest completed level.",
        "cols":  ["GC Education"],
    },
    {
        "key":   "geography",
        "label": "Location & Geography",
        "desc":  "Region, province, city, site distance bucket, site latitude/longitude, zip code, valley fault proximity.",
        "cols":  ["GC Region", "GC Province", "GC City", "GC Barangay", "xSite",
                  "Site Distance", "dist_bucket", "Site Latitude", "Site Longitude",
                  "Zip Code", "Valley Fault"],
    },
    {
        "key":   "counts",
        "label": "Activity & Flag Counts",
        "desc":  "Derived count features: total certifications, verticals, skills, and overall flag completeness.",
        "col_startswith": ["cnt_", "flag_completeness"],
    },
]


def assign_group(col_name: str, feature_groups: list) -> str:
    """Return the group key that owns this column, or 'other'."""
    for g in feature_groups:
        if "cols" in g and col_name in g["cols"]:
            return g["key"]
        if "prefix" in g and col_name.startswith(g["prefix"]):
            return g["key"]
        if "col_startswith" in g:
            for pfx in g["col_startswith"]:
                if col_name.startswith(pfx):
                    return g["key"]
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load data + rebuild test split (same RANDOM_STATE as training)
# ─────────────────────────────────────────────────────────────────────────────

log.info("Loading data …")
sdd1   = load_sdd1()
cdm    = load_cdm()
roster = build_master_roster(sdd1, cdm)
flag_cols = get_flag_cols(roster)

labeled = roster[roster['_y'].notna()].copy()
labeled = engineer_features(labeled, flag_cols)
labeled = coerce_flags(labeled, flag_cols)

feat_cols = get_prehire_features(labeled, flag_cols, include_onboarding=True)
feat_cols = [f for f in feat_cols if f in labeled.columns]

X_all = labeled[feat_cols].copy()
y_all = labeled['_y'].astype(int)

# Coerce dtypes (mirror train_model_a.py)
for col in X_all.select_dtypes(include='object').columns:
    num_try = pd.to_numeric(X_all[col], errors='coerce')
    if num_try.notna().mean() >= 0.5:
        X_all[col] = num_try
    else:
        X_all[col] = X_all[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
)
log.info(f"Train={len(X_train):,}  Test={len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load saved artifact + apply encoder to test set
# ─────────────────────────────────────────────────────────────────────────────

log.info("Loading model_a_plus.joblib …")
artifact   = joblib.load(MODELS_DIR / "model_a_plus.joblib")
model      = artifact['model']
enc        = artifact['encoder']
high_card  = artifact['high_card_cols']

X_test_e = coerce_dtypes(enc.transform(X_test.copy()), high_card)
# Ensure all columns from training are present
for col in artifact['feature_cols']:
    if col not in X_test_e.columns:
        X_test_e[col] = 0.0

X_test_e = X_test_e[artifact['feature_cols']]

log.info("Computing base predictions …")
base_proba    = model.predict_proba(X_test_e)   # (n, 4)
base_log_odds = np.log(np.clip(base_proba, 1e-9, 1.0))
y_pred        = np.argmax(base_proba, axis=1)

base_acc = float(accuracy_score(y_test, y_pred))
base_auc = float(roc_auc_score(
    pd.get_dummies(y_test.values).values,
    base_proba, multi_class='ovr', average='macro'
))
log.info(f"Base accuracy={base_acc:.4f}  AUC={base_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute SHAP on full test set
# ─────────────────────────────────────────────────────────────────────────────

log.info("Computing SHAP values (full test set) …")
explainer  = shap.TreeExplainer(model)
shap_raw   = explainer.shap_values(X_test_e)

# Normalize to (n_samples, n_features, n_classes)
if isinstance(shap_raw, list):
    shap_3d = np.stack(shap_raw, axis=2)          # old format: list of (n,f) → (n,f,c)
elif np.array(shap_raw).ndim == 3:
    shap_3d = np.array(shap_raw)                   # (n,f,c) new format
else:
    shap_3d = np.array(shap_raw)[:, :, np.newaxis] # edge case

n_samples, n_features, n_classes = shap_3d.shape
log.info(f"SHAP shape: {shap_3d.shape}")

feat_list = list(X_test_e.columns)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Sum SHAP values by feature group (per sample, per class)
# ─────────────────────────────────────────────────────────────────────────────

group_keys = [g["key"] for g in FEATURE_GROUPS]

# Map each feature to a group index
feat_to_group = {}
for i, col in enumerate(feat_list):
    gkey = assign_group(col, FEATURE_GROUPS)
    if gkey == "other":
        gkey = "career"  # fallback: lump unknowns into career
    feat_to_group[i] = group_keys.index(gkey)

# Aggregate: group_shap[sample, group, class]
group_shap = np.zeros((n_samples, len(group_keys), n_classes), dtype=np.float32)
for feat_idx in range(n_features):
    g_idx = feat_to_group[feat_idx]
    group_shap[:, g_idx, :] += shap_3d[:, feat_idx, :]

log.info("SHAP grouped.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature group → STAR correlation
#    Use mean absolute SHAP per group per class as a "contribution strength"
# ─────────────────────────────────────────────────────────────────────────────

correlations = {}
for g_idx, g in enumerate(FEATURE_GROUPS):
    # Mean signed SHAP per class — positive = pushes toward that class
    # Represents "average contribution of this group to each STAR class"
    mean_shap = group_shap[:, g_idx, :].mean(axis=0).tolist()
    correlations[g["key"]] = [round(v, 5) for v in mean_shap]

# ─────────────────────────────────────────────────────────────────────────────
# 7. Build sample-level data for JS
#    Format: [actual(int), base_log_odds[4], group_shap[n_groups][4]]
# ─────────────────────────────────────────────────────────────────────────────

log.info("Building sample array …")
samples = []
for i in range(n_samples):
    actual = int(y_test.iloc[i])
    blo    = [round(float(v), 5) for v in base_log_odds[i]]
    gs     = [[round(float(group_shap[i, g, c]), 5) for c in range(n_classes)]
              for g in range(len(group_keys))]
    samples.append([actual, blo, gs])

log.info(f"Sample array built: {len(samples):,} entries")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature group descriptions with per-class contribution labels
# ─────────────────────────────────────────────────────────────────────────────

groups_out = []
for g in FEATURE_GROUPS:
    gkey = g["key"]
    corr = correlations[gkey]
    dominant_class = int(np.argmax(np.abs(corr)))
    dominant_dir   = "↑ STAR 5 signal" if dominant_class == 3 else \
                     "↑ STAR 1 risk"   if dominant_class == 0 else \
                     f"↑ STAR {[1,3,4,5][dominant_class]} signal"
    groups_out.append({
        "key":           gkey,
        "label":         g["label"],
        "desc":          g["desc"],
        "dominant_dir":  dominant_dir,
        "mean_shap":     corr,
    })

# ─────────────────────────────────────────────────────────────────────────────
# 9. Write JS file
# ─────────────────────────────────────────────────────────────────────────────

payload = {
    "meta": {
        "model":        "Model A+ (Pre-Hire + Onboarding)",
        "n_test":       n_samples,
        "n_features":   n_features,
        "n_groups":     len(group_keys),
        "n_classes":    n_classes,
        "star_labels":  STAR_LABELS,
        "star_values":  [1, 3, 4, 5],
        "baseline_acc": 0.5425,
        "base_accuracy": round(base_acc, 4),
        "base_auc":      round(base_auc, 4),
    },
    "groups":       groups_out,
    "correlations": correlations,
    "samples":      samples,
}

js_path = REPORTS_DIR / "model_viz_data.js"
js_content = "window.MODEL_DATA = " + json.dumps(payload, separators=(',', ':')) + ";"

js_path.write_text(js_content, encoding="utf-8")
size_mb = js_path.stat().st_size / 1e6
log.info(f"✓ Written → {js_path}  ({size_mb:.1f} MB)")
log.info(f"  Base accuracy: {base_acc:.4f}  |  Base AUC: {base_auc:.4f}")
log.info("  Load the updated executive_summary.html to use the simulator.")
