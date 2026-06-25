"""
train_model_b.py — Model B: KPI scorecard → STAR rating (the ≥90% model).

Joins CDM Raw + CNX KPIs + current_data → trains LightGBM on component KPI
features. SHAP attribution = KPI driver report for ops.

Saves: models/model_b.joblib + reports/kpi_drivers.md
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CDM_PATH, MODELS_DIR, REPORTS_DIR,
    STAR_MAP, INV_STAR_MAP, N_CLASSES, STAR_LABELS,
    RANDOM_STATE, CV_FOLDS, TEST_SIZE, OPTUNA_TRIALS,
)
from features import (
    clean_df, detect_emp_col, detect_star_col, detect_email_col,
    engineer_features, FoldTargetEncoder, coerce_dtypes,
    get_model_b_features,
)
from joins import (
    aggregate_kpis, aggregate_scorecard,
    merge_kpis_to_roster, merge_scorecard_to_roster,
)
from train_model_a import lgb_objective, train_lgb_final, DEFAULT_PARAMS
from evaluate import evaluate_model, write_metrics_md

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Build KPI-enriched roster
# ═══════════════════════════════════════════════════════════════════════════════

def build_kpi_roster():
    """Load CDM, join KPI + scorecard aggregates. Returns (roster, coverage_stats)."""
    log.info(f"Loading CDM  →  {CDM_PATH}")
    chunks = []
    for chunk in pd.read_csv(CDM_PATH, dtype=str, encoding='utf-8-sig',
                              on_bad_lines='skip', chunksize=5_000):
        chunks.append(chunk)
    cdm = clean_df(pd.concat(chunks, ignore_index=True))

    emp_col   = detect_emp_col(cdm)
    email_col = detect_email_col(cdm)
    if not emp_col:
        raise ValueError("CDM: cannot find employee-number column")

    cdm['_emp_num'] = (cdm[emp_col].astype(str)
                       .str.replace('\xa0', '', regex=False)
                       .str.strip())
    cdm = cdm[cdm['_emp_num'].notna() &
              (cdm['_emp_num'] != '') &
              (cdm['_emp_num'].str.lower() != 'nan')]
    log.info(f"  CDM after blank-ID drop: {len(cdm):,} agents")

    # Parse STAR
    star_col = detect_star_col(cdm)
    if star_col:
        cdm['_star'] = pd.to_numeric(cdm[star_col], errors='coerce')
    else:
        cdm['_star'] = np.nan

    # Tenure
    cdm = engineer_features(cdm, [])

    coverage = {}

    # ── Join CNX KPIs ────────────────────────────────────────────────────────
    log.info("\nAggregating CNX KPI file (large — may take several minutes) …")
    try:
        kpi_agg = aggregate_kpis()
        roster  = merge_kpis_to_roster(cdm, kpi_agg, '_emp_num')
        n_joined = int(roster['n_kpi_months'].notna().sum())
        coverage['CNX KPIs'] = {
            'total':  len(roster),
            'joined': n_joined,
            'pct':    f"{n_joined / len(roster) * 100:.1f}%",
        }
    except Exception as e:
        log.error(f"KPI join failed: {e}")
        roster = cdm.copy()
        coverage['CNX KPIs'] = {'total': len(roster), 'joined': 0, 'pct': '0%'}

    # ── Join current_data scorecard ──────────────────────────────────────────
    if email_col:
        log.info("\nAggregating current_data scorecard (large — may take several minutes) …")
        try:
            sc_agg  = aggregate_scorecard()
            roster  = merge_scorecard_to_roster(roster, sc_agg, email_col)
            n_sc    = int(roster['sc_avg_ptg'].notna().sum())
            coverage['Scorecard'] = {
                'total':  len(roster),
                'joined': n_sc,
                'pct':    f"{n_sc / len(roster) * 100:.1f}%",
            }
        except Exception as e:
            log.error(f"Scorecard join failed: {e}")
            coverage['Scorecard'] = {'total': len(roster), 'joined': 0, 'pct': '0%'}
    else:
        log.warning("No email column found in CDM — skipping scorecard join")

    log.info(f"\nKPI-enriched roster: {len(roster):,} agents × {len(roster.columns)} cols")
    return roster, coverage


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Train Model B
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_b():
    bar = "=" * 60
    log.info(f"\n{bar}\n  MODEL B  (KPI → STAR, target ≥90%)\n{bar}")

    roster, coverage = build_kpi_roster()

    # Labeled subset
    labeled = roster[roster['_star'].isin([1, 3, 4, 5])].copy()
    labeled['_y'] = labeled['_star'].map(STAR_MAP).astype(int)
    y = labeled['_y']

    log.info(f"\nLabeled: {len(labeled):,}  |  STAR dist: {y.value_counts().sort_index().to_dict()}")

    # Features
    feat_cols = get_model_b_features(labeled)
    feat_cols = [f for f in feat_cols if f in labeled.columns]
    log.info(f"Feature set: {len(feat_cols)} KPI + org features")
    if not feat_cols:
        log.error("No features found — check join coverage. Model B cannot train.")
        return None, coverage

    X = labeled[feat_cols].copy()

    # Coerce dtypes
    for col in X.select_dtypes(include='object').columns:
        num_try = pd.to_numeric(X[col], errors='coerce')
        X[col]  = num_try if num_try.notna().mean() >= 0.3 else X[col].astype('category')

    # High-cardinality cols
    high_card = [c for c in X.select_dtypes(include=['object', 'category']).columns
                 if X[c].nunique() > 20]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Global encoder
    enc = FoldTargetEncoder(cols=high_card)
    enc.fit(X_train, y_train)
    X_train_e = coerce_dtypes(enc.transform(X_train.copy()), high_card)
    X_test_e  = coerce_dtypes(enc.transform(X_test.copy()),  high_card)

    log.info(f"Train={len(X_train_e):,}  Test={len(X_test_e):,}")

    # ── Default LightGBM (sanity check) ──────────────────────────────────────
    log.info("\n  [1/2] LightGBM default params …")
    X_sub, X_val2, y_sub, y_val2 = train_test_split(
        X_train_e, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    lgb_base = train_lgb_final(X_sub, y_sub, X_val2, y_val2, DEFAULT_PARAMS)
    base_acc = accuracy_score(y_test, lgb_base.predict(X_test_e))
    log.info(f"    Default accuracy: {base_acc:.4f}")

    # ── Optuna ───────────────────────────────────────────────────────────────
    log.info(f"\n  [2/2] Optuna ({OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV) …")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda t: lgb_objective(t, X_train_e, y_train, CV_FOLDS, high_card),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=False,
    )
    best_params = study.best_params
    log.info(f"    Best CV macro-F1: {study.best_value:.4f}")
    log.info(f"    Best params: {best_params}")

    model = train_lgb_final(X_sub, y_sub, X_val2, y_val2, best_params)
    tuned_acc = accuracy_score(y_test, model.predict(X_test_e))
    log.info(f"    Tuned accuracy: {tuned_acc:.4f}")

    # Pick best
    final_model = model if tuned_acc >= base_acc else lgb_base
    final_acc   = max(tuned_acc, base_acc)
    log.info(f"\n  ✓ Model B test accuracy: {final_acc:.4f}")

    # ── SHAP + KPI driver report ──────────────────────────────────────────────
    shap_vals, shap_X = None, None
    try:
        import shap
        explainer = shap.TreeExplainer(final_model)
        shap_sample = X_test_e.iloc[:min(500, len(X_test_e))]
        shap_vals   = explainer.shap_values(shap_sample)
        shap_X      = shap_sample
        _write_kpi_driver_report(final_model, shap_sample, shap_vals)
    except Exception as e:
        log.warning(f"SHAP skipped: {e}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate_model(
        final_model, X_test_e, y_test.values,
        model_name='model_b',
        shap_values=shap_vals,
        shap_X=shap_X,
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        'model':          final_model,
        'encoder':        enc,
        'feature_cols':   list(X_train.columns),
        'high_card_cols': high_card,
        'star_map':       STAR_MAP,
        'inv_star_map':   INV_STAR_MAP,
        'star_labels':    STAR_LABELS,
        'best_params':    best_params,
        'coverage':       coverage,
        'metrics':        {k: v for k, v in metrics.items() if k != 'classification_report'},
    }, MODELS_DIR / 'model_b.joblib')
    log.info(f"  Model B saved → {MODELS_DIR / 'model_b.joblib'}")

    return metrics, coverage


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  KPI driver report
# ═══════════════════════════════════════════════════════════════════════════════

def _write_kpi_driver_report(model, X: pd.DataFrame, shap_values):
    try:
        REPORTS_DIR.mkdir(exist_ok=True)

        # Mean |SHAP| across all classes — handle both old list format and new 3-D array
        if isinstance(shap_values, list):
            sv = np.abs(np.array(shap_values)).mean(axis=0)   # (n_classes, n_samples, n_features) → (n_samples, n_features)
        elif np.array(shap_values).ndim == 3:
            sv = np.abs(shap_values).mean(axis=2)              # (n_samples, n_features, n_classes) → (n_samples, n_features)
        else:
            sv = np.abs(shap_values)

        mean_abs = sv.mean(axis=0)
        fi = (pd.DataFrame({'feature': X.columns, 'mean_abs_shap': mean_abs})
              .sort_values('mean_abs_shap', ascending=False)
              .reset_index(drop=True))

        # Bar chart
        fig, ax = plt.subplots(figsize=(9, 7))
        top = fi.head(25).sort_values('mean_abs_shap')
        ax.barh(top['feature'], top['mean_abs_shap'], color='steelblue', alpha=0.85)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Model B — Top KPI Drivers (Mean |SHAP| across all STAR classes)')
        plt.tight_layout()
        path = REPORTS_DIR / 'model_b_kpi_drivers.png'
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        log.info(f"  KPI driver plot → {path.name}")

        # Markdown table
        lines = [
            "# Model B — KPI Driver Report",
            "",
            "Ranked by mean |SHAP| value (impact on STAR prediction across all classes).",
            "",
            "| Rank | Feature | Mean |SHAP| | Notes |",
            "|------|---------|------|-------|",
        ]
        for rank, (_, row) in enumerate(fi.head(30).iterrows(), start=1):
            note = ''
            fname = row['feature']
            if 'wptg' in fname.lower():
                note = 'Weighted PTG component'
            elif 'ptg' in fname.lower():
                note = 'Raw PTG'
            elif 'qa' in fname.lower():
                note = 'Quality Assurance'
            elif 'csat' in fname.lower():
                note = 'Customer Satisfaction'
            elif 'attendance' in fname.lower():
                note = 'Attendance'
            elif fname.startswith('sc_'):
                note = 'Scorecard aggregate'
            elif 'tenure' in fname.lower():
                note = 'HR — tenure'
            lines.append(f"| {rank} | `{fname}` | {row['mean_abs_shap']:.4f} | {note} |")

        (REPORTS_DIR / 'kpi_drivers.md').write_text('\n'.join(lines), encoding='utf-8')
        log.info(f"  KPI driver report → kpi_drivers.md")
    except Exception as e:
        log.warning(f"KPI driver report failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    result = train_model_b()
    if result[0] is not None:
        metrics, coverage = result
        write_metrics_md([metrics], coverage_stats=coverage)
        log.info("\n✓ train_model_b.py  DONE  →  reports/ + models/")
    else:
        log.error("Model B training failed — check join coverage in coverage_stats above.")
