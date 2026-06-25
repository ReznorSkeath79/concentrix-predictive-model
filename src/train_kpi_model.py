"""
train_kpi_model.py — KPI-tier prediction model (replaces STAR as target).

Source:  CDM Raw-smaller.csv (has Latest Weighted PTG)
Target:  KPI tier {0=At-Risk, 1=Developing, 2=High Performer} via tercile cutoffs
Features: pre-hire flags + demographics + onboarding context (same as Model A+)
Saves:   models/model_kpi_a.joblib + models/schema_kpi.json + reports/kpi_metrics.md
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb

warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CDM_PATH, MODELS_DIR, REPORTS_DIR,
    KPI_TIER_LABELS, N_KPI_CLASSES,
    RANDOM_STATE, CV_FOLDS, TEST_SIZE, OPTUNA_TRIALS,
    LEAKAGE_COLS, ALWAYS_DROP,
)
from features import (
    clean_df, detect_emp_col, get_flag_cols, coerce_flags,
    engineer_features, get_prehire_features, FoldTargetEncoder, coerce_dtypes,
)
from kpi_targets import build_kpi_labels
from evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── 1. Load CDM ───────────────────────────────────────────────────────────────

def load_cdm() -> pd.DataFrame:
    log.info(f"Loading CDM → {CDM_PATH}")
    chunks = []
    reader = pd.read_csv(
        CDM_PATH, dtype=str, encoding='utf-8-sig',
        on_bad_lines='skip', chunksize=5_000,
    )
    for chunk in reader:
        chunks.append(chunk)
    df = clean_df(pd.concat(chunks, ignore_index=True))
    log.info(f"  CDM: {len(df):,} rows × {len(df.columns)} cols")
    return df


# ── 2. Build (X, y) for the KPI-labeled subset ───────────────────────────────

def prep_Xy(df: pd.DataFrame) -> tuple:
    """
    Returns (X, y_tier, feat_cols, tier_info, flag_cols).
    Keeps only rows where Latest Weighted PTG is non-null.
    """
    _, y_tier, tier_info = build_kpi_labels(df)

    df = df.copy().reset_index(drop=True)
    df['_y_kpi'] = y_tier

    labeled = df[df['_y_kpi'].notna()].copy()
    y = labeled['_y_kpi'].astype(int)

    flag_cols = get_flag_cols(labeled)
    labeled = engineer_features(labeled, flag_cols)
    labeled = coerce_flags(labeled, flag_cols)

    feat_cols = get_prehire_features(labeled, flag_cols, include_onboarding=True)
    # Exclude the KPI target itself from features (safety net)
    kpi_target_lower = 'latest weighted ptg'
    feat_cols = [f for f in feat_cols
                 if f in labeled.columns
                 and f.lower().replace('\xa0', ' ').strip() != kpi_target_lower]

    log.info(f"  Labeled: {len(labeled):,} agents | {len(feat_cols)} features")
    log.info(f"  KPI tier dist: {dict(y.value_counts().sort_index())}")

    X = labeled[feat_cols].copy()
    for col in X.select_dtypes(include='object').columns:
        num_try = pd.to_numeric(X[col], errors='coerce')
        if num_try.notna().mean() >= 0.5:
            X[col] = num_try
        else:
            X[col] = X[col].astype('category')

    return X, y, feat_cols, tier_info, flag_cols


# ── 3. Optuna objective ───────────────────────────────────────────────────────

def lgb_objective(trial, X_tr, y_tr, n_folds, high_card_cols):
    params = {
        'n_estimators':      trial.suggest_int  ('n_estimators', 200, 1000),
        'max_depth':         trial.suggest_int  ('max_depth', 3, 8),
        'num_leaves':        trial.suggest_int  ('num_leaves', 15, 127),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'min_child_samples': trial.suggest_int  ('min_child_samples', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'objective':         'multiclass',
        'num_class':         N_KPI_CLASSES,
        'class_weight':      'balanced',
        'random_state':      RANDOM_STATE,
        'n_jobs':            -1,
        'verbose':           -1,
    }

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        X_f_tr, X_f_val = X_tr.iloc[tr_idx].copy(), X_tr.iloc[val_idx].copy()
        y_f_tr, y_f_val = y_tr.iloc[tr_idx],       y_tr.iloc[val_idx]

        enc = FoldTargetEncoder(cols=high_card_cols)
        enc.fit(X_f_tr, y_f_tr)
        X_f_tr  = enc.transform(X_f_tr)
        X_f_val = enc.transform(X_f_val)

        for col in X_f_tr.select_dtypes(include=['object']).columns:
            X_f_tr[col]  = X_f_tr[col].astype('category')
            X_f_val[col] = X_f_val[col].astype('category')

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_f_tr, y_f_tr,
            eval_set=[(X_f_val, y_f_val)],
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_f_val)
        scores.append(f1_score(y_f_val, preds, average='macro', zero_division=0))

    return float(np.mean(scores))


# ── 4. Train final model ──────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    'n_estimators': 500, 'max_depth': 6, 'num_leaves': 63,
    'learning_rate': 0.05, 'min_child_samples': 20,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
}


def train_final(X_tr, y_tr, X_val, y_val, params: dict) -> lgb.LGBMClassifier:
    p = {**params, 'objective': 'multiclass', 'num_class': N_KPI_CLASSES,
         'class_weight': 'balanced', 'random_state': RANDOM_STATE,
         'n_jobs': -1, 'verbose': -1}
    model = lgb.LGBMClassifier(**p)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )
    return model


# ── 5. Main driver ────────────────────────────────────────────────────────────

def run():
    log.info("\n" + "=" * 60 + "\n  MODEL KPI-A  (KPI tier prediction)\n" + "=" * 60)

    df = load_cdm()
    pre_filter = len(df)
    df = df[
        df["Job Title"].str.contains("Advisor", na=False, case=False) |
        (df["Role Type"] == "Production")
    ].copy()
    log.info(f"  Advisor/Production filter: {pre_filter:,} → {len(df):,} rows")
    X, y, feat_cols, tier_info, flag_cols = prep_Xy(df)

    log.info(f"\n  Tercile cutpoints → p33={tier_info['cutpoints']['p33']:.2f}  "
             f"p67={tier_info['cutpoints']['p67']:.2f}")
    log.info(f"  Labeled {tier_info['n_labeled']:,}/{tier_info['n_total']:,} "
             f"({tier_info['pct_labeled']:.1%}) agents have PTG")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    high_card = [c for c in X_train.select_dtypes(include=['object', 'category']).columns
                 if X_train[c].nunique() > 20]
    log.info(f"  High-card encode cols: {len(high_card)}")

    enc = FoldTargetEncoder(cols=high_card)
    enc.fit(X_train, y_train)
    X_train_e = coerce_dtypes(enc.transform(X_train.copy()), high_card)
    X_test_e  = coerce_dtypes(enc.transform(X_test.copy()),  high_card)

    # Majority baseline
    maj = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
    maj.fit(X_train_e, y_train)
    maj_acc = accuracy_score(y_test, maj.predict(X_test_e))
    log.info(f"\n  [1/3] Majority baseline accuracy: {maj_acc:.4f}")

    # Default LGB
    X_sub, X_val2, y_sub, y_val2 = train_test_split(
        X_train_e, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    lgb_base = train_final(X_sub, y_sub, X_val2, y_val2, DEFAULT_PARAMS)
    base_acc = accuracy_score(y_test, lgb_base.predict(X_test_e))
    log.info(f"  [2/3] LGB default test accuracy: {base_acc:.4f}")

    # Optuna
    log.info(f"\n  [3/3] Optuna ({OPTUNA_TRIALS} trials) …")
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
    log.info(f"  Best CV macro-F1: {study.best_value:.4f}")

    lgb_tuned = train_final(X_sub, y_sub, X_val2, y_val2, best_params)
    tuned_acc = accuracy_score(y_test, lgb_tuned.predict(X_test_e))
    log.info(f"  Tuned test accuracy: {tuned_acc:.4f}")

    # SHAP
    shap_vals, shap_X = None, None
    try:
        import shap
        explainer = shap.TreeExplainer(lgb_tuned)
        shap_sample = X_test_e.iloc[:min(500, len(X_test_e))]
        shap_vals = explainer.shap_values(shap_sample)
        shap_X = shap_sample
    except Exception as e:
        log.warning(f"  SHAP skipped: {e}")

    # Evaluate
    metrics = evaluate_model(
        lgb_tuned, X_test_e, y_test.values,
        model_name='model_kpi_a',
        shap_values=shap_vals,
        shap_X=shap_X,
        class_labels=KPI_TIER_LABELS,
        baseline_acc=maj_acc,
    )

    # Persist
    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        'model':          lgb_tuned,
        'encoder':        enc,
        'feature_cols':   list(X_train.columns),
        'flag_cols':      flag_cols,
        'high_card_cols': high_card,
        'tier_info':      tier_info,
        'kpi_tier_labels': KPI_TIER_LABELS,
        'best_params':    best_params,
        'metrics':        {k: v for k, v in metrics.items() if k != 'classification_report'},
        'majority_baseline_acc': maj_acc,
    }
    model_path = MODELS_DIR / 'model_kpi_a.joblib'
    joblib.dump(artifact, model_path)
    log.info(f"\n  Model saved → {model_path}")

    # Schema for Phase-2 upload site
    schema = _build_schema(X_train, flag_cols, enc, tier_info)
    schema_path = MODELS_DIR / 'schema_kpi.json'
    schema_path.write_text(json.dumps(schema, indent=2), encoding='utf-8')
    log.info(f"  Schema saved → {schema_path}")

    _write_kpi_metrics_md(metrics, tier_info, maj_acc)
    return metrics


def _build_schema(X, flag_cols, encoder, tier_info):
    schema = {
        'version':    '1.0',
        'target':     'KPI Tier (Latest Weighted PTG)',
        'kpi_output': {
            'n_classes': N_KPI_CLASSES,
            'labels':    KPI_TIER_LABELS,
            'cutpoints': tier_info['cutpoints'],
        },
        'features': {},
    }
    for col in X.columns:
        dtype = str(X[col].dtype)
        if col in flag_cols:
            schema['features'][col] = {'type': 'binary', 'values': [0, 1]}
        elif 'float' in dtype or 'int' in dtype:
            lo = float(X[col].min()) if X[col].notna().any() else 0.0
            hi = float(X[col].max()) if X[col].notna().any() else 1.0
            schema['features'][col] = {'type': 'numeric', 'range': [lo, hi]}
        else:
            vals = sorted(X[col].dropna().astype(str).unique().tolist())
            schema['features'][col] = {'type': 'categorical', 'values': vals[:50]}
    return schema


def _write_kpi_metrics_md(metrics: dict, tier_info: dict, maj_acc: float):
    """Write KPI-specific metrics report to reports/kpi_metrics.md."""
    REPORTS_DIR.mkdir(exist_ok=True)

    cp = tier_info['cutpoints']
    n_labeled = tier_info['n_labeled']
    n_total   = tier_info['n_total']
    pct       = tier_info['pct_labeled']
    counts    = tier_info.get('counts', {})

    acc       = metrics.get('accuracy', 0)
    auc       = metrics.get('auc_ovr_macro', 'N/A')
    macro_f1  = metrics.get('macro_f1', 0)
    n_test    = metrics.get('n_test', 0)
    cr        = metrics.get('classification_report', {})

    acc_icon = "✅" if acc >= 0.90 else ("🟡" if acc >= 0.70 else "🔴")

    lines = [
        "# KPI Tier Prediction Model (Model KPI-A) — Metrics Report",
        "",
        f"> Generated: 2026-06-25 | Target: Latest Weighted PTG → 3-class tercile tiers",
        "",
        "---",
        "",
        "## Data Summary",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Total CDM agents | {n_total:,} |",
        f"| Labeled (PTG non-null) | {n_labeled:,} ({pct:.1%}) |",
        f"| Tercile p33 cutpoint | {cp['p33']:.4f} |",
        f"| Tercile p67 cutpoint | {cp['p67']:.4f} |",
        f"| At-Risk agents (tier 0) | {counts.get(0, 'N/A'):,} |" if isinstance(counts.get(0), int) else f"| At-Risk agents (tier 0) | {counts.get(0, 'N/A')} |",
        f"| Developing agents (tier 1) | {counts.get(1, 'N/A'):,} |" if isinstance(counts.get(1), int) else f"| Developing agents (tier 1) | {counts.get(1, 'N/A')} |",
        f"| High Performer agents (tier 2) | {counts.get(2, 'N/A'):,} |" if isinstance(counts.get(2), int) else f"| High Performer agents (tier 2) | {counts.get(2, 'N/A')} |",
        "",
        "---",
        "",
        "## Model Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Test samples | {n_test:,} |",
        f"| **Accuracy** | {acc_icon} **{acc:.1%}** |",
        f"| Majority baseline accuracy | {maj_acc:.1%} |",
        f"| Lift vs majority baseline | {acc - maj_acc:+.1%} |",
        f"| Macro F1 | {macro_f1:.4f} |",
        f"| AUC (OvR macro) | {auc if auc is not None else 'N/A'} |",
        "",
        "### Per-Class Results",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]

    for i, lbl in enumerate(KPI_TIER_LABELS):
        r = cr.get(str(i), {})
        lines.append(
            f"| {lbl} | {r.get('precision', 0):.3f} | "
            f"{r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | "
            f"{int(r.get('support', 0)):,} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Phase-2 Readiness",
        "",
        "- Artifact: `models/model_kpi_a.joblib` ✅",
        "- Schema: `models/schema_kpi.json` ✅",
        "- 3-class output: At-Risk | Developing | High Performer ✅",
        "- Tercile cutpoints computed from actual PTG distribution (no forced curve) ✅",
        "",
    ]

    out = REPORTS_DIR / "kpi_metrics.md"
    out.write_text('\n'.join(lines), encoding='utf-8')
    log.info(f"KPI metrics report → {out}")


if __name__ == '__main__':
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    run()
