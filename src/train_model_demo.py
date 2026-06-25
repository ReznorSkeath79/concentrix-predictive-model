"""
train_model_demo.py — Three demographic experiments against STAR prediction.

  EXP 1  Pure Demographics  — survey flags stripped, raw demo fields only (~16 features)
  EXP 2  Full CDM Demo      — all non-leakage CDM columns (~60-80 features)
  EXP 3  Interpretability   — Model-A feature set + LogisticRegression coefficients

Outputs:
  reports/demo_experiments.md  — comparison table + EXP3 coefficient highlights
  models/model_exp1.joblib
  models/model_exp2.joblib
  models/model_exp3_lgb.joblib
  models/model_exp3_lr.joblib
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SDD1_PATH, CDM_PATH, MODELS_DIR, REPORTS_DIR,
    STAR_MAP, N_CLASSES, STAR_LABELS,
    RANDOM_STATE, CV_FOLDS, TEST_SIZE, OPTUNA_TRIALS,
    LEAKAGE_COLS, ALWAYS_DROP, FLAG_PREFIXES,
    PREHIRE_CAT_COLS, PREHIRE_NUM_COLS,
)
from features import (
    clean_df, detect_emp_col, detect_star_col,
    get_flag_cols, coerce_flags, engineer_features,
    get_prehire_features, FoldTargetEncoder, coerce_dtypes,
)
from train_model_a import load_sdd1, load_cdm, build_master_roster, lgb_objective, train_lgb_final

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ─── shared leakage sets (lowercase for case-insensitive checks) ──────────────
_LEAKAGE_L  = {c.lower() for c in LEAKAGE_COLS}
_ALWAYS_L   = {c.lower() for c in ALWAYS_DROP}
_INTERNAL   = {'_emp_num', '_source', '_star', '_y', '_src_order',
               'email_lower', '_email_lower', '_emp_str'}
_RAW_DATES  = {'hire date', 'date', 'termination date', 'rehire date'}


# ═══════════════════════════════════════════════════════════════════════════════
# Shared LightGBM training driver (reusable across experiments)
# ═══════════════════════════════════════════════════════════════════════════════

def run_lgb_experiment(X: pd.DataFrame, y: pd.Series,
                       high_card_cols: list, label: str) -> dict:
    """
    Full LightGBM pipeline:
      1. Majority baseline
      2. Default params
      3. Optuna (OPTUNA_TRIALS, CV_FOLDS)
      4. Final refit on train split
    Returns metrics dict.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"  Train={len(X_tr):,}  Test={len(X_te):,}")

    # Baseline
    base = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
    base.fit(X_tr, y_tr)
    base_acc = accuracy_score(y_te, base.predict(X_te))
    log.info(f"  Baseline accuracy: {base_acc:.4f}")

    # Default LightGBM
    enc0 = FoldTargetEncoder(cols=high_card_cols)
    enc0.fit(X_tr, y_tr)
    X_tr0 = coerce_dtypes(enc0.transform(X_tr), high_card_cols)
    X_te0 = coerce_dtypes(enc0.transform(X_te), high_card_cols)
    default_params = {
        'n_estimators': 500, 'max_depth': 6, 'num_leaves': 63,
        'learning_rate': 0.05, 'min_child_samples': 20,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'objective': 'multiclass', 'num_class': N_CLASSES,
        'class_weight': 'balanced', 'random_state': RANDOM_STATE,
        'n_jobs': -1, 'verbose': -1,
    }
    m0 = lgb.LGBMClassifier(**default_params)
    m0.fit(X_tr0, y_tr, eval_set=[(X_te0, y_te)],
           callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    log.info(f"  Default accuracy: {accuracy_score(y_te, m0.predict(X_te0)):.4f}")

    # Optuna search
    log.info(f"  Optuna ({OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV) …")
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda t: lgb_objective(t, X_tr.copy(), y_tr, CV_FOLDS, high_card_cols),
        n_trials=OPTUNA_TRIALS, show_progress_bar=False,
    )
    best_params = study.best_params
    log.info(f"  Best CV macro-F1: {study.best_value:.4f}")

    # Final model
    enc_fin = FoldTargetEncoder(cols=high_card_cols)
    enc_fin.fit(X_tr, y_tr)
    X_tr_f = coerce_dtypes(enc_fin.transform(X_tr), high_card_cols)
    X_te_f = coerce_dtypes(enc_fin.transform(X_te), high_card_cols)
    model = train_lgb_final(X_tr_f, y_tr, X_te_f, y_te, best_params)

    y_pred  = model.predict(X_te_f)
    y_proba = model.predict_proba(X_te_f)
    acc     = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro')
    except Exception:
        auc = None

    log.info(f"  [{label}] Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | AUC={f'{auc:.4f}' if auc else 'N/A'}")

    return {
        'label':      label,
        'n_features': X.shape[1],
        'n_train':    len(X_tr),
        'n_test':     len(X_te),
        'baseline':   round(base_acc, 4),
        'accuracy':   round(acc, 4),
        'macro_f1':   round(macro_f1, 4),
        'auc':        round(auc, 4) if auc else None,
        'model':      model,
        'encoder':    enc_fin,
        'X_te':       X_te_f,
        'y_te':       y_te,
        'feat_cols':  list(X.columns),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1 — Pure Demographics (no survey flags)
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp1(roster: pd.DataFrame, flag_cols: list) -> dict:
    log.info("\n" + "=" * 60)
    log.info("  EXP 1 — Pure Demographics (no survey flags)")
    log.info("=" * 60)

    labeled = roster[roster['_y'].notna()].copy()
    labeled = engineer_features(labeled, flag_cols)
    y = labeled['_y'].astype(int)

    # Pure demo = PREHIRE_CAT + PREHIRE_NUM + engineered geo/distance only
    # Explicitly exclude: all flag cols and flag-count/completeness engineered cols
    flag_set = set(flag_cols)
    flag_eng = {c for c in labeled.columns
                if c.startswith('cnt_') or c == 'flag_completeness'}

    demo_candidates = []
    for col in PREHIRE_CAT_COLS + PREHIRE_NUM_COLS + ['dist_bucket', 'tenure_months', 'tenure_bucket']:
        if col in labeled.columns:
            demo_candidates.append(col)

    feat_cols = [
        c for c in demo_candidates
        if c not in flag_set
        and c not in flag_eng
        and c.lower() not in _LEAKAGE_L
        and c.lower() not in _ALWAYS_L
    ]
    feat_cols = list(dict.fromkeys(feat_cols))  # dedup preserve order
    log.info(f"  Feature cols: {len(feat_cols)} (pure demo, no flags)")

    X = labeled[feat_cols].copy()
    for col in X.select_dtypes(include='object').columns:
        num_try = pd.to_numeric(X[col], errors='coerce')
        if num_try.notna().mean() >= 0.5:
            X[col] = num_try
        else:
            X[col] = X[col].astype('category')

    high_card = [c for c in feat_cols
                 if c in labeled.columns
                 and labeled[c].dtype == object
                 and labeled[c].nunique() > 20]

    result = run_lgb_experiment(X, y, high_card, 'exp1_pure_demo')
    joblib.dump(result['model'], MODELS_DIR / 'model_exp1.joblib')
    log.info(f"  Saved → models/model_exp1.joblib")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 2 — Full CDM Demographics
# ═══════════════════════════════════════════════════════════════════════════════

def get_cdm_features(df: pd.DataFrame) -> list:
    """
    Select all CDM columns that are not leakage, IDs, internal keys, or raw dates.
    High-cardinality string cols will be target-encoded.
    """
    candidates = []
    for col in df.columns:
        cl = col.lower().strip()
        if cl in _LEAKAGE_L:
            continue
        if cl in _ALWAYS_L:
            continue
        if col in _INTERNAL:
            continue
        if cl in _RAW_DATES:
            continue
        # Skip flag columns — they belong to EXP1/EXP3, not CDM-only analysis
        is_flag = any(col.startswith(p) for p in FLAG_PREFIXES)
        if is_flag:
            continue
        # Skip engineered flag counts (they're derived from flags)
        if col.startswith('cnt_') or col == 'flag_completeness':
            continue
        candidates.append(col)
    return list(dict.fromkeys(candidates))


def run_exp2(roster: pd.DataFrame, flag_cols: list) -> dict:
    log.info("\n" + "=" * 60)
    log.info("  EXP 2 — Full CDM Demographics")
    log.info("=" * 60)

    labeled = roster[roster['_y'].notna()].copy()
    labeled = engineer_features(labeled, flag_cols)
    y = labeled['_y'].astype(int)

    feat_cols = get_cdm_features(labeled)
    feat_cols = [c for c in feat_cols if c in labeled.columns]
    log.info(f"  Feature cols: {len(feat_cols)} (all CDM non-leakage cols)")

    X = labeled[feat_cols].copy()
    for col in X.select_dtypes(include='object').columns:
        num_try = pd.to_numeric(X[col], errors='coerce')
        if num_try.notna().mean() >= 0.5:
            X[col] = num_try
        else:
            X[col] = X[col].astype('category')

    high_card = [c for c in feat_cols
                 if c in labeled.columns
                 and labeled[c].dtype == object
                 and labeled[c].nunique() > 20]

    log.info(f"  High-card target-encode cols ({len(high_card)}): {high_card[:5]}…")
    result = run_lgb_experiment(X, y, high_card, 'exp2_full_cdm')
    joblib.dump(result['model'], MODELS_DIR / 'model_exp2.joblib')
    log.info(f"  Saved → models/model_exp2.joblib")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 3 — Interpretability: LightGBM + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_logreg(X_tr: pd.DataFrame, y_tr: pd.Series,
                X_te: pd.DataFrame, y_te: pd.Series):
    """
    Fit a LogisticRegression on the same feature set as Model A.
    All categoricals → OrdinalEncoder; all numerics → StandardScaler.
    Returns (fitted_pipeline, metrics_dict).
    """
    cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_tr.select_dtypes(include=['number']).columns.tolist()

    # Encode cats with ordinal (unknown → -1)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr_cat = pd.DataFrame(
        oe.fit_transform(X_tr[cat_cols].astype(str)) if cat_cols else np.empty((len(X_tr), 0)),
        columns=cat_cols, index=X_tr.index,
    )
    X_te_cat = pd.DataFrame(
        oe.transform(X_te[cat_cols].astype(str)) if cat_cols else np.empty((len(X_te), 0)),
        columns=cat_cols, index=X_te.index,
    )

    # Scale numerics
    sc = StandardScaler()
    X_tr_num = pd.DataFrame(
        sc.fit_transform(X_tr[num_cols].fillna(0)) if num_cols else np.empty((len(X_tr), 0)),
        columns=num_cols, index=X_tr.index,
    )
    X_te_num = pd.DataFrame(
        sc.transform(X_te[num_cols].fillna(0)) if num_cols else np.empty((len(X_te), 0)),
        columns=num_cols, index=X_te.index,
    )

    X_tr_enc = pd.concat([X_tr_num, X_tr_cat], axis=1).fillna(0)
    X_te_enc = pd.concat([X_te_num, X_te_cat], axis=1).fillna(0)

    lr = LogisticRegression(
        C=1.0, max_iter=2000,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
    )
    lr.fit(X_tr_enc, y_tr)
    y_pred  = lr.predict(X_te_enc)
    y_proba = lr.predict_proba(X_te_enc)
    acc      = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro')
    except Exception:
        auc = None

    log.info(f"  [exp3_logreg] Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | AUC={f'{auc:.4f}' if auc else 'N/A'}")

    return lr, X_tr_enc, {
        'accuracy': round(acc, 4),
        'macro_f1': round(macro_f1, 4),
        'auc':      round(auc, 4) if auc else None,
        'oe':       oe, 'sc': sc,
        'cat_cols': cat_cols, 'num_cols': num_cols,
    }


def _coef_table(lr: LogisticRegression, feat_cols: list, top_n: int = 12) -> str:
    """
    Produce a markdown table of top-N positive + top-N negative coefficients
    per STAR class (OvR coefficients).
    """
    lines = []
    for cls_idx, cls_label in enumerate(STAR_LABELS):
        coefs = lr.coef_[cls_idx]
        feat_arr = np.array(feat_cols)
        order = np.argsort(coefs)

        top_pos = order[-top_n:][::-1]
        top_neg = order[:top_n]

        lines.append(f"\n### {cls_label} — top drivers")
        lines.append("")
        lines.append("| Direction | Feature | Coefficient |")
        lines.append("|-----------|---------|-------------|")
        for i in top_pos:
            lines.append(f"| ↑ pushes toward | `{feat_arr[i]}` | +{coefs[i]:.4f} |")
        for i in top_neg:
            lines.append(f"| ↓ pushes away   | `{feat_arr[i]}` | {coefs[i]:.4f} |")

    return '\n'.join(lines)


def run_exp3(roster: pd.DataFrame, flag_cols: list) -> dict:
    log.info("\n" + "=" * 60)
    log.info("  EXP 3 — Interpretability (LightGBM + LogisticRegression)")
    log.info("=" * 60)

    labeled = roster[roster['_y'].notna()].copy()
    labeled = engineer_features(labeled, flag_cols)
    labeled = coerce_flags(labeled, flag_cols)
    y = labeled['_y'].astype(int)

    feat_cols = get_prehire_features(labeled, flag_cols, include_onboarding=False)
    feat_cols = [f for f in feat_cols if f in labeled.columns]
    log.info(f"  Feature cols: {len(feat_cols)} (same as Model A)")

    X = labeled[feat_cols].copy()
    for col in X.select_dtypes(include='object').columns:
        num_try = pd.to_numeric(X[col], errors='coerce')
        if num_try.notna().mean() >= 0.5:
            X[col] = num_try
        else:
            X[col] = X[col].astype('category')

    high_card = [c for c in feat_cols
                 if c in labeled.columns
                 and labeled[c].dtype == object
                 and labeled[c].nunique() > 20]

    # ── LightGBM (reference) ──────────────────────────────────────────────────
    log.info("  [3a] LightGBM reference …")
    lgb_result = run_lgb_experiment(X.copy(), y, high_card, 'exp3_lgb')
    joblib.dump(lgb_result['model'], MODELS_DIR / 'model_exp3_lgb.joblib')
    log.info(f"  Saved → models/model_exp3_lgb.joblib")

    # ── Logistic Regression ───────────────────────────────────────────────────
    log.info("  [3b] Logistic Regression …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    # Target-encode high-card cols first (fold-safe for LR too)
    enc = FoldTargetEncoder(cols=high_card)
    enc.fit(X_tr, y_tr)
    X_tr_enc_pre = enc.transform(X_tr)
    X_te_enc_pre = enc.transform(X_te)

    lr_model, X_tr_lr, lr_metrics = _fit_logreg(X_tr_enc_pre, y_tr, X_te_enc_pre, y_te)
    joblib.dump({'model': lr_model, **lr_metrics}, MODELS_DIR / 'model_exp3_lr.joblib')
    log.info(f"  Saved → models/model_exp3_lr.joblib")

    # Build coefficient feature names (num cols first, then cat cols)
    coef_feat_names = lr_metrics['num_cols'] + lr_metrics['cat_cols']
    coef_table = _coef_table(lr_model, coef_feat_names, top_n=12)

    return {
        'lgb':        lgb_result,
        'lr_acc':     lr_metrics['accuracy'],
        'lr_f1':      lr_metrics['macro_f1'],
        'lr_auc':     lr_metrics['auc'],
        'coef_table': coef_table,
        'n_features': len(feat_cols),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_demo_report(exp1: dict, exp2: dict, exp3: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / 'demo_experiments.md'

    model_a_ref = {
        'label': 'Model A (reference)',
        'n_features': 107, 'accuracy': 0.4764,
        'macro_f1': 0.4040, 'auc': 0.7012, 'baseline': 0.5425,
    }

    rows = [
        model_a_ref,
        exp1,
        exp2,
        exp3['lgb'],
        {
            'label': 'EXP 3 — LogReg (interpretable)',
            'n_features': exp3['n_features'],
            'accuracy': exp3['lr_acc'],
            'macro_f1': exp3['lr_f1'],
            'auc': exp3['lr_auc'],
            'baseline': model_a_ref['baseline'],
        },
    ]

    def _fmt(val, is_auc=False):
        if val is None:
            return 'N/A'
        return f'{val:.4f}'

    lines = [
        "# Demo Experiments — STAR Prediction Comparison",
        "",
        "> Three experiments testing how much demographic signal exists in the data,",
        "> and whether a human-readable model can match gradient boosting.",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Model | Features | Accuracy | vs Baseline | Macro-F1 | AUC (OvR) |",
        "|-------|----------|----------|-------------|----------|-----------|",
    ]

    for r in rows:
        baseline  = r.get('baseline', 0.5425)
        lift      = r['accuracy'] - baseline
        lift_str  = f"{lift:+.4f}"
        acc_emoji = '🟢' if r['accuracy'] > baseline + 0.01 else ('🟡' if r['accuracy'] > baseline else '🔴')
        lines.append(
            f"| {r['label']} | {r['n_features']} | "
            f"{acc_emoji} {r['accuracy']:.4f} | {lift_str} | "
            f"{_fmt(r['macro_f1'])} | {_fmt(r.get('auc'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## What Each Experiment Answers",
        "",
        "### EXP 1 — Pure Demographics",
        "> Features: Education, Region, Province, City, Barangay, xSite, Work At Home,",
        "> Job Grade, Management Level, MSA Fusion, Site Distance, Lat/Lon (~16 features).",
        "> **No survey flags. No org context.**",
        "",
        "This is the floor: how much does *who you are and where you live* predict STAR?",
        "If EXP 1 ≈ baseline, demographics alone carry no actionable signal.",
        "If EXP 1 > baseline, geography / background has independent predictive value.",
        "",
        "### EXP 2 — Full CDM Demographics",
        "> All non-leakage CDM columns: Client, Program, Site, Role Type, Job Title,",
        "> Support Type, Person Status, City, Country, tenure, and all demographic fields.",
        "",
        "This is the org-context ceiling: how much does *where you work and what you do*",
        "add on top of pure demographics? CDM has rich assignment and role data.",
        "If EXP 2 >> EXP 1, role/program placement is a stronger driver than background.",
        "",
        "### EXP 3 — Interpretability",
        "> Same 107 features as Model A, two model types side by side.",
        "> LightGBM shows the performance ceiling; LogisticRegression shows *what drives it*.",
        "",
        "The coefficient table below shows which specific features the model learned",
        "as predictors for each STAR class.",
        "",
        "---",
        "",
        "## EXP 3 — Logistic Regression Coefficients",
        "",
        "> OvR (One-vs-Rest) coefficients after StandardScaler + OrdinalEncoder.",
        "> Positive = feature pushes prediction toward this STAR class.",
        "> Negative = feature pushes prediction away from this STAR class.",
        "",
        exp3['coef_table'],
        "",
        "---",
        "",
        "## Interpretation Guide",
        "",
        "- **Accuracy ≈ baseline** → model learned nothing useful from those features",
        "- **AUC > 0.65** → model ranks agents meaningfully even if raw accuracy is low",
        "- **LogReg AUC close to LightGBM AUC** → relationship is mostly linear; non-linear",
        "  interactions (captured by LightGBM) are not adding much",
        "- **LogReg AUC << LightGBM AUC** → complex interactions matter; coefficients alone",
        "  are a partial picture",
    ]

    path.write_text('\n'.join(lines), encoding='utf-8')
    log.info(f"  Report → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    log.info("Loading source files …")
    sdd1   = load_sdd1()
    cdm    = load_cdm()
    roster = build_master_roster(sdd1, cdm)

    flag_cols = get_flag_cols(roster)
    log.info(f"Binary flag columns detected: {len(flag_cols)}")

    # ── Run experiments ───────────────────────────────────────────────────────
    exp1_result = run_exp1(roster, flag_cols)
    exp2_result = run_exp2(roster, flag_cols)
    exp3_result = run_exp3(roster, flag_cols)

    # ── Write comparison report ───────────────────────────────────────────────
    write_demo_report(exp1_result, exp2_result, exp3_result)

    log.info("")
    log.info("✓ train_model_demo.py  DONE  →  reports/demo_experiments.md + models/")
