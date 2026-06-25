"""
train_model_a.py — Model A (resume/pre-hire) and A+ (onboarding) pipeline.

Union: Sample Demo Data 1 + CDM Raw → master roster (~116k agents)
Target: STAR {1,3,4,5} → ordinal {0,1,2,3}
Saves: models/model_a.joblib + models/schema_a.json (Phase-2 artifacts)
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
    SDD1_PATH, CDM_PATH, MODELS_DIR, REPORTS_DIR,
    STAR_MAP, INV_STAR_MAP, N_CLASSES, STAR_LABELS,
    RANDOM_STATE, CV_FOLDS, TEST_SIZE, OPTUNA_TRIALS,
    LEAKAGE_COLS, ALWAYS_DROP,
)
from features import (
    clean_df, detect_emp_col, detect_star_col, detect_email_col,
    get_flag_cols, coerce_flags, engineer_features,
    get_prehire_features, FoldTargetEncoder, coerce_dtypes,
)
from evaluate import evaluate_model, write_metrics_md

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Load + clean source files
# ═══════════════════════════════════════════════════════════════════════════════

def load_sdd1() -> pd.DataFrame:
    """Load SDD1 in 5 000-row chunks (monolithic load OOMs the C parser on 135 cols / 67MB)."""
    log.info(f"Loading SDD1  →  {SDD1_PATH}")
    chunks = []
    reader = pd.read_csv(
        SDD1_PATH, dtype=str, encoding='utf-8-sig',
        on_bad_lines='skip', chunksize=5_000,
    )
    for chunk in reader:
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df = clean_df(df)
    log.info(f"  SDD1: {len(df):,} rows × {len(df.columns)} cols")
    return df


def load_cdm() -> pd.DataFrame:
    """Load CDM in 5 000-row chunks (monolithic load OOMs the C parser on 316 cols)."""
    log.info(f"Loading CDM   →  {CDM_PATH}")
    chunks = []
    reader = pd.read_csv(
        CDM_PATH, dtype=str, encoding='utf-8-sig',
        on_bad_lines='skip', chunksize=5_000,
    )
    for chunk in reader:
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df = clean_df(df)
    log.info(f"  CDM:  {len(df):,} rows × {len(df.columns)} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Build master roster (union SDD1 + CDM, freshest STAR label wins)
# ═══════════════════════════════════════════════════════════════════════════════

def build_master_roster(sdd1: pd.DataFrame, cdm: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      1. Identify emp# column in each file.
      2. Normalise emp# strings; drop blank-ID rows.
      3. Union; when an agent appears in both, keep SDD1 row (fresher STAR).
      4. Left-join CDM-only columns (Hire Date, Role Type, Email …) onto roster.
    """
    log.info("Building master roster …")

    sdd1_emp = detect_emp_col(sdd1)
    cdm_emp  = detect_emp_col(cdm)
    if not sdd1_emp:
        raise ValueError("SDD1: cannot find employee-number column")
    if not cdm_emp:
        raise ValueError("CDM: cannot find employee-number column")

    log.info(f"  Emp cols  →  SDD1='{sdd1_emp}'  CDM='{cdm_emp}'")

    sdd1 = sdd1.copy()
    cdm  = cdm.copy()

    def _norm_emp(df, col):
        return df[col].astype(str).str.replace('\xa0', '', regex=False).str.strip()

    sdd1['_emp_num'] = _norm_emp(sdd1, sdd1_emp)
    cdm ['_emp_num'] = _norm_emp(cdm,  cdm_emp)

    # Drop blank / nan employee numbers
    for name, df in [('SDD1', sdd1), ('CDM', cdm)]:
        before = len(df)
        mask = df['_emp_num'].notna() & (df['_emp_num'] != '') & (df['_emp_num'].str.lower() != 'nan')
        df.drop(df[~mask].index, inplace=True)
        log.info(f"  {name}: dropped {before - len(df):,} blank-ID rows → {len(df):,} remain")

    # Standardise STAR column to '_star'
    for name, df in [('SDD1', sdd1), ('CDM', cdm)]:
        star_col = detect_star_col(df)
        if star_col:
            df.rename(columns={star_col: '_star'}, inplace=True)
        if '_star' not in df.columns:
            df['_star'] = np.nan

    sdd1['_source'] = 'sdd1'
    cdm ['_source'] = 'cdm'

    # Shared columns (present in both files)
    shared = ['_emp_num', '_source', '_star'] + sorted(
        set(sdd1.columns) & set(cdm.columns) - {'_emp_num', '_source', '_star'}
    )

    sdd1_sub = sdd1[[c for c in shared if c in sdd1.columns]].copy()
    cdm_sub  = cdm [[c for c in shared if c in cdm.columns]].copy()

    union = pd.concat([sdd1_sub, cdm_sub], ignore_index=True)

    # Deduplicate: SDD1 rows have priority (fresher labels)
    union['_ord'] = (union['_source'] == 'cdm').astype(int)
    union = (union
             .sort_values(['_emp_num', '_ord'])
             .drop_duplicates('_emp_num', keep='first')
             .drop(columns=['_ord']))

    log.info(f"  Union (dedup): {len(union):,} unique agents")

    # Append CDM-only columns (HR enrichment)
    cdm_only_cols = ['_emp_num'] + [
        c for c in cdm.columns
        if c not in sdd1.columns and c not in ('_emp_num', '_source', '_star')
    ]
    cdm_extras = (cdm[[c for c in cdm_only_cols if c in cdm.columns]]
                  .drop_duplicates('_emp_num'))
    roster = union.merge(cdm_extras, on='_emp_num', how='left', suffixes=('', '_cdm'))

    # Parse and map STAR
    roster['_star'] = pd.to_numeric(roster['_star'], errors='coerce')
    valid  = roster['_star'].isin([1, 3, 4, 5])
    roster['_y'] = np.where(valid, roster['_star'].map(STAR_MAP), np.nan)

    star_dist = roster.loc[valid, '_star'].value_counts().sort_index()
    log.info(f"  STAR labeled: {valid.sum():,}/{len(roster):,} ({valid.mean():.1%})")
    log.info(f"  STAR dist:\n{star_dist.to_string()}")
    log.info(f"  Roster final: {len(roster):,} agents × {len(roster.columns)} cols")

    return roster


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Feature preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prep_Xy(roster: pd.DataFrame, flag_cols: list,
            include_onboarding: bool = False):
    """Return (X, y) for the labeled subset with all features engineered."""
    labeled = roster[roster['_y'].notna()].copy()
    y = labeled['_y'].astype(int)

    labeled = engineer_features(labeled, flag_cols)
    labeled = coerce_flags(labeled, flag_cols)

    feat_cols = get_prehire_features(labeled, flag_cols,
                                     include_onboarding=include_onboarding)
    feat_cols = [f for f in feat_cols if f in labeled.columns]
    log.info(f"  Feature cols: {len(feat_cols)} {'(+onboarding)' if include_onboarding else '(pre-hire only)'}")

    X = labeled[feat_cols].copy()

    # Coerce dtypes
    for col in X.select_dtypes(include='object').columns:
        num_try = pd.to_numeric(X[col], errors='coerce')
        if num_try.notna().mean() >= 0.5:
            X[col] = num_try
        else:
            X[col] = X[col].astype('category')

    return X, y, feat_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  LightGBM Optuna objective
# ═══════════════════════════════════════════════════════════════════════════════

def lgb_objective(trial, X_tr, y_tr, n_folds, high_card_cols):
    """CV macro-F1 objective for Optuna. Fold-safe target encoding within each fold."""
    params = {
        'n_estimators':     trial.suggest_int  ('n_estimators', 200, 1000),
        'max_depth':        trial.suggest_int  ('max_depth', 3, 8),
        'num_leaves':       trial.suggest_int  ('num_leaves', 15, 127),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'min_child_samples':trial.suggest_int  ('min_child_samples', 10, 100),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'objective':        'multiclass',
        'num_class':        N_CLASSES,
        'class_weight':     'balanced',
        'random_state':     RANDOM_STATE,
        'n_jobs':           -1,
        'verbose':          -1,
    }

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        X_f_tr, X_f_val = X_tr.iloc[tr_idx].copy(), X_tr.iloc[val_idx].copy()
        y_f_tr, y_f_val = y_tr.iloc[tr_idx],       y_tr.iloc[val_idx]

        # Per-fold target encoding
        enc = FoldTargetEncoder(cols=high_card_cols)
        enc.fit(X_f_tr, y_f_tr)
        X_f_tr  = enc.transform(X_f_tr)
        X_f_val = enc.transform(X_f_val)

        # Ensure category dtype
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Train final LightGBM on full train split
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_final(X_tr, y_tr, X_val, y_val, params: dict) -> lgb.LGBMClassifier:
    final_params = {
        **params,
        'objective':    'multiclass',
        'num_class':    N_CLASSES,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs':       -1,
        'verbose':      -1,
    }
    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )
    return model


DEFAULT_PARAMS = {
    'n_estimators': 500, 'max_depth': 6, 'num_leaves': 63,
    'learning_rate': 0.05, 'min_child_samples': 20,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Main training driver
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_a(roster: pd.DataFrame,
                  include_onboarding: bool = False) -> dict:
    """
    Full pipeline for Model A (include_onboarding=False) or
    Model A+ (include_onboarding=True).
    """
    label = "model_a_plus" if include_onboarding else "model_a"
    bar   = "=" * 60
    log.info(f"\n{bar}\n  {label.upper()}\n{bar}")

    flag_cols = get_flag_cols(roster)
    log.info(f"  Binary flag columns detected: {len(flag_cols)}")

    X, y, feat_cols = prep_Xy(roster, flag_cols, include_onboarding)
    log.info(f"  Labeled set: {len(X):,} agents  |  {len(X.columns)} features")
    log.info(f"  Class dist:  {pd.Series(y).value_counts().sort_index().to_dict()}")

    # ── Stratified hold-out ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"  Train={len(X_train):,}  Test={len(X_test):,}")

    # High-cardinality cols → target encoding
    high_card = [c for c in X_train.select_dtypes(include=['object', 'category']).columns
                 if X_train[c].nunique() > 20]
    log.info(f"  High-card target-encode cols ({len(high_card)}): {high_card[:5]}{'…' if len(high_card)>5 else ''}")

    # Global encoder (for final model)
    enc = FoldTargetEncoder(cols=high_card)
    enc.fit(X_train, y_train)
    X_train_e = coerce_dtypes(enc.transform(X_train.copy()), high_card)
    X_test_e  = coerce_dtypes(enc.transform(X_test.copy()),  high_card)

    # ── Majority baseline ────────────────────────────────────────────────────
    log.info("\n  [1/4] Majority baseline")
    maj = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
    maj.fit(X_train_e, y_train)
    maj_acc = accuracy_score(y_test, maj.predict(X_test_e))
    log.info(f"    Accuracy: {maj_acc:.4f}")

    # ── LightGBM (default params) ────────────────────────────────────────────
    log.info("\n  [2/4] LightGBM default params")
    X_sub, X_val2, y_sub, y_val2 = train_test_split(
        X_train_e, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    lgb_base = train_lgb_final(X_sub, y_sub, X_val2, y_val2, DEFAULT_PARAMS)
    base_acc = accuracy_score(y_test, lgb_base.predict(X_test_e))
    log.info(f"    Test accuracy: {base_acc:.4f}")

    # ── Optuna hyper-parameter search ────────────────────────────────────────
    log.info(f"\n  [3/4] Optuna search ({OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV) …")
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

    lgb_tuned = train_lgb_final(X_sub, y_sub, X_val2, y_val2, best_params)
    tuned_acc = accuracy_score(y_test, lgb_tuned.predict(X_test_e))
    log.info(f"    Tuned test accuracy: {tuned_acc:.4f}")

    # ── CatBoost challenger ───────────────────────────────────────────────────
    log.info("\n  [4/4] CatBoost challenger")
    cb_model, cb_acc = None, -1.0
    try:
        from catboost import CatBoostClassifier
        X_cb_tr = X_train_e.copy()
        X_cb_te = X_test_e.copy()
        # CatBoost needs string categoricals
        cat_idx = []
        for i, col in enumerate(X_cb_tr.columns):
            if X_cb_tr[col].dtype.name in ('category', 'object'):
                X_cb_tr[col] = X_cb_tr[col].astype(str)
                X_cb_te[col] = X_cb_te[col].astype(str)
                cat_idx.append(i)

        cb_model = CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6,
            loss_function='MultiClass', eval_metric='Accuracy',
            random_seed=RANDOM_STATE, verbose=0,
            early_stopping_rounds=50,
        )
        cb_model.fit(
            X_cb_tr, y_train,
            eval_set=(X_cb_te, y_test),
            cat_features=cat_idx,
        )
        cb_acc = accuracy_score(y_test, cb_model.predict(X_cb_te).flatten().astype(int))
        log.info(f"    CatBoost test accuracy: {cb_acc:.4f}")
    except Exception as e:
        log.warning(f"    CatBoost skipped: {e}")

    # ── Select best model ────────────────────────────────────────────────────
    candidates = {'lgb_base': base_acc, 'lgb_tuned': tuned_acc}
    if cb_model:
        candidates['catboost'] = cb_acc
    best_name = max(candidates, key=candidates.get)
    best_acc  = candidates[best_name]
    log.info(f"\n  ✓ Best model: {best_name}  |  Test accuracy: {best_acc:.4f}")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    shap_vals, shap_X = None, None
    try:
        import shap
        explainer = shap.TreeExplainer(lgb_tuned)
        shap_sample = X_test_e.iloc[:min(500, len(X_test_e))]
        shap_vals = explainer.shap_values(shap_sample)
        shap_X    = shap_sample
    except Exception as e:
        log.warning(f"  SHAP skipped: {e}")

    # ── Full evaluation ───────────────────────────────────────────────────────
    metrics = evaluate_model(
        lgb_tuned, X_test_e, y_test.values,
        model_name=label,
        shap_values=shap_vals,
        shap_X=shap_X,
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        'model':        lgb_tuned,
        'encoder':      enc,
        'feature_cols': list(X_train.columns),
        'flag_cols':    flag_cols,
        'high_card_cols': high_card,
        'star_map':     STAR_MAP,
        'inv_star_map': INV_STAR_MAP,
        'star_labels':  STAR_LABELS,
        'best_params':  best_params,
        'metrics':      {k: v for k, v in metrics.items() if k != 'classification_report'},
    }
    model_path = MODELS_DIR / f'{label}.joblib'
    joblib.dump(artifact, model_path)
    log.info(f"  Model saved  →  {model_path}")

    # Schema for Phase 2 (pre-hire model only)
    if not include_onboarding:
        schema = _build_schema(X_train, flag_cols, enc)
        schema_path = MODELS_DIR / 'schema_a.json'
        schema_path.write_text(json.dumps(schema, indent=2), encoding='utf-8')
        log.info(f"  Schema saved →  {schema_path}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Phase-2 schema builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_schema(X: pd.DataFrame, flag_cols: list,
                  encoder: FoldTargetEncoder) -> dict:
    schema = {
        'version':     '1.0',
        'description': 'Model A (pre-hire) input schema — Phase 2 upload site',
        'star_output': {
            'values':      [1, 3, 4, 5],
            'ordinal_map': STAR_MAP,
            'labels':      STAR_LABELS,
        },
        'features': {},
    }
    for col in X.columns:
        dtype = str(X[col].dtype)
        if col in flag_cols:
            schema['features'][col] = {
                'type': 'binary', 'values': [0, 1],
            }
        elif 'float' in dtype or 'int' in dtype:
            lo = float(X[col].min()) if X[col].notna().any() else 0.0
            hi = float(X[col].max()) if X[col].notna().any() else 1.0
            schema['features'][col] = {'type': 'numeric', 'range': [lo, hi]}
        else:
            vals = sorted(X[col].dropna().astype(str).unique().tolist())
            schema['features'][col] = {'type': 'categorical', 'values': vals[:50]}
    return schema


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Phase-2 readiness check
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_readiness_check():
    log.info("\n── Phase-2 Readiness Check ──────────────────────────────────")
    try:
        artifact    = joblib.load(MODELS_DIR / 'model_a.joblib')
        schema      = json.loads((MODELS_DIR / 'schema_a.json').read_text())

        # Build dummy example row from schema
        example = {}
        for col, info in schema['features'].items():
            if info['type'] == 'binary':
                example[col] = 0
            elif info['type'] == 'numeric':
                example[col] = info['range'][0]
            else:
                vals = info.get('values', [None])
                example[col] = vals[0] if vals else None

        X_ex = pd.DataFrame([example])
        enc  = artifact['encoder']
        X_ex = coerce_dtypes(enc.transform(X_ex), artifact['high_card_cols'])
        for col in X_ex.select_dtypes(include=['object']).columns:
            X_ex[col] = X_ex[col].astype('category')

        model = artifact['model']
        pred  = int(model.predict(X_ex)[0])
        proba = model.predict_proba(X_ex)[0]

        star_pred   = INV_STAR_MAP[pred]
        at_risk     = float(proba[0])
        top_perf    = float(proba[2] + proba[3])

        log.info(f"  Example prediction:")
        log.info(f"    Predicted STAR : {star_pred}")
        log.info(f"    P(STAR=1/At-risk)  : {at_risk:.3f}")
        log.info(f"    P(STAR∈{{4,5}}/Top) : {top_perf:.3f}")
        log.info("  ✓ Phase-2 readiness: PASS")
        return True
    except Exception as e:
        log.error(f"  ✗ Phase-2 readiness FAILED: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    sdd1   = load_sdd1()
    cdm    = load_cdm()
    roster = build_master_roster(sdd1, cdm)

    all_metrics = []

    m_a    = train_model_a(roster, include_onboarding=False)
    all_metrics.append(m_a)

    m_aplus = train_model_a(roster, include_onboarding=True)
    all_metrics.append(m_aplus)

    write_metrics_md(all_metrics)
    phase2_readiness_check()

    log.info("\n✓ train_model_a.py  DONE  →  reports/ + models/")
