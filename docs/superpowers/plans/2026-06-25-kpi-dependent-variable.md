# KPI-as-Dependent-Variable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace STAR rating (forced-distribution, bell-curved, ~50% are STAR 1 by policy) with `Latest Weighted PTG` from CDM as the dependent variable, so the model predicts actual KPI performance — not where an agent lands in a politically-assigned ranking.

**Architecture:** Load CDM only (it has `Latest Weighted PTG`); compute tercile-based tier labels from the actual PTG distribution; train a 3-class LightGBM classifier (At-Risk / Developing / High Performer) using the same pre-hire + onboarding feature set as before. The key mechanical change is promoting `Latest Weighted PTG` from `LEAKAGE_COLS` to the target column.

**Tech Stack:** Python 3.12, LightGBM, Optuna, SHAP, pandas, scikit-learn, joblib. All in `requirements.txt` already.

## Global Constraints

- Python 3.12 — no walrus operator issues, use `match` if needed
- All CSVs are in the project root (`E:\work\concentrix\PredictiveModel\`)
- All source modules live in `src/`; import with `sys.path.insert(0, str(Path(__file__).parent))`
- CDM is streamed in 5,000-row chunks (monolithic load OOMs on 316 cols)
- `RANDOM_STATE = 42`, `CV_FOLDS = 5`, `TEST_SIZE = 0.20`, `OPTUNA_TRIALS = 50`
- Models save to `models/`, reports to `reports/`
- Never use STAR columns (`Latest STAR Rating`, `Average STAR Last 3 Months`) as features — they are now irrelevant

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `src/config.py` | Remove `Latest Weighted PTG` from `LEAKAGE_COLS`; add KPI target constants |
| Create | `src/kpi_targets.py` | Compute tercile-based tier labels from `Latest Weighted PTG` |
| Modify | `src/evaluate.py` | Add `class_labels` param to `evaluate_model()` and plot functions |
| Create | `src/train_kpi_model.py` | Full training pipeline: CDM → 3-class KPI tier classifier |

---

## Task 1: Update config.py — promote PTG from leakage to target

**Files:**
- Modify: `src/config.py`

**Interfaces:**
- Produces: `KPI_TARGET_COL`, `KPI_TIER_LABELS`, `N_KPI_CLASSES` constants imported by `kpi_targets.py` and `train_kpi_model.py`
- Produces: `LEAKAGE_COLS` no longer contains `"Latest Weighted PTG"` or its lowercase variant

- [ ] **Step 1: Make the change to config.py**

In `src/config.py`, make these exact changes:

Remove from `LEAKAGE_COLS` (lines 39 and 45):
```
    "Latest Weighted PTG",
    ...
    "latest weighted ptg",
```

Add after the existing `LEAKAGE_COLS` block:

```python
# ── KPI target (new dependent variable — replaces STAR) ───────────────────────
KPI_TARGET_COL  = "Latest Weighted PTG"
KPI_TIER_LABELS = ["At-Risk", "Developing", "High Performer"]
N_KPI_CLASSES   = 3

# Probability head indices for the 3-class KPI model
KPI_AT_RISK_CLASS    = 0   # tercile 1 — PTG below p33
KPI_DEVELOPING_CLASS = 1   # tercile 2 — PTG p33..p67
KPI_HIGH_PERF_CLASS  = 2   # tercile 3 — PTG above p67
```

Keep `STAR_MAP`, `STAR_LABELS`, etc. in place — the old STAR models still load and predict; don't break backward compat.

- [ ] **Step 2: Verify the import works**

```bash
cd "E:\work\concentrix\PredictiveModel"
python -c "from src.config import KPI_TARGET_COL, KPI_TIER_LABELS, N_KPI_CLASSES; print(KPI_TARGET_COL, KPI_TIER_LABELS, N_KPI_CLASSES)"
```

Expected output:
```
Latest Weighted PTG ['At-Risk', 'Developing', 'High Performer'] 3
```

- [ ] **Step 3: Verify PTG is NOT in leakage anymore**

```bash
python -c "from src.config import LEAKAGE_COLS; ptg_in = any('weighted ptg' in c.lower() for c in LEAKAGE_COLS); print('PTG in leakage:', ptg_in)"
```

Expected:
```
PTG in leakage: False
```

- [ ] **Step 4: Commit**

```bash
git add src/config.py
git commit -m "config: promote Latest Weighted PTG from leakage to KPI target; add KPI_TARGET_COL, KPI_TIER_LABELS, N_KPI_CLASSES"
```

---

## Task 2: Create src/kpi_targets.py — tercile-based KPI tier builder

**Files:**
- Create: `src/kpi_targets.py`
- Test: run inline (no separate test file — verify with print statements per step)

**Interfaces:**
- Consumes: `df: pd.DataFrame` — cleaned CDM DataFrame (output of `clean_df(load_cdm())`)
- Produces: `build_kpi_labels(df) -> tuple[pd.Series, pd.Series, dict]`
  - `y_continuous`: `pd.Series[float]`, index-aligned to `df`, NaN where PTG missing
  - `y_tier`: `pd.Series[int]`, values `{0, 1, 2}`, NaN where PTG missing
  - `tier_info`: `dict` with keys `cutpoints` (p33/p67 floats), `labels` (list[str]), `counts` (dict)

- [ ] **Step 1: Create `src/kpi_targets.py`**

```python
"""
kpi_targets.py — Builds KPI-based tier labels from Latest Weighted PTG.

Replaces STAR as the dependent variable. Tiers are tercile-based (p33, p67)
computed from the actual PTG distribution — no forced bell curve.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import KPI_TARGET_COL, KPI_TIER_LABELS, N_KPI_CLASSES


def detect_ptg_col(df: pd.DataFrame) -> str | None:
    """Find Latest Weighted PTG column (case-insensitive, xa0-stripped)."""
    lower_map = {c.lower().replace('\xa0', ' ').strip(): c for c in df.columns}
    for candidate in [KPI_TARGET_COL.lower(), 'latest weighted ptg', 'weighted ptg']:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def build_kpi_labels(df: pd.DataFrame) -> tuple:
    """
    Compute tercile-based KPI tier labels from Latest Weighted PTG.

    Returns
    -------
    y_continuous : pd.Series[float]   raw PTG values, NaN where missing
    y_tier       : pd.Series[Int64]   tier {0=At-Risk, 1=Developing, 2=High Performer}, NaN where missing
    tier_info    : dict               cutpoints, label map, counts, pct_labeled
    """
    ptg_col = detect_ptg_col(df)
    if ptg_col is None:
        raise ValueError(
            f"Cannot find '{KPI_TARGET_COL}' in DataFrame. "
            f"Available cols: {list(df.columns[:10])} ..."
        )

    y_continuous = pd.to_numeric(df[ptg_col], errors='coerce').reset_index(drop=True)
    valid = y_continuous.dropna()

    if len(valid) < 30:
        raise ValueError(
            f"Only {len(valid)} non-null PTG values — need ≥30 to compute tercile cutoffs."
        )

    p33 = float(valid.quantile(1 / 3))
    p67 = float(valid.quantile(2 / 3))

    y_tier = pd.cut(
        y_continuous,
        bins=[-np.inf, p33, p67, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype('Int64')   # pandas nullable int — preserves NaN

    counts = {int(k): int(v) for k, v in y_tier.value_counts().sort_index().items()}
    pct_labeled = float(y_tier.notna().mean())

    tier_info = {
        'cutpoints': {'p33': p33, 'p67': p67},
        'labels': {0: KPI_TIER_LABELS[0], 1: KPI_TIER_LABELS[1], 2: KPI_TIER_LABELS[2]},
        'counts': counts,
        'pct_labeled': pct_labeled,
        'n_labeled': int(y_tier.notna().sum()),
        'n_total': len(df),
    }

    return y_continuous, y_tier, tier_info
```

- [ ] **Step 2: Verify against a slice of the CDM file**

```bash
cd "E:\work\concentrix\PredictiveModel"
python -c "
import pandas as pd
from src.features import clean_df
from src.kpi_targets import build_kpi_labels

chunks = []
reader = pd.read_csv('CDM Raw-smaller.csv', dtype=str, encoding='utf-8-sig',
                     on_bad_lines='skip', chunksize=5000)
for chunk in reader:
    chunks.append(chunk)
df = clean_df(pd.concat(chunks, ignore_index=True))

y_cont, y_tier, info = build_kpi_labels(df)
print('PTG range:', y_cont.min(), '-', y_cont.max())
print('p33:', info['cutpoints']['p33'])
print('p67:', info['cutpoints']['p67'])
print('Tier counts:', info['counts'])
print('Labeled:', info['n_labeled'], '/', info['n_total'])
"
```

Expected: 3 counts each containing roughly equal thirds of labeled rows; p33 < p67; no crash.

- [ ] **Step 3: Commit**

```bash
git add src/kpi_targets.py
git commit -m "feat: add kpi_targets.py — tercile-based KPI tier labels from Latest Weighted PTG"
```

---

## Task 3: Update src/evaluate.py — accept custom class labels

**Files:**
- Modify: `src/evaluate.py`

**Context:** `evaluate_model()` currently uses `STAR_LABELS` from config to title confusion matrix axes and SHAP plots. The KPI model needs `["At-Risk", "Developing", "High Performer"]` instead.

**Interfaces:**
- Consumes: `class_labels: list[str] | None = None` new optional parameter
- Produces: same return value; if `class_labels` is None, falls back to `STAR_LABELS` (backward compat)

- [ ] **Step 1: Read the current evaluate.py signature**

```bash
cd "E:\work\concentrix\PredictiveModel"
python -c "import inspect; from src.evaluate import evaluate_model; print(inspect.signature(evaluate_model))"
```

Note the exact current signature so you can match it precisely in Step 2.

- [ ] **Step 2: Add `class_labels` parameter to `evaluate_model()`**

Find the line in `src/evaluate.py` that defines `evaluate_model(...)`. Add `class_labels: list = None` as the last parameter. Inside the function, add at the top:

```python
from config import STAR_LABELS as _DEFAULT_LABELS
_labels = class_labels if class_labels is not None else _DEFAULT_LABELS
```

Then replace every reference to `STAR_LABELS` inside that function with `_labels`.

Do the same for any helper function that receives labels for plot titles / confusion matrix tick labels.

- [ ] **Step 3: Verify backward compat — old call still works**

```bash
python -c "
import inspect
from src.evaluate import evaluate_model
sig = inspect.signature(evaluate_model)
print('Parameters:', list(sig.parameters.keys()))
print('class_labels default:', sig.parameters.get('class_labels').default)
"
```

Expected: `class_labels` appears in the parameter list with default `None`.

- [ ] **Step 4: Commit**

```bash
git add src/evaluate.py
git commit -m "evaluate: add class_labels param to evaluate_model — enables KPI tier labels without breaking STAR models"
```

---

## Task 4: Create src/train_kpi_model.py — full KPI tier training pipeline

**Files:**
- Create: `src/train_kpi_model.py`
- Outputs: `models/model_kpi_a.joblib`, `reports/kpi_metrics.md`

**Interfaces:**
- Consumes: `CDM_PATH` (CDM only — SDD1 has no PTG), `build_kpi_labels()` from `kpi_targets`, `get_prehire_features()` from `features`, `evaluate_model()` from `evaluate`
- Produces:
  - `models/model_kpi_a.joblib` — dict: `{model, encoder, feature_cols, flag_cols, high_card_cols, tier_info, best_params, metrics}`
  - `models/schema_kpi.json` — Phase-2 schema with KPI tier output spec
  - `reports/kpi_metrics.md` — accuracy, AUC, F1, confusion for 3-class KPI model

- [ ] **Step 1: Create `src/train_kpi_model.py`**

```python
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
from evaluate import evaluate_model, write_metrics_md

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

    write_metrics_md([metrics])
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


if __name__ == '__main__':
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    run()
```

- [ ] **Step 2: Dry-run — verify imports only**

```bash
cd "E:\work\concentrix\PredictiveModel"
python -c "import src.train_kpi_model; print('imports OK')"
```

Expected: `imports OK` (no errors)

- [ ] **Step 3: Run the full training pipeline**

```bash
cd "E:\work\concentrix\PredictiveModel"
python src/train_kpi_model.py 2>&1 | tee reports/kpi_training_log.txt
```

Expected log output (watch for these lines):
```
  CDM: [N]:,} rows × [N] cols
  Tercile cutpoints → p33=[X]  p67=[Y]
  Labeled [N]/[N] ([%]) agents have PTG
  Majority baseline accuracy: [X]
  LGB default test accuracy: [X]
  Optuna (50 trials) …
  Best CV macro-F1: [X]
  Tuned test accuracy: [X]
  Model saved → models/model_kpi_a.joblib
  Schema saved → models/schema_kpi.json
```

No crash, no `KeyError`, no `MemoryError`.

- [ ] **Step 4: Verify model artifact and Phase-2 readiness**

```bash
python -c "
import joblib, json, pandas as pd
from src.features import coerce_dtypes

artifact = joblib.load('models/model_kpi_a.joblib')
schema   = json.loads(open('models/schema_kpi.json').read())

# Build dummy row from schema
example = {}
for col, info in schema['features'].items():
    if info['type'] == 'binary':       example[col] = 0
    elif info['type'] == 'numeric':    example[col] = info['range'][0]
    else:                              example[col] = info['values'][0] if info['values'] else None

X_ex  = pd.DataFrame([example])
enc   = artifact['encoder']
X_enc = coerce_dtypes(enc.transform(X_ex), artifact['high_card_cols'])
for col in X_enc.select_dtypes(include=['object']).columns:
    X_enc[col] = X_enc[col].astype('category')

pred  = int(artifact['model'].predict(X_enc)[0])
proba = artifact['model'].predict_proba(X_enc)[0]
label = artifact['kpi_tier_labels'][pred]

print(f'Predicted tier: {pred} ({label})')
print(f'P(At-Risk): {proba[0]:.3f}')
print(f'P(Developing): {proba[1]:.3f}')
print(f'P(High Performer): {proba[2]:.3f}')
print('Phase-2 readiness: PASS')
"
```

Expected: 3 probabilities sum to ~1.0, label is one of `['At-Risk', 'Developing', 'High Performer']`, no crash.

- [ ] **Step 5: Commit**

```bash
git add src/train_kpi_model.py
git commit -m "feat: train_kpi_model.py — 3-class KPI tier model (replaces STAR); tercile cutpoints from Latest Weighted PTG"
```

---

## Self-Review

**Spec coverage:**
- [x] STAR removed as dependent variable — `train_kpi_model.py` never reads `_star` or `STAR_MAP`
- [x] KPI (Weighted PTG) used as target — `build_kpi_labels()` in Task 2
- [x] No forced distribution — tercile cutpoints computed from actual PTG distribution
- [x] Pre-hire features preserved — `get_prehire_features(include_onboarding=True)` in Task 4
- [x] Model artifact saved with Phase-2 schema — `model_kpi_a.joblib` + `schema_kpi.json`
- [x] Backward compat — old STAR models untouched; `evaluate_model` still works without `class_labels`

**Placeholder scan:** None found — all code blocks are complete.

**Type consistency:**
- `build_kpi_labels()` returns `(pd.Series[float], pd.Series[Int64], dict)` — matches usage in `prep_Xy()`
- `tier_info['cutpoints']` is `{'p33': float, 'p67': float}` — matches schema builder
- `KPI_TIER_LABELS: list[str]` — used in `evaluate_model(class_labels=KPI_TIER_LABELS)` which now accepts `list[str]`

---

## Expected Results

The KPI model should outperform the old STAR model meaningfully because:
1. The target is a continuous performance score (tercile-cut), not forced ranking
2. The labeled set expands — more CDM rows have PTG values than have valid STAR ratings
3. The signal is cleaner — PTG is a direct metric, not a policy outcome

**Targets to watch:**
- AUC > 0.75 (vs STAR model's 0.72 on same feature set)
- Macro-F1 > 0.45
- Classes should be roughly balanced (tercile design guarantees ~33% each)
- Majority baseline = 33.3% (vs old 54% on STAR 1 majority) — higher baseline accuracy now has real meaning
