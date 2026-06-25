"""
features.py — Feature cleaning, engineering, selection, and fold-safe encoding.
All feature logic lives here; training scripts import utilities from this module.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STAR_MAP, FLAG_PREFIXES,
    PREHIRE_CAT_COLS, PREHIRE_NUM_COLS, ONBOARDING_COLS,
    LEAKAGE_COLS, ALWAYS_DROP,
    EMP_NUM_COLS, STAR_TARGET_COL,
    RANDOM_STATE,
)


# ── 1. Header & value cleaning ────────────────────────────────────────────────

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip non-breaking spaces and extra whitespace from column names."""
    df = df.copy()
    df.columns = [c.replace('\xa0', ' ').strip() for c in df.columns]
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pass:
      - Strip \\xa0 from all column names
      - Replace blank/null-like strings with np.nan
      - Strip leading/trailing whitespace from string values
    """
    df = clean_headers(df)
    blank_vals = {'', ' ', '\xa0', 'N/A', 'n/a', 'NA', 'None', 'none', '#N/A', '-'}
    df = df.replace(blank_vals, np.nan)
    # Vectorised strip on object columns
    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    df[obj_cols] = df[obj_cols].replace('', np.nan)
    return df


# ── 2. Column detection ───────────────────────────────────────────────────────

def detect_emp_col(df: pd.DataFrame) -> str | None:
    """Find the employee-number column by name (case-insensitive, \\xa0-tolerant)."""
    lower_map = {c.lower().replace('\xa0', ' ').strip(): c for c in df.columns}
    for candidate in EMP_NUM_COLS:
        if candidate in lower_map:
            return lower_map[candidate]
    # Broad fallback
    for c in df.columns:
        cl = c.lower().replace('\xa0', ' ')
        if 'employee number' in cl or 'advisor id' in cl:
            return c
    return None


def detect_star_col(df: pd.DataFrame) -> str | None:
    """Find the STAR target column."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in ['latest star rating', 'star rating', 'latest_star_rating']:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def detect_email_col(df: pd.DataFrame) -> str | None:
    """Find Email Address column."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in ['email address', 'email', 'emailaddress']:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


# ── 3. Flag column detection ──────────────────────────────────────────────────

def get_flag_cols(df: pd.DataFrame) -> list:
    """Return all binary flag columns matching FLAG_PREFIXES."""
    flag_cols = []
    for col in df.columns:
        for prefix in FLAG_PREFIXES:
            if col.startswith(prefix):
                flag_cols.append(col)
                break
    return flag_cols


def coerce_flags(df: pd.DataFrame, flag_cols: list) -> pd.DataFrame:
    """Force binary flag columns to 0/1 float; NaN → 0."""
    df = df.copy()
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(0, 1)
    return df


# ── 4. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, flag_cols: list) -> pd.DataFrame:
    """
    Add derived features:
      - cnt_{prefix}: count of active flags per group
      - flag_completeness: fraction of flag cols filled
      - dist_bucket: Site Distance binned
      - tenure_months, tenure_bucket: from Hire Date (CDM only)
    """
    df = df.copy()

    # Count flags per prefix group
    for prefix in FLAG_PREFIXES:
        cols_in_group = [c for c in flag_cols if c.startswith(prefix)]
        if cols_in_group:
            safe = prefix.rstrip('-').replace('-', '_')
            df[f'cnt_{safe}'] = (
                df[cols_in_group]
                .apply(pd.to_numeric, errors='coerce')
                .sum(axis=1)
            )

    # Overall flag completeness
    if flag_cols:
        df['flag_completeness'] = df[flag_cols].notna().mean(axis=1)

    # Site distance bucket
    if 'Site Distance' in df.columns:
        dist = pd.to_numeric(df['Site Distance'], errors='coerce')
        df['dist_bucket'] = pd.cut(
            dist,
            bins=[-1, 5, 15, 30, 60, 9999],
            labels=['<5km', '5-15km', '15-30km', '30-60km', '>60km'],
            right=True,
        ).astype(str)

    # Tenure from Hire Date (CDM only)
    if 'Hire Date' in df.columns:
        hire = pd.to_datetime(df['Hire Date'], errors='coerce')
        ref  = pd.Timestamp('2026-06-23')
        df['tenure_months'] = ((ref - hire).dt.days / 30.44).clip(lower=0)
        df['tenure_bucket'] = pd.cut(
            df['tenure_months'],
            bins=[-1, 3, 6, 12, 24, 60, 9999],
            labels=['<3m', '3-6m', '6-12m', '1-2yr', '2-5yr', '>5yr'],
            right=True,
        ).astype(str)

    return df


# ── 5. Feature set selection ──────────────────────────────────────────────────

def get_prehire_features(df: pd.DataFrame, flag_cols: list,
                         include_onboarding: bool = False) -> list:
    """
    Return ordered, deduplicated feature column list for Model A (resume) or
    A+ (+ onboarding). Guarantees leakage and always-drop cols are excluded.
    """
    leakage_lower  = {c.lower() for c in LEAKAGE_COLS}
    always_drop_lower = {c.lower() for c in ALWAYS_DROP}

    candidates = []

    # Binary flags (the richest pre-hire signal)
    candidates.extend(flag_cols)

    # Pre-hire categorical
    for col in PREHIRE_CAT_COLS:
        if col in df.columns:
            candidates.append(col)

    # Pre-hire numeric
    for col in PREHIRE_NUM_COLS:
        if col in df.columns:
            candidates.append(col)

    # Engineered features
    eng = [c for c in df.columns
           if c.startswith('cnt_') or c in ('dist_bucket', 'flag_completeness')]
    candidates.extend(eng)

    # Onboarding tier (Model A+ only)
    if include_onboarding:
        for col in ONBOARDING_COLS:
            if col in df.columns:
                candidates.append(col)

    # Deduplicate; filter leakage & always-drop
    seen, clean = set(), []
    for f in candidates:
        if f in seen:
            continue
        if f.lower() in leakage_lower or f.lower() in always_drop_lower:
            continue
        seen.add(f)
        clean.append(f)

    return clean


def get_model_b_features(df: pd.DataFrame) -> list:
    """
    Feature set for Model B (KPI → STAR driver model).
    Includes KPI aggregated cols (sc_*, metric_*) + org context.
    Excludes the final STAR / Weighted-PTG aggregates (the target, not the components).
    """
    leakage_lower    = {c.lower() for c in LEAKAGE_COLS}
    always_drop_lower = {c.lower() for c in ALWAYS_DROP}

    # KPI pivot column prefixes produced by joins.py
    kpi_prefixes = (
        'sc_',                     # scorecard aggregates
        'Resolved_PTG_', 'QA_', 'CSAT_', 'Attendance_',   # CNX KPI pivots
        'Resolved PTG_', 'n_kpi_',
    )

    org_cols = {
        'Role Type', 'Support Type', 'Person Status', 'Job Title',
        'Client', 'Program', 'Activity', 'Site', 'Campus', 'City', 'State',
        'Management Level Description', 'tenure_months', 'tenure_bucket',
    }

    candidates = []
    for col in df.columns:
        cl = col.lower()
        if cl in leakage_lower or cl in always_drop_lower:
            continue
        if any(col.startswith(p) for p in kpi_prefixes):
            candidates.append(col)
        elif col in org_cols:
            candidates.append(col)

    return list(dict.fromkeys(candidates))  # preserve order, deduplicate


# ── 6. Fold-safe target encoder ───────────────────────────────────────────────

class FoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoder fit per CV fold to prevent leakage.
    For the final model: fitted on the full training set.

    Usage:
        enc = FoldTargetEncoder(cols=['City', 'GC Province'])
        enc.fit(X_train, y_train)
        X_train_enc = enc.transform(X_train)
    """

    def __init__(self, cols: list = None, smoothing: float = 10.0):
        self.cols      = cols or []
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FoldTargetEncoder':
        self.global_mean_ = float(y.astype(float).mean())
        self.col_maps_    = {}
        for col in self.cols:
            if col not in X.columns:
                continue
            frame = pd.DataFrame({'col': X[col].astype(str), 'y': y.astype(float)})
            stats = frame.groupby('col')['y'].agg(['mean', 'count'])
            smooth = (
                (stats['count'] * stats['mean'] + self.smoothing * self.global_mean_)
                / (stats['count'] + self.smoothing)
            )
            self.col_maps_[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col not in X.columns or col not in self.col_maps_:
                continue
            X[col] = (
                X[col].astype(str)
                    .map(self.col_maps_[col])
                    .fillna(self.global_mean_)
            )
        return X


# ── 7. Coerce feature matrix dtypes ──────────────────────────────────────────

def coerce_dtypes(X: pd.DataFrame, high_card_cols: list) -> pd.DataFrame:
    """
    After target-encoding high-card cols, coerce remaining object/category
    columns to LightGBM-compatible 'category' dtype.
    Numeric-looking object cols are coerced to float.
    """
    X = X.copy()
    for col in X.select_dtypes(include=['object']).columns:
        numeric_try = pd.to_numeric(X[col], errors='coerce')
        if numeric_try.notna().mean() >= 0.5:
            X[col] = numeric_try
        else:
            X[col] = X[col].astype('category')
    for col in X.select_dtypes(include=['category']).columns:
        pass  # LightGBM handles category natively
    return X
