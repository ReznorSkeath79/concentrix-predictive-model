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
from config import KPI_TARGET_COL, KPI_TIER_LABELS


def detect_ptg_col(df: pd.DataFrame) -> str | None:
    """Find Latest Weighted PTG column (case-insensitive, xa0-stripped)."""
    lower_map = {c.lower().replace('\xa0', ' ').strip(): c for c in df.columns}
    for candidate in [KPI_TARGET_COL.lower(), 'latest weighted ptg', 'weighted ptg']:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def build_kpi_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
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
