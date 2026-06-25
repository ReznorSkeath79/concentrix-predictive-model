"""
joins.py — Chunked aggregation of large KPI + scorecard files; join to CDM roster.

CNX KPIs (1.1 GB, 4.3M rows) and current_data (547 MB, 1M rows) are streamed
in chunks so they never fully load into RAM.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    KPI_PATH, SCORECARD_PATH,
    KPI_ADVISOR_ID, KPI_MONTH, KPI_METRIC, KPI_SCORE,
    KPI_PTG, KPI_WEIGHTED_PTG, KPI_WEIGHTS,
    SCORECARD_EMAIL, SCORECARD_METRIC, SCORECARD_DATE,
    SCORECARD_PTG, SCORECARD_KPI_PASSED,
    SCORECARD_CONSEC_FAIL, SCORECARD_MET, SCORECARD_FAIL,
    SCORECARD_AVG_PTG, SCORECARD_RANK, SCORECARD_QUARTILE,
    KPI_CHUNK_SIZE, SCORECARD_CHUNK_SIZE,
)

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return first matching column (case-insensitive, \\xa0-stripped)."""
    lower_map = {c.lower().replace('\xa0', ' ').strip(): c for c in df.columns}
    for name in candidates:
        key = name.lower().replace('\xa0', ' ').strip()
        if key in lower_map:
            return lower_map[key]
    return None


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _norm_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.replace('\xa0', ' ').strip() for c in df.columns]
    return df


# ── CNX KPI aggregation ───────────────────────────────────────────────────────

def aggregate_kpis(kpi_path: Path = KPI_PATH) -> pd.DataFrame:
    """
    Stream CNX KPI file in chunks using vectorized groupby.
    Returns per-advisor DataFrame with:
      - 7 overall aggregates: kpi_avg_score/ptg/wptg, kpi_latest_ptg,
        kpi_pct_above_target, n_kpi_months, n_kpi_metrics
      - Top-30 metric per-metric avg PTG/wPTG/score columns
    Capped at TOP_N metrics to avoid OOM from thousands of unique metric names.
    """
    log.info(f"[joins] Streaming KPI file: {kpi_path}")

    frames = []
    total_rows = 0
    chunks_read = 0

    reader = pd.read_csv(
        kpi_path,
        chunksize=KPI_CHUNK_SIZE,
        dtype=str,
        low_memory=False,
        encoding='utf-8-sig',
    )
    for chunk in reader:
        chunk = _norm_headers(chunk)
        chunk = chunk.replace({'': np.nan, '\xa0': np.nan})

        adv_col    = _find_col(chunk, [KPI_ADVISOR_ID, 'Advisor ID', 'advisor id'])
        metric_col = _find_col(chunk, [KPI_METRIC, 'Metric', 'metric'])
        score_col  = _find_col(chunk, [KPI_SCORE, 'Score', 'score'])
        ptg_col    = _find_col(chunk, [KPI_PTG, 'PTG', 'ptg'])
        wptg_col   = _find_col(chunk, [KPI_WEIGHTED_PTG, 'Weighted PTG', 'weighted ptg'])
        month_col  = _find_col(chunk, [KPI_MONTH, 'Month year', 'month year'])

        if not adv_col or not metric_col:
            log.warning(f"  KPI chunk #{chunks_read}: missing key columns — skipping")
            chunks_read += 1
            continue

        # Select and rename only needed columns — avoids itertuples identifier renaming
        col_map = {adv_col: '_adv', metric_col: '_metric'}
        if score_col:  col_map[score_col]  = '_score'
        if ptg_col:    col_map[ptg_col]    = '_ptg'
        if wptg_col:   col_map[wptg_col]   = '_wptg'
        if month_col:  col_map[month_col]  = '_month'

        sub = chunk[list(col_map.keys())].rename(columns=col_map).copy()
        sub['_adv'] = sub['_adv'].astype(str).str.strip()
        sub = sub[sub['_adv'].str.lower().notna() & (sub['_adv'] != '') & (sub['_adv'].str.lower() != 'nan')]

        for c in ['_score', '_ptg', '_wptg']:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')

        total_rows += len(chunk)
        chunks_read += 1
        if chunks_read % 10 == 0:
            log.info(f"  ... KPI chunk {chunks_read}, {total_rows:,} rows processed")

        if not sub.empty:
            frames.append(sub)

    log.info(f"  KPI: {total_rows:,} rows streamed across {chunks_read} chunks")

    if not frames:
        log.warning("  KPI: no data accumulated — returning empty pivot")
        return pd.DataFrame(columns=['Advisor ID'])

    all_kpi = pd.concat(frames, ignore_index=True)
    unique_advisors = all_kpi['_adv'].nunique()
    log.info(f"  KPI pivot: building from {unique_advisors:,} unique advisors")

    # ── Overall aggregates (7 dense features) ─────────────────────────────────
    grp = all_kpi.groupby('_adv', sort=False)

    agg_funcs = {c: ['mean'] for c in ['_score', '_ptg', '_wptg'] if c in all_kpi.columns}
    result = grp.agg(agg_funcs) if agg_funcs else pd.DataFrame(index=all_kpi['_adv'].unique())
    result.columns = ['_'.join(c).strip('_') for c in result.columns]
    result = result.rename(columns={
        '_score_mean': 'kpi_avg_score',
        '_ptg_mean':   'kpi_avg_ptg',
        '_wptg_mean':  'kpi_avg_wptg',
    })

    if '_month' in all_kpi.columns and '_ptg' in all_kpi.columns:
        latest_ptg = (
            all_kpi.dropna(subset=['_month', '_ptg'])
            .sort_values('_month')
            .groupby('_adv', sort=False)['_ptg'].last()
            .rename('kpi_latest_ptg')
        )
        result = result.join(latest_ptg, how='left')

    # Composite wPTG for latest month — closest proxy to the STAR-determining composite PTG
    if '_month' in all_kpi.columns and '_wptg' in all_kpi.columns:
        latest_month_per_adv = (
            all_kpi.dropna(subset=['_month'])
            .groupby('_adv', sort=False)['_month'].max()
        )
        all_kpi2 = all_kpi.join(latest_month_per_adv.rename('_latest_m'), on='_adv')
        on_latest = all_kpi2[all_kpi2['_month'] == all_kpi2['_latest_m']].dropna(subset=['_wptg'])
        composite = (
            on_latest.groupby('_adv', sort=False)['_wptg']
            .sum()
            .rename('kpi_composite_wptg_latest')
        )
        result = result.join(composite, how='left')

    if '_ptg' in all_kpi.columns:
        ptg_v = all_kpi.dropna(subset=['_ptg'])
        above = ptg_v[ptg_v['_ptg'] >= 100].groupby('_adv', sort=False).size()
        total = ptg_v.groupby('_adv', sort=False).size()
        result = result.join((above / total).rename('kpi_pct_above_target'), how='left')

    if '_month' in all_kpi.columns:
        n_months = (all_kpi.dropna(subset=['_month'])
                    .groupby('_adv', sort=False)['_month'].nunique()
                    .rename('n_kpi_months'))
        result = result.join(n_months, how='left')

    n_metrics_col = all_kpi.groupby('_adv', sort=False)['_metric'].nunique().rename('n_kpi_metrics')
    result = result.join(n_metrics_col, how='left')

    # ── Per-metric pivot for top-N most frequent metrics ──────────────────────
    # STAR is mechanically composed from individual KPI PTG scores — per-metric
    # features give the model direct signal. We cap at TOP_N to avoid OOM.
    TOP_N = 30
    top_metrics = all_kpi['_metric'].value_counts().head(TOP_N).index.tolist()
    log.info(f"  Per-metric pivot: top {TOP_N} of {all_kpi['_metric'].nunique():,} unique metrics")

    metric_sub = all_kpi[all_kpi['_metric'].isin(top_metrics)].copy()
    for metric in top_metrics:
        safe = str(metric).replace(' ', '_').replace('-', '_').replace('/', '_').replace('.', '_')
        m_grp = metric_sub[metric_sub['_metric'] == metric].groupby('_adv', sort=False)
        for src, suffix in [('_ptg', 'ptg_avg'), ('_wptg', 'wptg_avg'), ('_score', 'score_avg')]:
            if src in metric_sub.columns:
                col = m_grp[src].mean().rename(f'{safe}_{suffix}')
                result = result.join(col, how='left')

    result = result.reset_index().rename(columns={'_adv': 'Advisor ID'})
    log.info(f"  KPI pivot: {len(result):,} advisors × {len(result.columns)} features")
    return result


def _aggregate_kpis_iterrows(kpi_path, existing_accum=None):
    """Fallback: slower iterrows-based aggregation."""
    accum = existing_accum or {}
    total = 0
    reader = pd.read_csv(
        kpi_path, chunksize=KPI_CHUNK_SIZE, dtype=str,
        low_memory=False, encoding='utf-8-sig',
    )
    for chunk in reader:
        chunk = _norm_headers(chunk)
        chunk = chunk.replace({'': np.nan, '\xa0': np.nan})
        adv_col    = _find_col(chunk, ['Advisor ID', 'advisor id'])
        metric_col = _find_col(chunk, ['Metric', 'metric'])
        score_col  = _find_col(chunk, ['Score', 'score'])
        ptg_col    = _find_col(chunk, ['PTG', 'ptg'])
        wptg_col   = _find_col(chunk, ['Weighted PTG', 'weighted ptg'])
        month_col  = _find_col(chunk, ['Month year', 'month year'])
        if not adv_col or not metric_col:
            continue
        total += len(chunk)
        for _, row in chunk.iterrows():
            adv_id = str(row.get(adv_col, '')).strip()
            if not adv_id or adv_id.lower() == 'nan':
                continue
            metric = str(row.get(metric_col, 'Unknown')).strip()
            score  = _safe_float(row.get(score_col))
            ptg    = _safe_float(row.get(ptg_col))
            wptg   = _safe_float(row.get(wptg_col))
            month  = str(row.get(month_col, '')).strip() if month_col else ''
            if adv_id not in accum:
                accum[adv_id] = {}
            if metric not in accum[adv_id]:
                accum[adv_id][metric] = []
            accum[adv_id][metric].append((month, score, ptg, wptg))
    log.info(f"  KPI (iterrows): {total:,} rows → {len(accum):,} advisors")
    return _build_kpi_pivot(accum)


def _build_kpi_pivot(accum: dict) -> pd.DataFrame:
    """Turn accum dict into per-advisor feature DataFrame."""
    rows = []
    for adv_id, metric_dict in accum.items():
        row = {'Advisor ID': adv_id}
        all_months: set = set()
        for metric, entries in metric_dict.items():
            safe = metric.replace(' ', '_').replace('-', '_').replace('/', '_')
            scores = [e[1] for e in entries if e[1] is not None]
            ptgs   = [e[2] for e in entries if e[2] is not None]
            wptgs  = [e[3] for e in entries if e[3] is not None]
            months = [e[0] for e in entries if e[0]]
            all_months.update(months)

            row[f'{safe}_score_avg']    = np.nanmean(scores) if scores else np.nan
            row[f'{safe}_score_latest'] = scores[-1] if scores else np.nan
            row[f'{safe}_ptg_avg']      = np.nanmean(ptgs)   if ptgs   else np.nan
            row[f'{safe}_ptg_latest']   = ptgs[-1]   if ptgs   else np.nan
            row[f'{safe}_wptg_avg']     = np.nanmean(wptgs)  if wptgs  else np.nan
            row[f'{safe}_wptg_latest']  = wptgs[-1]  if wptgs  else np.nan

        row['n_kpi_months'] = len(all_months)
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"  KPI pivot: {len(df):,} advisors × {len(df.columns)} features")
    return df


# ── Scorecard (current_data) aggregation ─────────────────────────────────────

def aggregate_scorecard(scorecard_path: Path = SCORECARD_PATH) -> pd.DataFrame:
    """
    Stream current_data in chunks.
    Returns per-email DataFrame with columns:
      sc_avg_ptg, sc_latest_ptg
      sc_avg_rank, sc_best_rank, sc_avg_quartile
      sc_max_consec_failing, sc_total_met, sc_total_fail
      sc_pct_kpi_passed, sc_n_records
    """
    log.info(f"[joins] Streaming scorecard: {scorecard_path}")

    accum: dict = {}   # email_lower -> dict of lists
    total_rows = 0
    chunks_read = 0

    reader = pd.read_csv(
        scorecard_path,
        chunksize=SCORECARD_CHUNK_SIZE,
        dtype=str,
        low_memory=False,
        encoding='utf-8-sig',
        on_bad_lines='skip',
    )
    for chunk in reader:
        chunk = _norm_headers(chunk)
        chunk = chunk.replace({'': np.nan, '\xa0': np.nan})

        email_col  = _find_col(chunk, [SCORECARD_EMAIL, 'Email Address', 'email address', 'email'])
        ptg_col    = _find_col(chunk, [SCORECARD_PTG, 'PTG', 'ptg'])
        kpi_col    = _find_col(chunk, [SCORECARD_KPI_PASSED, 'KPI Passed', 'kpi passed'])
        consec_col = _find_col(chunk, [SCORECARD_CONSEC_FAIL, 'Consecutive Failing Metrics'])
        met_col    = _find_col(chunk, [SCORECARD_MET, 'Met Count', 'met count'])
        fail_col   = _find_col(chunk, [SCORECARD_FAIL, 'Fail Count', 'fail count'])
        rank_col   = _find_col(chunk, [SCORECARD_RANK, 'Rank', 'rank'])
        quart_col  = _find_col(chunk, [SCORECARD_QUARTILE, 'Quartile', 'quartile'])

        if not email_col:
            chunks_read += 1
            continue

        total_rows += len(chunk)
        chunks_read += 1
        if chunks_read % 20 == 0:
            log.info(f"  ... Scorecard chunk {chunks_read}, {total_rows:,} rows, {len(accum):,} agents")

        for _, row in chunk.iterrows():
            key = str(row.get(email_col, '')).strip().lower()
            if not key or key == 'nan':
                continue

            if key not in accum:
                accum[key] = {
                    'ptgs': [], 'ranks': [], 'quartiles': [],
                    'consec': [], 'met': [], 'fail': [], 'kpi_passed': [],
                }

            if ptg_col:    accum[key]['ptgs'].append(_safe_float(row.get(ptg_col)))
            if rank_col:   accum[key]['ranks'].append(_safe_float(row.get(rank_col)))
            if quart_col:  accum[key]['quartiles'].append(_safe_float(row.get(quart_col)))
            if consec_col: accum[key]['consec'].append(_safe_float(row.get(consec_col)))
            if met_col:    accum[key]['met'].append(_safe_float(row.get(met_col)))
            if fail_col:   accum[key]['fail'].append(_safe_float(row.get(fail_col)))
            if kpi_col:
                v = str(row.get(kpi_col, '')).strip().upper()
                if v in ('Y', 'YES', '1', 'TRUE'):
                    accum[key]['kpi_passed'].append(1)
                elif v in ('N', 'NO', '0', 'FALSE'):
                    accum[key]['kpi_passed'].append(0)

    log.info(f"  Scorecard: {total_rows:,} rows → {len(accum):,} unique agents")

    rows = []
    for email, d in accum.items():
        def _clean(lst): return [x for x in lst if x is not None]
        ptgs   = _clean(d['ptgs'])
        ranks  = _clean(d['ranks'])
        quarts = _clean(d['quartiles'])
        consec = _clean(d['consec'])
        met    = _clean(d['met'])
        fail   = _clean(d['fail'])
        kpi_p  = d['kpi_passed']

        r = {
            'email_lower':           email,
            'sc_avg_ptg':            np.nanmean(ptgs)   if ptgs   else np.nan,
            'sc_latest_ptg':         ptgs[-1]            if ptgs   else np.nan,
            'sc_avg_rank':           np.nanmean(ranks)  if ranks  else np.nan,
            'sc_best_rank':          np.nanmin(ranks)   if ranks  else np.nan,
            'sc_avg_quartile':       np.nanmean(quarts) if quarts else np.nan,
            'sc_max_consec_failing': np.nanmax(consec)  if consec else np.nan,
            'sc_total_met':          np.nansum(met)     if met    else np.nan,
            'sc_total_fail':         np.nansum(fail)    if fail   else np.nan,
            'sc_pct_kpi_passed':     np.mean(kpi_p)     if kpi_p  else np.nan,
            'sc_n_records':          len(d['ptgs']),
        }
        rows.append(r)

    df = pd.DataFrame(rows)
    log.info(f"  Scorecard pivot: {len(df):,} agents × {len(df.columns)} features")
    return df


# ── Merge helpers ─────────────────────────────────────────────────────────────

def merge_kpis_to_roster(roster: pd.DataFrame, kpi_df: pd.DataFrame,
                          emp_col: str) -> pd.DataFrame:
    """Left-join KPI aggregates onto roster by employee number."""
    if kpi_df.empty or 'Advisor ID' not in kpi_df.columns:
        log.warning("  KPI df empty or missing 'Advisor ID' — skipping KPI join")
        return roster
    kpi = kpi_df.copy()
    kpi['_k'] = kpi['Advisor ID'].astype(str).str.strip()
    r = roster.copy()
    r['_k'] = r[emp_col].astype(str).str.strip()

    merged = r.merge(kpi.drop(columns=['Advisor ID']), on='_k', how='left').drop(columns=['_k'])

    n_joined = int(merged['n_kpi_months'].notna().sum())
    pct      = n_joined / len(merged) * 100
    log.info(f"  KPI join coverage: {n_joined:,}/{len(merged):,} ({pct:.1f}%)")
    return merged


def merge_scorecard_to_roster(roster: pd.DataFrame, sc_df: pd.DataFrame,
                               email_col: str) -> pd.DataFrame:
    """Left-join scorecard aggregates onto roster by email (lowercased)."""
    sc = sc_df.copy()
    r  = roster.copy()
    r['_ek'] = r[email_col].astype(str).str.strip().str.lower()

    merged = r.merge(sc.rename(columns={'email_lower': '_ek'}), on='_ek', how='left').drop(columns=['_ek'])

    n_joined = int(merged['sc_avg_ptg'].notna().sum())
    pct      = n_joined / len(merged) * 100
    log.info(f"  Scorecard join coverage: {n_joined:,}/{len(merged):,} ({pct:.1f}%)")
    return merged
