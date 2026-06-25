# Lessons Learned

---

## 2026-06-25 — Inline 14.7MB HTML causes browser "Initialising" hang; initSimulator() never fires

**What:** Added a SHAP-powered feature weight simulator to executive_summary.html. Data was embedded inline as a `<script>window.MODEL_DATA = {...7.3MB JSON...}</script>`. Browser opens the file but simulator stays stuck on "Initialising…" indefinitely.

**Root cause:** Two suspected causes (unconfirmed — needs browser DevTools on next session):
1. Parsing a 14.7MB HTML file with a 7.3MB inline JSON object blocks the main thread long enough that the `window.load` event fires but `window.MODEL_DATA` may not yet be hydrated (race between script parse order and event listener)
2. A silent JS error in `initSimulator()` or in the 15,549-sample array construction — no error surface because onerror only applies to external `<script src>`

**Rule:** Never embed >1MB JSON inline in HTML for a local file:// use case. Instead: (a) write data as a proper .js file and open the HTML via a local dev server (`python -m http.server 8080`), OR (b) split data into chunks and load lazily, OR (c) keep data external and instruct user to run `python -m http.server` first.

**Verification pattern:**
```
# Serve locally to avoid file:// issues:
cd E:\work\concentrix\predictivemodel\reports
python -m http.server 8080
# Then open: http://localhost:8080/executive_summary.html
# Check DevTools Console for errors
```

**Cost:** ~30 min — two build iterations + export run

---

## 2026-06-24 — pandas itertuples silently renames space-containing columns to `_0, _1...`

**What:** `aggregate_kpis` in `joins.py` streamed 4.3M rows and accumulated 0 unique advisors despite finding the 'Advisor ID' column. No exception was raised — the fallback never triggered.

**Root cause:** `pandas.DataFrame.itertuples()` renames any column whose name fails `str.isidentifier()` to `_N` (positional index). `'Advisor ID'` contains a space → fails `isidentifier()` → becomes `_0`. The code used `getattr(row, adv_col.replace(' ', '_'), '')` expecting `'Advisor_ID'`, but the actual field was `'_0'`. `getattr` returned the default `''` for every row, all rows were silently skipped.

**Rule:** Never use `itertuples` + `getattr(row, col.replace(' ','_'))` for column access. Column names with spaces, dots, hyphens, or digits at the start all get positional aliases. Use `iterrows()` (safe but slow) or vectorized groupby/column selection (preferred).

**Verification pattern:**
```python
df = pd.DataFrame({'Advisor ID': ['x']})
row = next(df.itertuples(index=False))
print(row._fields)  # ('_0',) — NOT ('Advisor_ID',)
print(getattr(row, 'Advisor_ID', 'MISSING'))  # 'MISSING'
```

**Cost:** ~3 debug cycles, ~45 min total.

---

## 2026-06-24 — KPI pivot OOM: per-metric columns explode to N_metrics × 6 (27K cols)

**What:** After fixing the itertuples bug, the KPI pivot attempted to create a DataFrame with shape (117,609 advisors × 27,032 metric-feature columns) — 23.7 GiB of object arrays. OOM killed the process.

**Root cause:** The KPI file has 4,513 unique metric names (client-specific variants, historical names, composite labels). Per-metric pivoting at 6 features/metric = 27K columns. The data has fragmented metric naming — no canonical metric taxonomy.

**Rule:** Always check `nunique()` on the groupby key before building a wide pivot. If N_unique > 100, either: (a) cap at top-N by frequency, or (b) aggregate to overall statistics. For KPI-style data, top-30 by row count captures the operationally significant metrics.

**Verification pattern:**
```python
n = df['_metric'].nunique()
print(f"{n:,} unique metrics → {n * 6:,} columns if fully pivoted")
# If > 500: use top-N approach
```

**Cost:** 1 failed run + ~20 min re-run time.

---

## 2026-06-24 — evaluate.py f-string conditional inside format spec

**What:** `f"AUC={auc:.4f if auc else 'N/A'}"` raised `ValueError: Invalid format specifier`. Everything after `:` inside `{}` is treated as a format spec — Python doesn't evaluate conditionals there.

**Root cause:** Python f-string grammar: `{expr:format_spec}`. The `if` is part of `format_spec`, not `expr`. This is a syntax-level constraint, not a runtime issue.

**Rule:** Never embed conditionals inside the format spec slot. Compute the display string first: `auc_str = f'{auc:.4f}' if auc else 'N/A'`, then `f"AUC={auc_str}"`.

**Verification pattern:** `python -c "auc=0.75; print(f'{auc:.4f if auc else \"N/A\"}')"` → ValueError immediately.

**Cost:** 1 failed run (caught immediately from traceback).

---

## 2026-06-24 — SHAP returns 3D array (n_samples, n_features, n_classes) in newer versions

**What:** `KPI driver report failed: Per-column arrays must each be 1-dimensional`. The SHAP report code averaged over axis=0 expecting shape `(n_samples, n_features)` but got `(n_features, n_classes)` for `mean_abs`.

**Root cause:** Newer SHAP (`>=0.43`) returns a 3D numpy array `(n_samples, n_features, n_classes)` for multiclass TreeExplainer instead of the old list-of-2D-arrays format. The code only handled `isinstance(shap_values, list)` (old format) and the else branch did `sv = np.abs(shap_values)` keeping the 3D shape intact.

**Rule:** Always handle both SHAP output shapes for multiclass:
```python
if isinstance(shap_values, list):
    sv = np.abs(np.array(shap_values)).mean(axis=0)   # old: (n_cls, n_samp, n_feat) → (n_samp, n_feat)
elif np.array(shap_values).ndim == 3:
    sv = np.abs(shap_values).mean(axis=2)              # new: (n_samp, n_feat, n_cls) → (n_samp, n_feat)
else:
    sv = np.abs(shap_values)
```

**Cost:** 1 warning-only failure (model still trained), ~10 min debug.

---

## 2026-06-24 — Model B ≥88% accuracy requires period-aligned KPI data

**What:** Model B achieved 58.7% accuracy / AUC 0.789, not the ≥88% target. All KPI features are time-averaged aggregates; STAR is assigned at a specific evaluation period.

**Root cause:** The KPI file contains all-time historical KPI rows. Averaging over all months dilutes the signal — the model cannot distinguish an advisor's current performance tier from their historical average. `Latest Weighted PTG` in CDM is the true composite that determines STAR but is rightfully excluded as leakage.

**Rule:** For STAR prediction from KPIs, filter KPI rows to the evaluation period matching each STAR label before aggregating. This requires a timestamp join between the STAR rating date and the KPI month. With that join, the model effectively learns to replicate the STAR formula → ≥88% becomes achievable.

**Verification pattern:** Check if `Average PTG Last 3 Months` from CDM used as a (non-leakage) proxy for the evaluation-period signal improves accuracy significantly. If yes, the temporal alignment is the bottleneck.

**Cost:** Multiple re-runs (~1.5 hrs total compute), model performs but at 58.7% vs 88% target.
