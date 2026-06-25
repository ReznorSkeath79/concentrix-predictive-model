# KPI Tier Prediction Model (Model KPI-A) — Metrics Report

> Generated: 2026-06-25 | Target: Latest Weighted PTG → 3-class tercile tiers

---

## Data Summary

| Field | Value |
|-------|-------|
| Total CDM agents | 65,039 |
| Labeled (PTG non-null) | 41,717 (64.1%) |
| Tercile p33 cutpoint | 91.6565 |
| Tercile p67 cutpoint | 102.9797 |
| At-Risk agents (tier 0) | 13,906 |
| Developing agents (tier 1) | 13,905 |
| High Performer agents (tier 2) | 13,906 |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test samples | 8,344 |
| **Accuracy** | 🔴 **58.3%** |
| Majority baseline accuracy | 33.3% |
| Lift vs majority baseline | +25.0% |
| Macro F1 | 0.5820 |
| AUC (OvR macro) | 0.7624 |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| At-Risk | 0.622 | 0.618 | 0.620 | 2,782 |
| Developing | 0.521 | 0.496 | 0.508 | 2,781 |
| High Performer | 0.602 | 0.635 | 0.618 | 2,781 |

---

## Phase-2 Readiness

- Artifact: `models/model_kpi_a.joblib` ✅
- Schema: `models/schema_kpi.json` ✅
- 3-class output: At-Risk | Developing | High Performer ✅
- Tercile cutpoints computed from actual PTG distribution (no forced curve) ✅
