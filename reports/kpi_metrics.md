# KPI Tier Prediction Model (Model KPI-A) — Metrics Report

> Generated: 2026-06-25 | Target: Latest Weighted PTG → 3-class tercile tiers

---

## Data Summary

| Field | Value |
|-------|-------|
| Total CDM agents | 58,140 |
| Labeled (PTG non-null) | 40,919 (70.4%) |
| Tercile p33 cutpoint | 91.6147 |
| Tercile p67 cutpoint | 102.9463 |
| At-Risk agents (tier 0) | 13,640 |
| Developing agents (tier 1) | 13,639 |
| High Performer agents (tier 2) | 13,640 |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test samples | 8,184 |
| **Accuracy** | 🔴 **60.5%** |
| Majority baseline accuracy | 33.3% |
| Lift vs majority baseline | +27.1% |
| Macro F1 | 0.6032 |
| AUC (OvR macro) | 0.7893 |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| At-Risk | 0.656 | 0.647 | 0.651 | 2,728 |
| Developing | 0.540 | 0.497 | 0.518 | 2,728 |
| High Performer | 0.613 | 0.670 | 0.641 | 2,728 |

---

## Phase-2 Readiness

- Artifact: `models/model_kpi_a.joblib` ✅
- Schema: `models/schema_kpi.json` ✅
- 3-class output: At-Risk | Developing | High Performer ✅
- Tercile cutpoints computed from actual PTG distribution (no forced curve) ✅
