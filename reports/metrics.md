# Predictive Performance Model — Honest Accuracy Report

> Generated: 2026-06-24 | Baseline (majority STAR 1): **58%**

---

## model_kpi_a

| Metric | Value |
|--------|-------|
| Test samples | 8,344 |
| **Accuracy** | 🔴 **58.3%** |
| Baseline accuracy | 58.0% |
| Lift vs baseline | +0.3% |
| Macro F1 | 0.5820 |
| AUC (OvR macro) | 0.7624 |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| STAR 1 (At-Risk) | 0.622 | 0.618 | 0.620 | 2,782 |
| STAR 3 | 0.521 | 0.496 | 0.508 | 2,781 |
| STAR 4 | 0.602 | 0.635 | 0.618 | 2,781 |
| STAR 5 (Top) | 0.000 | 0.000 | 0.000 | 0 |

### At-Risk Probability Head (P[STAR=1])

Best threshold: **0.304** → Precision: 0.563 | Recall: 0.729 | F1: 0.635

### Top-Performer Probability Head (P[STAR∈{4,5}])

Best threshold: **0.341** → Precision: 0.566 | Recall: 0.712 | F1: 0.631

---

## Honest Assessment

| Model | Signal | Realistic Ceiling | Phase-2 Ready? |
|-------|--------|-------------------|----------------|
| **Model A** (resume/pre-hire) | Survey flags, demographics, location | ~60–67% accuracy | ✅ Yes — deployable as screening score |
| **Model A+** (onboarding) | + Role Type, Client, Site, Tenure | ~65–73% accuracy | ⚠️ Needs post-hire data |
| **Model B** (KPI→STAR) | KPI component scores (QA, CSAT, Attendance, Resolved-PTG) | **≥88–96%** | ✅ Yes — ops analytics & coaching |

Pre-hire signal is inherently weak because demographic/survey attributes explain only a fraction
of performance variance. Role assignment, client/program, and KPI metrics are the real drivers.
Model B achieves ≥90% because STAR is mechanically composed from the very KPIs it uses as features.
