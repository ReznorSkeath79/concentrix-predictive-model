# Predictive Performance Model — Honest Accuracy Report

> Generated: 2026-06-24 | Baseline (majority STAR 1): **58%**

---

## Data Join Coverage

| Source | Total Agents | Joined | % |
|--------|-------------|--------|---|
| CNX KPIs | 65,039 | 44,519 | 68.4% |
| Scorecard | 65,039 | 4,029 | 6.2% |

---

## model_b

| Metric | Value |
|--------|-------|
| Test samples | 8,423 |
| **Accuracy** | 🔴 **58.7%** |
| Baseline accuracy | 58.0% |
| Lift vs baseline | +0.7% |
| Macro F1 | 0.4839 |
| AUC (OvR macro) | 0.7884 |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| STAR 1 (At-Risk) | 0.807 | 0.692 | 0.745 | 4,898 |
| STAR 3 | 0.405 | 0.460 | 0.430 | 1,793 |
| STAR 4 | 0.317 | 0.363 | 0.339 | 1,087 |
| STAR 5 (Top) | 0.355 | 0.518 | 0.421 | 645 |

### At-Risk Probability Head (P[STAR=1])

Best threshold: **0.140** → Precision: 0.703 | Recall: 0.915 | F1: 0.795

### Top-Performer Probability Head (P[STAR∈{4,5}])

Best threshold: **0.438** → Precision: 0.463 | Recall: 0.650 | F1: 0.541

---

## Honest Assessment

| Model | Accuracy | AUC | Phase-2 Ready? |
|-------|----------|-----|----------------|
| **Model A** (resume/pre-hire, 107 features) | 47.6% | 0.701 | ✅ Use AUC/risk score, not raw accuracy |
| **Model A+** (onboarding, 122 features) | 50.9% | 0.723 | ✅ Best pre-hire model |
| **Model B** (KPI→STAR, 32 features) | **58.7%** | **0.788** | ✅ Ops analytics & coaching |

### Why Model B raw accuracy is near baseline

Raw accuracy of 58.7% reflects majority-class dominance (STAR 1 = 58% of labeled data). The
**AUC of 0.789** is the right signal: the model correctly ranks 78.9% of advisor pairs by
predicted STAR — strong enough for risk triage, coaching prioritization, and ops dashboards.

### Why ≥88% was not achieved

The original ≥88% target assumed **period-aligned KPI component scores** (the exact QA, CSAT,
Attendance, Weighted PTG readings from the evaluation period that produced each STAR rating).
What the KPI file provides is **all-time per-metric averages** across every month on record.
Averaging over time dilutes the signal: an advisor who performed well 18 months ago but is
currently at-risk looks the same as a consistently average advisor.

`Latest Weighted PTG` in CDM is the true composite that mechanically determines STAR — but
using it as a feature is circular (it encodes STAR directly, not a predictor of it).

**To reach ≥88%**, the next iteration needs period-matched KPI data: filter KPI rows to the
evaluation month/quarter for each STAR label. That join is possible but requires exposing the
evaluation period timestamp from the STAR rating process.
