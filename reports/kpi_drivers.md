# Model B — KPI Driver Report

Ranked by mean |SHAP| value (impact on STAR prediction across all classes).

| Rank | Feature | Mean |SHAP| | Notes |
|------|---------|------|-------|
| 1 | `Activity` | 0.4695 |  |
| 2 | `tenure_months` | 0.1766 | HR — tenure |
| 3 | `n_kpi_months` | 0.1510 |  |
| 4 | `Program` | 0.1438 |  |
| 5 | `CSAT_ptg_avg` | 0.1431 | Raw PTG |
| 6 | `n_kpi_metrics` | 0.1242 |  |
| 7 | `City` | 0.1150 |  |
| 8 | `Client` | 0.1034 |  |
| 9 | `Person Status` | 0.0814 |  |
| 10 | `CSAT_score_avg` | 0.0804 | Customer Satisfaction |
| 11 | `CSAT_wptg_avg` | 0.0748 | Weighted PTG component |
| 12 | `Site` | 0.0670 |  |
| 13 | `Job Title` | 0.0653 |  |
| 14 | `Campus` | 0.0446 |  |
| 15 | `Support Type` | 0.0393 |  |
| 16 | `Management Level Description` | 0.0259 |  |
| 17 | `Role Type` | 0.0246 |  |
| 18 | `tenure_bucket` | 0.0217 | HR — tenure |
| 19 | `QA_ptg_avg` | 0.0192 | Raw PTG |
| 20 | `QA_score_avg` | 0.0187 | Quality Assurance |
| 21 | `sc_avg_rank` | 0.0162 | Scorecard aggregate |
| 22 | `State` | 0.0153 |  |
| 23 | `sc_avg_ptg` | 0.0145 | Raw PTG |
| 24 | `sc_total_met` | 0.0139 | Scorecard aggregate |
| 25 | `sc_total_fail` | 0.0138 | Scorecard aggregate |
| 26 | `QA_wptg_avg` | 0.0128 | Weighted PTG component |
| 27 | `sc_n_records` | 0.0116 | Scorecard aggregate |
| 28 | `sc_best_rank` | 0.0107 | Scorecard aggregate |
| 29 | `sc_latest_ptg` | 0.0072 | Raw PTG |
| 30 | `sc_max_consec_failing` | 0.0006 | Scorecard aggregate |