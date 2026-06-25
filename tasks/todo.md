# Predictive Performance Model — Task Tracker

## Phase 1: Build the Model

### Setup
- [x] Plan approved (glimmering-floating-reddy.md)
- [x] Create project structure (src/, models/, reports/, tasks/)
- [x] `pip install -r requirements.txt`

### Source Files
- [x] Write `requirements.txt`
- [x] Write `src/config.py` — paths, column groups, leakage lists, target map
- [x] Write `src/features.py` — cleaning, feature engineering, fold-safe encoding
- [x] Write `src/joins.py` — chunked KPI + scorecard aggregation
- [x] Write `src/evaluate.py` — metrics, SHAP, confusion matrix, reports/metrics.md
- [x] Write `src/train_model_a.py` — Model A (pre-hire) + A+ (onboarding) pipeline
- [x] Write `src/train_model_b.py` — Model B (KPI→STAR, ≥90% target)

### Execution
- [x] Run `python src/train_model_a.py` — clean run, reports generated
- [x] Run `python src/train_model_b.py` — clean run, Model B trained
- [x] Verify reports/metrics.md generated with honest accuracy vs 58% baseline
- [x] Verify SHAP plots in reports/
- [x] Phase-2 readiness: load model_a.joblib + schema_a.json, feed example → prediction ✅ PASS

### Documentation
- [ ] Write README.md with join coverage stats and honest accuracy contract

---

## Review / Results

### Model A (pre-hire only, 107 features)
- Accuracy: 47.6% | AUC: 0.701 | Macro-F1: 0.404
- Baseline: 54.25% — AUC is the operative metric (raw accuracy suppressed by majority class)
- Phase-2 readiness: PASS (example prediction returned STAR=3, at-risk/top-perf probs)

### Model A+ (pre-hire + onboarding, 122 features)
- Accuracy: 50.9% | AUC: 0.723 | Macro-F1: 0.432
- Best pre-hire variant — adds Client, Role Type, Site, Tenure

### Model B (KPI → STAR, 32 features)
- Accuracy: 58.7% | AUC: 0.789 | Macro-F1: 0.484
- KPI join: 44,519/65,039 agents (68.4% coverage) | Scorecard: 4,029/65,039 (6.2%)
- Did NOT hit ≥88% target — root cause documented in reports/metrics.md
  (requires period-aligned KPI data, not all-time averages)
- At-risk threshold F1: 0.795 @ P=0.140 — usable for ops triage

### Bugs fixed during execution
1. `evaluate.py:82` — f-string format spec with conditional inside `:` (invalid syntax)
2. `joins.py` — `itertuples` renames space-containing columns to `_0, _1...` silently
3. `joins.py` — KPI pivot OOM (117K × 27K object array = 23.7 GiB); switched to top-30 + overall aggregates
4. `joins.py` — scorecard CSV malformed line 5 (56 fields, expected 55); fixed with `on_bad_lines='skip'`
5. `train_model_b.py` SHAP — newer SHAP returns 3D array `(n_samples, n_features, n_classes)`; code expected list/2D

---

### Demographic Experiments (2026-06-24)
- EXP 1 (Pure Demographics, 17 features): Accuracy=0.4964, AUC=0.7375
- EXP 2 (Full CDM Org, 220 features): Accuracy=0.5350, AUC=0.7632 — best pre-hire alternative
- EXP 3 LightGBM (107 features): Accuracy=0.4946, AUC=0.7320
- EXP 3 LogReg (107 features): Accuracy=0.3348, AUC=0.5752 — confirms non-linear interactions dominate
- Key finding: 17 core demographics outperform 107 survey flags; program placement is dominant predictor

### Executive Report (2026-06-25)
- [x] Built `reports/executive_summary.html` — full one-page non-technical exec brief
- [x] Reframed objective: "what pre-hire factors predict STAR/KPI?" (not "can we predict STAR")
- [x] Clarified STAR = KPI (STAR is derived from Weighted PTG composite of KPI components)
- [x] Added flow diagram: who you are → work background → placement → STAR/KPI
- [x] Added 6 key predictor signal cards (career track, certifications, WFH, job grade, vertical, placement)
- [x] Built `src/export_for_viz.py` — exports SHAP group contributions + predictions to model_viz_data.js
- [x] Built `src/build_report.py` — embeds 7.3MB data inline for file:// compatibility
- [ ] **BUG OPEN**: Simulator section shows "Initialising…" indefinitely — browser not executing initSimulator()
  - Likely cause: 14.7MB inline JSON parse time OR JS error in initSimulator()
  - Next step: open browser DevTools console → check for errors, verify window.MODEL_DATA is populated

---

## Phase 2 (Future)
- Upload site: form → schema_a.json → model_a.joblib → STAR + risk/top scores
- Stack: Streamlit or FastAPI+React (decide when Phase 1 complete)
- Model B improvement: filter KPI rows to evaluation period matching each STAR label → should unlock ≥88%
